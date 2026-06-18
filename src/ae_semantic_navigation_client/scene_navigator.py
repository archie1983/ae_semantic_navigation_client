from ai2_thor_model_training import RobotNavigationControl
from ai2_thor_model_training import AI2THORUtils
from thortils import launch_controller
from thortils.utils.math import sep_spatial_sample
import thortils as tt
import random, cv2
from PIL import Image
import numpy as np

##
# This class will use one or more of our neural network models and navigate through a scene
##
class SceneNavigator:
    def __init__(self, action_generator):
        self.rnc = RobotNavigationControl()
        self.controller = None
        self.atu = AI2THORUtils()
        self.action_generator = action_generator
        self.grid_size = 0.125

    def process_required_habitats(self):
        self.process_habitat(10)
        self.controller.stop()

    def set_action_gen(self, action_generator):
        self.action_generator = action_generator

    ##
    # Process the given habitat- load it, put agent in random places and navigate from those places to some set goal.
    ##
    def process_habitat(self, habitat_id):
        # load required habitat
        habitat = self.atu.load_proctor_habitat(habitat_id)

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
            self.rnc.set_controller(self.controller) # This allows our control scripts to interact with AI2-THOR environment
        else:
            self.controller.reset(habitat)
            self.reset_state()
            self.rnc.reset_state()
            #self.rnc.set_controller(self.controller)

        self.process_random_placements_in_habitat()

    ##
    # Here we will select a number of random placements and then attempt to navigate from each of them
    # to some goal.
    ##
    def process_random_placements_in_habitat(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 5
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have

        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        rnd = random.Random(seed)

        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        reachable_positions = tt.thor_reachable_positions(self.controller)
        placements = sep_spatial_sample(reachable_positions, sep, num_stops,
                                        rnd=rnd)

        #print(placements)

        explorations_processed = 0
        for p in placements:
            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.
            self.navigate_to_goal()

            explorations_processed += 1

    ##
    # Use a neural network to navigate to the required goal.
    # For now that will be navigating to the middle of the room.
    ##
    def navigate_to_goal(self):
        next_move_str = "START"
        while next_move_str != "STOP":
            # first get the from view image
            event = self.controller.last_event
            img = event.cv2img
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_img)

            next_move_str = self.action_generator(pil_image)
            print(next_move_str)

            if next_move_str == "STOP":
                continue
            else:
                self.rnc.execute_action(next_move_str, moveMagnitude=self.grid_size, grid_size=self.grid_size,
                                        adhere_to_grid=True)

class TestActionGenerator:
    def __init__(self):
        self.moves = ["RotateLeft", "MoveAhead", "MoveAhead", "MoveAhead", "MoveAhead", "MoveAhead", "MoveAhead",
                      "RotateRight", "STOP"]
        self.action_counter = 0

    def __call__(self, pil_image):
        print(pil_image)
        next_move_str = self.moves[self.action_counter]
        self.action_counter += 1

        if self.action_counter >= len(self.moves):
            self.action_counter = 0
        return next_move_str

if __name__ == "__main__":
   sn = SceneNavigator(TestActionGenerator())
   sn.process_required_habitats()