from ai2_thor_model_training import RobotNavigationControl
from ai2_thor_model_training import AI2THORUtils
from thortils import launch_controller
from thortils.utils.math import sep_spatial_sample
import thortils as tt
import random, cv2
from PIL import Image
import copy
import matplotlib.pyplot as plt

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
        seed = 1983
        self.rnd = random.Random(seed)
        self.h_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.placements = []

        # If we want to interrupt the work of current SNP, then we'll set this to True
        self.interrupt = False
        self.intrp_callback_function = None

        # Matplotlib stuff
        plt.ion()
        self.fig = None
        self.ax = None
        self.im = None

    def process_required_habitats(self):
        self.process_habitat(10)
        self.controller.stop()

    def set_action_gen(self, action_generator):
        self.action_generator = action_generator

    ##
    # Process the given habitat- load it, put agent in random places and navigate from those places to some set goal.
    ##
    def open_habitat(self, habitat_id):
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

    # def show_top_down_map(self):
    #     grid_size = self.controller.initialization_parameters["gridSize"]
    #
    #     # reachable_positions = [
    #     #     tuple(map(lambda x: roundany(x, grid_size), pos))
    #     #     for pos in thor_reachable_positions(self.controller)]
    #
    #     x_max = max([pos[0] for pos in self.reachable_positions])
    #     z_max = max([pos[1] for pos in self.reachable_positions])
    #     x_min = min([pos[0] for pos in self.reachable_positions])
    #     z_min = min([pos[1] for pos in self.reachable_positions])
    #
    #     fig, ax = plt.subplots()
    #
    #     # setting up for the top-down picture of the habitat
    #     print(str(x_min-grid_size) + " " + str(x_max+grid_size) + " " + str(z_min-grid_size) + " " + str(z_max+grid_size))
    #     img = self.get_top_down_frame()
    #     ex_mul = 7
    #     ax.imshow(img, extent=[x_min-ex_mul*grid_size, x_max+ex_mul*grid_size, z_min-ex_mul*grid_size, z_max+ex_mul*grid_size])
    #
    #     plt.show()
    #     plt.close()

    def show_top_down_map(self):
        grid_size = self.controller.initialization_parameters["gridSize"]

        x_max = max([pos[0] for pos in self.reachable_positions])
        z_max = max([pos[1] for pos in self.reachable_positions])
        x_min = min([pos[0] for pos in self.reachable_positions])
        z_min = min([pos[1] for pos in self.reachable_positions])

        ex_mul = 7
        extent = [x_min - ex_mul * grid_size, x_max + ex_mul * grid_size,
                  z_min - ex_mul * grid_size, z_max + ex_mul * grid_size]

        # Create figure only ONCE
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            img = self.get_top_down_frame()
            self.im = self.ax.imshow(img, extent=extent)
            self.ax.set_title("Top-Down View")
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking show
        else:
            # Clear and redraw (or update existing artist)
            self.ax.clear()
            img = self.get_top_down_frame()
            self.ax.imshow(img, extent=extent)

            self.ax.set_title("Top-Down View")

        # Draw and pause briefly to update
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to let GUI process events

    def generate_placements(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        num_stops = 20
        sep = 1.0
        v_angles = [30]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have

        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        self.reachable_positions = tt.thor_reachable_positions(self.controller)
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops,
                                        rnd=self.rnd)
        self.placements = placements
        return placements

    def load_next_placement(self):
        print("placemnts left: ", len(self.placements))
        if len(self.placements) > 1:
            p = self.placements.pop()
            # append a rotation to the place.
            yaw = self.rnd.sample(self.h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)
        else:
            raise ValueError("No placements left")

    ##
    # Process the given habitat- load it, put agent in random places and navigate from those places to some set goal.
    ##
    def process_habitat(self, habitat_id):
        # load required habitat
        self.open_habitat(habitat_id)
        self.process_random_placements_in_habitat()

    ##
    # Here we will select a number of random placements and then attempt to navigate from each of them
    # to some goal.
    ##
    def process_random_placements_in_habitat(self):
        placements = self.generate_placements()
        print(placements)
        explorations_processed = 0

        while len(self.placements) > 0:
            try:
                self.load_next_placement()
                self.navigate_to_goal()
                explorations_processed += 1
            except ValueError:
                return

    ##
    # Use a neural network to navigate to the required goal.
    # For now that will be navigating to the middle of the room.
    ##
    def navigate_to_goal(self):
        next_move_str = "START"
        while next_move_str != "STOP" and not self.interrupt:
            # first get the from view image
            event = self.controller.last_event
            img = event.cv2img

            next_move_str = self.action_generator(img)
            #print(next_move_str)
            print('.', sep='', end='')

            if next_move_str == "STOP":
                continue
            elif next_move_str == "TURN":
                for i in range(4):
                    self.rnc.execute_action("RotateRight", degrees=45, moveMagnitude=self.grid_size,
                                            grid_size=self.grid_size,
                                            adhere_to_grid=True)
            elif next_move_str == "PAUSE":
                continue
            else:
                self.rnc.execute_action(next_move_str, degrees=45, moveMagnitude=self.grid_size, grid_size=self.grid_size,
                                        adhere_to_grid=True)
                self.show_top_down_map()

        # If we had to interrupt, then reset the interupt and clean up action generator
        if self.interrupt:
            self.action_generator.stop_received() # or just self.action_generator.reset() ?
            self.interrupt = False
            if self.intrp_callback_function is not None:
                self.intrp_callback_function()

    def interrupt_navigation(self, intrp_callback_function = None):
        self.interrupt = True
        self.intrp_callback_function = intrp_callback_function

    ##
    # For display purposes - the top down view of the habitat
    ##
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        if not hasattr(self, 'has_top_down_camera'):
            self.has_top_down_camera = False
            self.top_down_camera_id = 0

        if not self.has_top_down_camera:
            #print("AE: Adding camera")
            # add the camera to the scene
            event = self.controller.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )
            self.top_down_camera_id = len(event.third_party_camera_frames) - 1
            self.has_top_down_camera = True
        else:
            #print("AE: updating camera")
            # If we already have a top-down view camera, then we need to update it to suit the current habitat
            self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self.top_down_camera_id,
                **pose,
                #position=position,
                #rotation=rotation
            )

        top_down_frame = event.third_party_camera_frames[self.top_down_camera_id]

        return Image.fromarray(top_down_frame)

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