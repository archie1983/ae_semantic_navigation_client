import zmq, glob, re
import numpy as np
import time, cv2, os
from PIL import Image
from scene_navigator import SceneNavigator
from ai2_thor_model_training import index_to_action

class ActionGenerator:
    def __init__(self, dreamer_socket):
        self._cur_obs = dict(
            pov = None,
            is_first = False,
            is_last = False,
            module = "snp",
            cmd = "genact"
        )
        self.socket = dreamer_socket
        self.reset()
        self.handshake_received = False
        self.image_receiver = None
        self.last_image_large = None

    def reset(self):
        self._cur_obs["is_first"] = True
        self._cur_obs["is_last"] = False

    def stop_received(self):
        self._cur_obs["is_first"] = False
        self._cur_obs["is_last"] = True

    def normal_op(self):
        self._cur_obs["is_first"] = False
        self._cur_obs["is_last"] = False

    def handshake(self):
        if (not self.handshake_received):
            print(f"Client sending handshake...")
            self.socket.send_pyobj({'module': 'snp', 'cmd': 'handshake'})
            #data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives
            # we want it to block here until client has connected and only then to continue on and start receiving observations

            #if (data['module'] == 'snp' and data['cmd'] == 'handshake2'):
            #    print("Handshake reply received. Now action should follow from Jetson.")

            print("Handshake sent. Now action should follow from Jetson.")
            response = self.socket.recv_pyobj()
            print("AE: rsp: ", response)
            self.reset()
        self.handshake_received = True

    def set_image_receiver(self, image_receiver):
        """
        This provides a way to gleam at the images received
        :param image_receiver:
        :return:
        """
        self.image_receiver = image_receiver

    def __call__(self, ai2_thor_image):
        self.handshake()

        # preparing 2 size images: 64x64 for DreamerV3 models and 600x600 or 640x640 whatever AI2-Thor launcher
        # is configured with for YOLO models.
        # Resize to 64 x 64
        img_64x64 = cv2.resize(
            ai2_thor_image,
            (64, 64),
            interpolation=cv2.INTER_LANCZOS4  # High quality
        )

        rgb_img_64x64 = cv2.cvtColor(img_64x64, cv2.COLOR_BGR2RGB)
        pil_image_64x64 = Image.fromarray(rgb_img_64x64)

        rgb_img_large = cv2.cvtColor(ai2_thor_image, cv2.COLOR_BGR2RGB)
        pil_image_large = Image.fromarray(rgb_img_large)

        if self.image_receiver is not None:
            self.image_receiver(pil_image_large)
            self.last_image_large = pil_image_large
        # image received, it now needs to be sent to a Dreamer model running on Jetson,
        # which will return an action. The action will then have to be returned from here
        # so that it can be executed in the simulation.
        #print(pil_image)
        img_array = np.stack([pil_image_64x64], axis=0)

        self._cur_obs["pov"] = {
            'shape': img_array.shape,
            'dtype': str(img_array.dtype),
            'bytes': img_array.tobytes(),
        }

        # Send request
        self.socket.send_pyobj(self._cur_obs)

        # Wait for response (this BLOCKS until Jetson replies)
        try:
            response = self.socket.recv_pyobj()
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            response = None

        next_move_str = index_to_action(response['action_bits']['action']) # <- this needs to talk to Jetson over ZMQ and pass it the image
        #print("ACT: ", next_move_str)
        # along with the rest of the observation.
        # next_move_str has to be returned from Dreamer running on Jetson and then if it is STOP, then we need to prepare
        # a observation with is_last = True. If however this is the very first image after loading a scene, then we
        # need to set is_first = True.

        if next_move_str == "STOP":
            self.stop_received()
        elif response['action_bits']['reset']:
            self.reset()
        else:
            self.normal_op()

        return next_move_str

class SemanticNavigationClient:
    LLM_PORT = 5555
    DR_NAV_PORT = 5556
    RC_NAV_PORT = 5557

    def __init__(self, jetson_ip):
        self.context = zmq.Context()
        # # LLM container
        self.llm_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.llm_socket.connect(f"tcp://{jetson_ip}:{self.LLM_PORT}")
        print(f"Connected to Jetson LLM container at {jetson_ip}:{self.LLM_PORT}")
        #
        # # Door navigation container
        # self.dr_socket = self.context.socket(zmq.REQ)  # REQuest socket
        # self.dr_socket.connect(f"tcp://{jetson_ip}:{self.DR_NAV_PORT}")
        # print(f"Connected to Jetson Door navigation container at {jetson_ip}:{self.DR_NAV_PORT}")

        # Room centre navigation container
        self.rc_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.rc_socket.connect(f"tcp://{jetson_ip}:{self.RC_NAV_PORT}")
        print(f"Connected to Jetson RoomCentre navigation container at {jetson_ip}:{self.RC_NAV_PORT}")

        # Local AI2-Thor simulation and action generators that talk to Dreamer models on Jetson:
        self.rc_action_gen = ActionGenerator(self.rc_socket)
        #		self.dr_action_gen = ActionGenerator(self.dr_socket)
        self.scene_navigator = SceneNavigator(self.rc_action_gen)

        # load a certain habitat
        self.scene_navigator.open_habitat(65)
        self.scene_navigator.generate_placements()
        self.scene_navigator.load_next_placement()

        # keeping track of the current room
        self.reset_seen_objs()

    def reset_seen_objs(self):
        self.objs_in_current_room = set()

    def collect_seen_objects(self, pil_image):
        objs_in_image_res = self.detect_objects_in_image(np.stack([pil_image], axis=0))
        #print("AE, tnp: ", objs_in_image_res, " ALL: ", self.objs_in_current_room)
        objs_in_image = set(objs_in_image_res['item_names'])
        self.objs_in_current_room = self.objs_in_current_room.union(objs_in_image)

    def go_to_room_centre(self):
        """
        Use remote DreamerV3 model on Jetson to put the agent at the centre of the current room
        :return:
        """
        self.rc_action_gen.set_image_receiver(self.collect_seen_objects)
        self.scene_navigator.set_action_gen(self.rc_action_gen)
        self.scene_navigator.navigate_to_goal()

    def go_to_next_room(self):
        """
        Use remote DreamerV3 model on Jetson to go through the nearest door and into the next room
        :return:
        """
        self.rc_action_gen.set_image_receiver(self.collect_seen_objects)
        self.scene_navigator.set_action_gen(self.dr_action_gen)
        self.scene_navigator.navigate_to_goal()

    def store_ref_path(self, path_imgs, path_id="?"):
        """
        Send an a collection of images, representing a reference path, to server.

        Args:
            image_np: numpy array (x, H, W, C) in BGR order (typical from OpenCV/AI2-THOR)

        Returns:
            success flag or None if error
        """
        # Serialize the images
        data = {
            'shape': path_imgs.shape,
            'dtype': str(path_imgs.dtype),
            'bytes': path_imgs.tobytes(),
            'path_id': path_id,
            'action': "store_ref_path",
            'module': "path_comparator"
        }

        ## debug
        path_id = str(path_id)
        os.makedirs(path_id, exist_ok=True)
        cnt = 0
        for img in path_imgs:
            cnt += 1
            cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
        ## /debug

        # Send request
        self.llm_socket.send_pyobj(data)

        # Wait for response (this BLOCKS until Jetson replies)
        try:
            response = self.llm_socket.recv_pyobj()
            return response
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            return None

    def qry_path_similarity(self, path_imgs):
        """
        Main navigation loop with real-time confidence feedback.

        Args:
            get_image_func: Function that captures current FPV image from AI2-THOR
            max_steps: Maximum number of steps to take
        """
        data = {
            'shape': path_imgs.shape,
            'dtype': str(path_imgs.dtype),
            'bytes': path_imgs.tobytes(),
            'action': "qry_path_similarity",
            'module': "path_comparator"
        }

        ## debug
        path_id = "tmp_cmp"
        os.makedirs(path_id, exist_ok=True)
        cnt = 0
        for img in path_imgs:
            cnt += 1
            cv2.imwrite(os.path.join(path_id, str(cnt) + ".png"), img)
        ## /debug

        # Send request
        self.llm_socket.send_pyobj(data)

        # Wait for response (this BLOCKS until Jetson replies)
        try:
            response = self.llm_socket.recv_pyobj()
            return response
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            return None

        # Small delay to avoid overwhelming the system
        time.sleep(0.05)

    def detect_objects_in_image(self, img):
        """
        Send an a collection of images, representing a reference path, to server.

        Args:
            image_np: numpy array (x, H, W, C) in BGR order (typical from OpenCV/AI2-THOR)

        Returns:
            success flag or None if error
        """
        # Serialize the images
        data = {
            'shape': img.shape,
            'dtype': str(img.dtype),
            'bytes': img.tobytes(),
            'action': "detect_objects_in_image",
            'module': "yolo_object_detector"
        }

        # Send request
        self.llm_socket.send_pyobj(data)

        # Wait for response (this BLOCKS until Jetson replies)
        try:
            response = self.llm_socket.recv_pyobj()
            return response
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            return None

    def classify_room_by_this_object_set_and_pic(self, obj_set = None, img_bytes = None):
        data = {
            'shape': img_bytes.shape,
            'dtype': str(img_bytes.dtype),
            'bytes': img_bytes.tobytes(),
            'obj_set': obj_set,
            'action': 'classify_room_by_this_object_set_and_pic',
            'module': 'llm_decisions'
        }

        # Send request
        self.llm_socket.send_pyobj(data)

        # Wait for response (this BLOCKS until Jetson replies)
        try:
            response = self.llm_socket.recv_pyobj()
            return response
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            return None

def extract_number(filename):
    # Extract the number from the filename (assuming it's the step count)
    # This regex looks for digits at the beginning, end, or between non-digits
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def load_images(path):
    imgs_path = glob.glob(path)
    imgs_path = sorted(imgs_path, key=extract_number)
    pil_images = [Image.open(fname).convert('RGB') for fname in imgs_path]
    return pil_images

def load_path(base_dir):
    return np.stack(load_images(base_dir + "/*.png"))

if __name__ == "__main__":
    # Create agent and connect to Jetson
    agent = SemanticNavigationClient(jetson_ip="192.168.0.109")

    # # Object detection in an image
    # pil_image = Image.open("/home/hp20024/robotics/latent_planning/dreamerv3/scene_pics/8.png")
    # img_array = np.stack([pil_image], axis = 0)
    # obj_det_res = agent.detect_objects_in_image(img_array)
    # det_objs = set(obj_det_res['item_names'])
    # print("AE: det objs: ", det_objs)
    #
    # # Room type inference from a set of objects and/or an image
    # print("AE: room type: ", agent.classify_room_by_this_object_set_and_pic(obj_set=det_objs, img_bytes = img_array))
    #
    # # Embedding of a path
    # ref_path1 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/1")
    # ref_path2 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/2")
    # ref_path3 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/3")
    # ref_path4 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/4")
    # ref_path7 = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/7")
    #
    # ref_cmp_path = load_path("/home/hp20024/robotics/latent_planning/snp_dreamerv3/ai2_thor_model_training_src/thortils/scripts/tmp_cmp")
    #
    # agent.store_ref_path(ref_path1, "ref_path1")
    # agent.store_ref_path(ref_path2, "ref_path2")
    # agent.store_ref_path(ref_path3, "ref_path3")
    # agent.store_ref_path(ref_path4, "ref_path4")
    # agent.store_ref_path(ref_path7, "ref_path7")
    #
    # # Comparison of a path against stored embedded ones
    # path_cmp_res = agent.qry_path_similarity(ref_cmp_path)
    # print("AE: path_cmp res: ", path_cmp_res)

    #agent.scene_navigator.process_habitat(10)
    agent.go_to_room_centre()
    print("While going to RC, I saw: ", agent.objs_in_current_room)
    print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room, np.stack([agent.rc_action_gen.last_image_large], axis=0)))

    agent.reset_seen_objs()
    agent.scene_navigator.load_next_placement()
    agent.go_to_room_centre()
    print("While going to RC, I saw: ", agent.objs_in_current_room)
    print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room,
                                                         np.stack([agent.rc_action_gen.last_image_large], axis=0)))

    agent.reset_seen_objs()
    agent.scene_navigator.load_next_placement()
    agent.go_to_room_centre()
    print("While going to RC, I saw: ", agent.objs_in_current_room)
    print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room,
                                                         np.stack([agent.rc_action_gen.last_image_large], axis=0)))