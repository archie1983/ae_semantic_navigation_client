import zmq, glob, re
import numpy as np
import time, cv2, os
from PIL import Image
from scene_navigator import SceneNavigator
from ai2_thor_model_training import index_to_action
from ae_llm_navigation_decisions import RoomType
from collections import Counter

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

    def __init__(self, jetson_ip, habitat_id = 78):
        self.context = zmq.Context()
        # # LLM container
        self.llm_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.llm_socket.connect(f"tcp://{jetson_ip}:{self.LLM_PORT}")
        print(f"Connected to Jetson LLM container at {jetson_ip}:{self.LLM_PORT}")
        #
        # Door navigation container
        self.dr_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.dr_socket.connect(f"tcp://{jetson_ip}:{self.DR_NAV_PORT}")
        print(f"Connected to Jetson Door navigation container at {jetson_ip}:{self.DR_NAV_PORT}")

        # Room centre navigation container
        self.rc_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.rc_socket.connect(f"tcp://{jetson_ip}:{self.RC_NAV_PORT}")
        print(f"Connected to Jetson RoomCentre navigation container at {jetson_ip}:{self.RC_NAV_PORT}")

        # Local AI2-Thor simulation and action generators that talk to Dreamer models on Jetson:
        self.rc_action_gen = ActionGenerator(self.rc_socket)
        self.dr_action_gen = ActionGenerator(self.dr_socket)
        self.scene_navigator = SceneNavigator(self.rc_action_gen)

        # load a certain habitat
        self.scene_navigator.open_habitat(habitat_id)
        self.scene_navigator.generate_placements()
        self.scene_navigator.load_next_placement()

        # keeping track of the current room
        self.reset_seen_objs()
        self.reset_open_door_incidence()
        self.reset_last_10_pics()
        self.reset_last_room_type_identifations()

        self.common_objs = {'OPENDOOR', 'CLOSEDDOOR', 'FLOOR'}
        self.current_room_type = RoomType.NOT_KNOWN
        self.prev_room_type = RoomType.NOT_KNOWN

    def reset_seen_objs(self):
        self.objs_in_current_room = set()

    def reset_open_door_incidence(self):
        self.open_door_incidence_last10 = []

    def reset_last_10_pics(self):
        self.fpv_images_last10 = []

    def reset_last_room_type_identifations(self):
        self.room_type_id_last10 = []
        self.room_detections_last10 = []

    def collect_seen_objects(self, pil_image):
        objs_in_image_res = self.detect_objects_in_image(np.stack([pil_image], axis=0))
        #print("AE, tnp: ", objs_in_image_res, " ALL: ", self.objs_in_current_room)
        objs_in_image = set(objs_in_image_res['item_names'])
        self.objs_in_current_room = self.objs_in_current_room.union(objs_in_image)

    def process_incoming_image_dr(self, pil_image):
        '''
        Receive an image on every step during DR SNP work and process it.
        :param pil_image:
        :return:
        '''
        # let's try to ID the room. If enough objects, then use quick ID, if not, also include the image
        objs_in_image_res = self.detect_objects_in_image(np.stack([pil_image], axis=0))
        #objs_in_image = set(objs_in_image_res['item_names'])
        #print(objs_in_image_res)

        item_infos = objs_in_image_res['item_infos']
        objs_in_image = set([item['name'] for item in item_infos])
        instability_info = objs_in_image_res['instability_info']

        # find out what room it is based on the items
        room_detection = self.item_infos_to_roomtype(item_infos)

        # if we have an open door, then remember that
        #self.detect_open_door_in_image(pil_image)
        if "OPENDOOR" in objs_in_image:
            self.open_door_incidence_last10.append(True)
        else:
            self.open_door_incidence_last10.append(False)

        if len(self.open_door_incidence_last10) > 10:
            self.open_door_incidence_last10 = self.open_door_incidence_last10[1:]

        # This is how we will store transfers between rooms:
        #  1) Store 10 images in a buffer at all times.
        #  2) At each step do a quick ID of the room if there's enough items. If not enough, use full ID with picture
        #  3) Once a change of room type is reliably detected, analyze the last 10 images. Check if we see doors.
        #  4) Those images with doors (or alternatively the first half images of the transition) get embedded and aggregated.
        #  5) The aggregate is stored as a transition between room type 1 and room type 2.
        # Now we will try to ID the room type
        # collect last 10 images
        self.fpv_images_last10.append(pil_image)
        if len(self.fpv_images_last10) > 10:
            self.fpv_images_last10 = self.fpv_images_last10[1:]

        room_type = room_detection['room_type']
        if room_type != None and room_type != room_type.NOT_KNOWN and room_type != room_type.NOT_CLASSIFIED:
            print("detected RT: ", room_type, objs_in_image)
            # keep last 10 IDs that were successfully identified
            self.room_detections_last10.append(room_detection)

            # update using instability info if needed
            self.update_room_detections_after_instability(instability_info)

            ## TODO: Here we have to rethink the decision. I think we need some clustering and the biggest cluster wins.
            if len(self.room_detections_last10) > 10:
                #self.room_type_id_last10 = self.room_type_id_last10[1:]
                #self.room_detections_last10 = self.room_detections_last10[1:]
                self.room_detections_last10.pop(0)

            self.room_type_id_last10 = [rd['room_type'] for rd in self.room_detections_last10]

            # Here we evaluate room type clusters
            if len(self.room_type_id_last10) >= 10:
                # Use standard library Counter to find the dominant room type in the buffer
                room_counts = Counter(self.room_type_id_last10)
                most_common_room, count = room_counts.most_common(1)[0]

                # Only transition if the dominant room has changed AND meets a threshold (e.g., 7/10 frames)
                if most_common_room != self.current_room_type and count >= 7:
                    # Trigger your embedding storage and transition mechanics here
                    self.prev_room_type = self.current_room_type
                    self.current_room_type = most_common_room

                    imgs_to_embed = self.fpv_images_last10[:5]  # Or save the mid-point transition images
                    self.store_door_transition(np.stack(imgs_to_embed), self.prev_room_type, self.current_room_type)
                    print("TRANS: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)
            #

            # # Now check if we have a new room type reliably detected
            # if len(self.room_type_id_last10) > 8:
            #     seen_room_types = list(set(self.room_type_id_last10))
            #     rt_1st_half = self.room_type_id_last10[:5]
            #     rt_2nd_half = self.room_type_id_last10[5:]
            #
            #     most_rt_ndx_1st_half = np.argmax([sum(1 if t == rt else 0 for t in rt_1st_half) for rt in seen_room_types])
            #     most_rt_1st_half = seen_room_types[most_rt_ndx_1st_half]
            #     most_rt_ndx_2nd_half = np.argmax([sum(1 if t == rt else 0 for t in rt_2nd_half) for rt in seen_room_types])
            #     most_rt_2nd_half = seen_room_types[most_rt_ndx_2nd_half]
            #     if most_rt_1st_half != most_rt_2nd_half:
            #         # so we detected a room change. Let's embed images leading to here
            #         # for door_present, img in zip(self.open_door_incidence_last10, self.fpv_images_last10):
            #         #     if door_present:
            #         imgs_to_embed = self.fpv_images_last10[:5]
            #         self.store_door_transition(np.stack(imgs_to_embed), most_rt_1st_half, most_rt_2nd_half)
            #         print("TRANS: ", self.room_type_id_last10, most_rt_1st_half, most_rt_2nd_half)

    def update_room_detections_after_instability(self, instability_info):
        if instability_info is None or len(instability_info) <= 0: return

        affected_ids = [instability['track_id'] for instability in instability_info]
        updated_rds = []

        # go through our collected room detections and check if we need to re-detect
        for rd in self.room_detections_last10:
            updated_rd_items = []
            item_infos_updated = False
            # look at each item
            for ii in rd['item_infos']:
                # if the track_id is affected, then exclude this item
                if ii['track_id'] not in affected_ids:
                    updated_rd_items.append(ii)
                else:
                    #print("AE: throwing out: ", ii)
                    item_infos_updated = True

            # now we have updated items (either same as before or fewer)
            rd['item_infos'] = updated_rd_items

            # if there was a change, then let's re-classify
            if item_infos_updated:
                new_rd = self.item_infos_to_roomtype(rd['item_infos'])
                #print("AE: reclass:  was: ", rd, " now: ", new_rd)
                # if classification was possible, then store it
                if new_rd['room_type'] != None:
                    updated_rds.append(new_rd)
            else:
                # if no change, then keep original
                updated_rds.append(rd)

        # update what we have
        #print("AE: changed self.room_detections_last10 from: ", self.room_detections_last10, " to: ", updated_rds, " affected_ids: ", affected_ids)
        self.room_detections_last10 = updated_rds
        return updated_rds

    ##
    # Turn a collection of items and their attributes into a room type
    ##
    def item_infos_to_roomtype(self, item_infos):
        objs_in_image = set([item['name'] for item in item_infos])
        objs_in_image_no_commons = objs_in_image - self.common_objs
        # decide how we're going to ID it
        if len(objs_in_image_no_commons) > 0:
            room_type = self.quick_classify_room_by_this_object_set(objs_in_image)
        else:
            #room_type = self.classify_room_by_this_object_set_and_pic(objs_in_image, np.stack([pil_image], axis = 0))
            room_type = None

        return {'room_type': room_type, 'item_infos': item_infos}

    def detect_open_door_in_image(self, pil_image):
        objs_in_image_res = self.detect_objects_in_image(np.stack([pil_image], axis=0))
        #print("AE, tnp: ", objs_in_image_res, " ALL: ", self.objs_in_current_room)
        objs_in_image = set(objs_in_image_res['item_names'])
        if "OPENDOOR" in objs_in_image:
            self.open_door_incidence_last10.append(True)
        else:
            self.open_door_incidence_last10.append(False)

        if len(self.open_door_incidence_last10) > 10:
            self.open_door_incidence_last10 = self.open_door_incidence_last10[1:]

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
        self.dr_action_gen.set_image_receiver(self.process_incoming_image_dr)
        self.scene_navigator.set_action_gen(self.dr_action_gen)
        self.scene_navigator.navigate_to_goal()

    def store_door_transition(self, path_imgs, room_from, room_to):
        """
        Send an a collection of images, representing a door entrance, to server.

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
            'room_from': room_from.name,
            'room_to': room_to.name,
            'action': "store_door_transition",
            'module': "path_comparator"
        }

        ## debug
        path_id = room_from.name + "_to_" + room_to.name
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

    def quick_classify_room_by_this_object_set(self, obj_set = None):
        data = {
            'obj_set': obj_set,
            'action': 'quick_classify_room_by_this_object_set',
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
    agent = SemanticNavigationClient(jetson_ip="192.168.0.109", habitat_id=65)

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

    # #agent.scene_navigator.process_habitat(10)
    # agent.go_to_room_centre()
    # print("While going to RC, I saw: ", agent.objs_in_current_room)
    # print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room, np.stack([agent.rc_action_gen.last_image_large], axis=0)))
    #
    # agent.reset_seen_objs()
    # agent.scene_navigator.load_next_placement()
    # agent.go_to_room_centre()
    # print("While going to RC, I saw: ", agent.objs_in_current_room)
    # print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room,
    #                                                      np.stack([agent.rc_action_gen.last_image_large], axis=0)))
    #
    # agent.reset_seen_objs()
    # agent.scene_navigator.load_next_placement()
    # agent.go_to_room_centre()
    # print("While going to RC, I saw: ", agent.objs_in_current_room)
    # print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room,
    #                                                      np.stack([agent.rc_action_gen.last_image_large], axis=0)))

    # agent.go_to_room_centre()
    # print("While going to RC, I saw: ", agent.objs_in_current_room)
    # #print(agent.classify_room_by_this_object_set_and_pic(agent.objs_in_current_room, np.stack([agent.rc_action_gen.last_image_large], axis=0)))
    # print(agent.quick_classify_room_by_this_object_set(agent.objs_in_current_room))

    agent.reset_seen_objs()
    agent.go_to_next_room()
    print("Presence of OPENDOOR in last 10 images: ", agent.open_door_incidence_last10)