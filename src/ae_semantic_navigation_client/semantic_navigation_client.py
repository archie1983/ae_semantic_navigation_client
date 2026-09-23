import zmq, glob, re
import numpy as np
import time, cv2, os
from PIL import Image
from scene_navigator import SceneNavigator
from ai2_thor_model_training import index_to_action
from ae_llm_navigation_decisions import RoomType
from collections import Counter
from enum import Enum
from collections import deque

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

class SNPType(Enum):
    NONE = 0
    ROOM_CENTRE_FINDER = 1
    DOOR_FINDER = 2
    PERIMETER_WALKER = 3

class SemanticNavigationClient:
    LLM_PORT = 5555
    DR_NAV_PORT = 5556
    RC_NAV_PORT = 5557
    PER_NAV_PORT = 5558
    # Images for VPR (Visual Place Recognition)
    IMGS_TO_KEEP = 40
    IMGS_TO_EMBED = 10

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

        # Perimeter navigation container
        self.per_socket = self.context.socket(zmq.REQ)  # REQuest socket
        self.per_socket.connect(f"tcp://{jetson_ip}:{self.PER_NAV_PORT}")
        print(f"Connected to Jetson RoomCentre navigation container at {jetson_ip}:{self.PER_NAV_PORT}")

        # Local AI2-Thor simulation and action generators that talk to Dreamer models on Jetson:
        self.rc_action_gen = ActionGenerator(self.rc_socket)
        self.dr_action_gen = ActionGenerator(self.dr_socket)
        self.per_action_gen = ActionGenerator(self.per_socket)
        self.scene_navigator = SceneNavigator(self.rc_action_gen)

        # load a certain habitat
        self.scene_navigator.open_habitat(habitat_id)
        self.scene_navigator.generate_placements()
        self.scene_navigator.load_next_placement()

        # keeping track of the current room
        self.reset_seen_objs()
        self.reset_open_door_incidence()
        self.reset_last_pics()
        self.reset_last_room_type_identifations()

        self.common_objs = {'OPENDOOR', 'CLOSEDDOOR', 'FLOOR'}
        self.current_room_type = RoomType.NOT_KNOWN
        self.prev_room_type = RoomType.NOT_KNOWN
        self.objects_by_room = dict()

        self.current_active_SNP = SNPType.NONE
        # A queue to hold pending remedy functions
        self.remedy_commands = deque()
        self.main_commands = deque()

        self.door_transitions_stored = 0

    def reset_seen_objs(self):
        self.objs_in_current_room = set()

    def reset_open_door_incidence(self):
        self.open_door_incidence_last10 = []

    def reset_last_pics(self):
        self.fpv_images_last_x = []

    def reset_last_room_type_identifations(self):
        self.room_type_id_last10 = []
        self.room_detections_last10 = []

    def process_incoming_image(self, pil_image):
        '''
        Receive an image on every step during an SNP work - steps that we need to take for all SNPs
        :param pil_image:
        :return:
        '''
        # what can we see in the image?
        objs_in_image_res = self.detect_objects_in_image(np.stack([pil_image], axis=0))
        item_infos = objs_in_image_res['item_infos']
        objs_in_image = set([item['name'] for item in item_infos])
        instability_info = objs_in_image_res['instability_info']
        room_transition_spotted = False

        # find out what room it is based on the items
        room_detection = self.item_infos_to_roomtype(item_infos)

        # This is how we will store transfers between rooms:
        #  1) Store 10 images in a buffer at all times.
        #  2) At each step do a quick ID of the room if there's enough items. If not enough, use full ID with picture
        #  3) Once a change of room type is reliably detected, analyze the last 10 images. Check if we see doors.
        #  4) Those images with doors (or alternatively the first half images of the transition) get embedded and aggregated.
        #  5) The aggregate is stored as a transition between room type 1 and room type 2.
        # Now we will try to ID the room type
        # collect last 10 images
        room_type = room_detection['room_type']
        if room_type != None and room_type != room_type.NOT_KNOWN and room_type != room_type.NOT_CLASSIFIED:
            #print("detected RT: ", room_type, objs_in_image)
            # keep last 10 IDs that were successfully identified
            self.room_detections_last10.append(room_detection)

            # update using instability info if needed
            self.update_room_detections_after_instability(instability_info)

            if len(self.room_detections_last10) > 10:
                #self.room_type_id_last10 = self.room_type_id_last10[1:]
                #self.room_detections_last10 = self.room_detections_last10[1:]
                self.room_detections_last10.pop(0)

            self.room_type_id_last10 = [rd['room_type'] for rd in self.room_detections_last10]

            #print("AE: RT: ", self.room_type_id_last10)

            # Here we evaluate room type clusters
            if len(self.room_type_id_last10) >= 10:
                # Use standard library Counter to find the dominant room type in the buffer
                room_counts = Counter(self.room_type_id_last10)
                most_common_room, count = room_counts.most_common(1)[0]

                # Only transition if the dominant room has changed AND meets a threshold (e.g., 7/10 frames)
                if most_common_room != self.current_room_type and count >= 6:
                    # Trigger your embedding storage and transition mechanics here
                    self.prev_room_type = self.current_room_type
                    self.current_room_type = most_common_room
                    room_transition_spotted = True
                # else:
                #     print("AE: most_common_room: ", most_common_room, " count: ", count)

        # store FPVs
        self.fpv_images_last_x.append(pil_image)
        if len(self.fpv_images_last_x) > self.IMGS_TO_KEEP:
            #self.fpv_images_last_x = self.fpv_images_last_x[1:]
            self.fpv_images_last_x.pop(0)

        # if we have an open door, then remember that
        #self.detect_open_door_in_image(pil_image)
        if "OPENDOOR" in objs_in_image:
            self.open_door_incidence_last10.append(True)
        else:
            self.open_door_incidence_last10.append(False)

        if len(self.open_door_incidence_last10) > 10:
            #self.open_door_incidence_last10 = self.open_door_incidence_last10[1:]
            self.open_door_incidence_last10.pop(0)

        # If room transition spotted, then we want to manage objects seen in the previous room
        if room_transition_spotted:
            # If previous room is defined, then reset objects seen in that room because we will store new objects
            # If it is not defined, then assume that we're discovering the room type for the first time and the
            # collected objects need not be erased, but collected for the new room type, which will happen outside this
            # if block.
            if (not(self.is_room_nonsense(self.prev_room_type) or self.is_room_nonsense(self.current_room_type))):
                self.reset_seen_objs()

            # this might be a case of walking through an open plan living room into a kitchen (in which case we won't
            # have a door, or this might be a transition through a door. If it's through a door, then we want to save it
            #
            # For now let's detect all transitions regardless of doors.
            #if sum(self.open_door_incidence_last10) > 5 and len(self.fpv_images_last_x) > 5:
                imgs_to_embed = self.fpv_images_last_x[:self.IMGS_TO_EMBED]  # Or save the mid-point transition images
                self.store_door_transition(np.stack(imgs_to_embed), self.prev_room_type, self.current_room_type)

        # collect seen objects for this room type (or room)
        self.objs_in_current_room = self.objs_in_current_room.union(objs_in_image)

        # if we have a defined current room, then store that room's objects in the dict
        if (not self.is_room_nonsense(self.current_room_type)):
            self.objects_by_room[self.current_room_type] = self.objs_in_current_room

        return item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted

    def is_room_nonsense(self, current_room_type):
        if (self.prev_room_type == None
            or self.prev_room_type == RoomType.NOT_KNOWN
            or self.prev_room_type == RoomType.NOT_CLASSIFIED):
            return True
        else:
            return False

    def process_incoming_image_dr(self, pil_image):
        '''
        Receive an image on every step during DR SNP work and process it.
        :param pil_image:
        :return:
        '''
        # let's try to ID the room.
        item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted = self.process_incoming_image(pil_image)

        # TODO: If opendoor incidence getting high, then start querying the door imagery:
        # TODO: Raise the threshold from 75% to at least 85%
        # TODO: Store images from earlier in the sequence so that we detect coming transition earlier and also to make it more distinct
        # TODO: Consider querying on a smaller set of images to avoid extra data in them
        # TODO: Implement SNP interuption when wrong door is approached
        if sum(self.open_door_incidence_last10) > 5 and len(self.fpv_images_last_x) > 5:#self.IMGS_TO_EMBED:
        #if room_transition_spotted:
            #imgs_to_embed = self.fpv_images_last_x[5:]
            imgs_to_embed = self.fpv_images_last_x[-5:]  # get last images -- self.IMGS_TO_EMBED
            qry_result = self.qry_door_transition(np.stack(imgs_to_embed))
            if qry_result and qry_result['success'] and len(qry_result['qry_results']) > 0:
                print("AE: IMG QUERY: ", qry_result['qry_results'][0], " imgs_cnt: ", len(imgs_to_embed))

            if qry_result and qry_result['success'] and len(qry_result['qry_results']) > 0 and qry_result['qry_results'][0]['similarity'] >= 0.92:
                best_match = qry_result['qry_results'][0]
                print(f"I am 100% sure I am walking from {best_match['room_from']} to {best_match['room_to']}, conf = {qry_result['qry_results'][0]['similarity']}")
                self.scene_navigator.interrupt_navigation(self.callback_from_interrupted_snp)

        if room_transition_spotted:
            print("TRANS DR: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)

    def callback_from_interrupted_snp(self):
        print("AE: SNP INTERRUPTED AND SCENE NAVIGATOR CALLED BACK. Current active: ", self.current_active_SNP)
        # if we interrupted a door walker, then we probably want to go back to the room centre, but we can't launch
        # that SNP directly from here because this interrupt function needs to exit so that self.scene_navigator.navigate_to_goal()
        # can complete and set self.current_active_SNP to NONE and only then we should launch the new SNP.
        if self.current_active_SNP == SNPType.DOOR_FINDER:
            # Instead of calling it, push the function reference to our queue
            print("AE: Enqueuing remedy actions...")
            self.remedy_commands.append(self.go_to_room_centre)
            self.remedy_commands.append(self.go_to_next_room)

    def process_incoming_image_rc(self, pil_image):
        '''
        Receive an image on every step during RC SNP work and process it.
        :param pil_image:
        :return:
        '''
        # let's try to ID the room.
        item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted = self.process_incoming_image(pil_image)

        if room_transition_spotted:
            print("TRANS RC: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)

        # Not sure what else we might want to do in the room centre finder - at least for now while I'm focussing on environment exploration.

    def process_incoming_image_per(self, pil_image):
        '''
        Receive an image on every step during PER SNP work and process it.
        :param pil_image:
        :return:
        '''
        # let's try to ID the room.
        item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted = self.process_incoming_image(pil_image)

        if room_transition_spotted:
            print("TRANS PER: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)

        # Not sure what else we might want to do in the perimeter finder - at least for now while I'm focussing on environment exploration.

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

    def go_to_room_centre(self):
        """
        Use remote DreamerV3 model on Jetson to put the agent at the centre of the current room
        :return:
        """
        self.rc_action_gen.set_image_receiver(self.process_incoming_image_rc)
        self.scene_navigator.set_action_gen(self.rc_action_gen)
        self.current_active_SNP = SNPType.ROOM_CENTRE_FINDER
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE

    def go_to_next_room(self):
        """
        Use remote DreamerV3 model on Jetson to go through the nearest door and into the next room
        :return:
        """
        self.dr_action_gen.set_image_receiver(self.process_incoming_image_dr)
        self.scene_navigator.set_action_gen(self.dr_action_gen)
        self.current_active_SNP = SNPType.DOOR_FINDER
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE

    def go_to_perimeter_of_room(self):
        """
        Use remote DreamerV3 model on Jetson to go through the nearest door and into the next room
        :return:
        """
        self.per_action_gen.set_image_receiver(self.process_incoming_image_per)
        self.scene_navigator.set_action_gen(self.per_action_gen)
        self.current_active_SNP = SNPType.PERIMETER_WALKER
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE

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
        self.door_transitions_stored += 1
        path_id = room_from.name + "_to_" + room_to.name + "_" + str(self.door_transitions_stored)
        os.makedirs(path_id, exist_ok=True)
        cnt = 0
        print("STORING ", len(path_imgs), " images.")
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

    def qry_door_transition(self, path_imgs):
        """
        Send a collection of images, representing a door entrance, to server and get back results of similar doors if any.

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
            'action': "qry_door_transition",
            'module': "path_comparator"
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

    def run_agent_tick(self):
        """Call this from a separate thread."""
        # 1. If there's an active queued command (like a remedy), run it first
        if self.remedy_commands:
            next_command = self.remedy_commands.popleft()
        # 2. Otherwise, continue standard routine behaviors
        elif self.main_commands:
            next_command = self.main_commands.popleft()
        else:
            next_command = None

        if next_command is not None:
            next_command()  # Executes natively on spawned thread
            return True
        else:
            return False

        # sleep

    def add_main_command(self, command):
        self.main_commands.append(command)

    def do_work(self):
        cmd_cnt = 0
        # do run_agent_tick until no more commands left to do
        while (self.run_agent_tick()):
            cmd_cnt += 1
            print("AE: end of command ", cmd_cnt)

        print("AE: All work complete")

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
    rooms_to_traverse = 7
    for i in range(rooms_to_traverse):
        agent.add_main_command(agent.go_to_room_centre)
        agent.add_main_command(agent.go_to_next_room)

    agent.do_work()

    # TODO: Next step: implement not going through a visited door again during exploration.