import zmq, glob, re, random
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
        self.img_cnt = 0

        # Sometimes we will need to interrupt the SNP and send a reset signal to DreamerV3 model. The reset condition
        # might persist, but we don't want to keep sending the reset signals. This variable will help with that.
        self.steps_after_reset = 0

    def reset(self):
        self._cur_obs["is_first"] = True
        self._cur_obs["is_last"] = False
        self.steps_after_reset = 0

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

        bbox_to_cover = None
        # Create a working copy so we don't permanently alter the simulation telemetry frame
        processed_image = ai2_thor_image.copy()

        # preparing 2 size images: 64x64 for DreamerV3 models and 600x600 or 640x640 whatever AI2-Thor launcher
        # is configured with for YOLO models.
        rgb_img_large = cv2.cvtColor(ai2_thor_image, cv2.COLOR_BGR2RGB)
        pil_image_large = Image.fromarray(rgb_img_large)

        if self.image_receiver is not None:
            vpr_info = self.image_receiver(pil_image_large)
            if vpr_info is not None:
                bbox_to_cover, early_or_late = vpr_info
            else:
                bbox_to_cover = None
                early_or_late = None
            self.last_image_large = pil_image_large
        # TODO: implement analysis of early_or_late
        # If we got back bbox_to_cover then we need to ablate that in the image before passing back to the SNP
        if bbox_to_cover is not None:
            # if this is an early detection, then just paint over the doors
            if early_or_late:
                print("AE: painting DOOR only bbox_to_cover: ", bbox_to_cover, " shape: ", processed_image.shape)
                # bbox expected format: [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = map(int, bbox_to_cover)
                # Paint a solid neutral grey polygon (128, 128, 128) over the door
                cv2.rectangle(
                    processed_image,
                    (xmin, ymin),
                    (xmax, ymax),
                    (128, 128, 128),
                    thickness=-1  # -1 fills the interior entirely
                )
            else: # otherwise blank out all
                print("AE: painting BLOCKING bbox_to_cover: ", bbox_to_cover, " shape: ", processed_image.shape)
                cv2.rectangle(
                    processed_image,
                    (0, 0),
                    (600, 600),
                    (128, 128, 128),
                    thickness=-1  # -1 fills the interior entirely
                )

            # Resize to 64 x 64
            img_64x64 = cv2.resize(
                processed_image,
                (64, 64),
                interpolation=cv2.INTER_LANCZOS4  # High quality
            )

            # if we have a bbox to cover and we've already ran for at least 10 steps, then reset
            if self.steps_after_reset >= 10:
                self.reset()

        else:
            # Resize to 64 x 64
            img_64x64 = cv2.resize(
                ai2_thor_image,
                (64, 64),
                interpolation=cv2.INTER_LANCZOS4  # High quality
            )

        rgb_img_64x64 = cv2.cvtColor(img_64x64, cv2.COLOR_BGR2RGB)
        pil_image_64x64 = Image.fromarray(rgb_img_64x64)

        if bbox_to_cover is not None:
            ## debug
            path_id = "masked_doors"
            os.makedirs(path_id, exist_ok=True)
            self.img_cnt += 1
            cv2.imwrite(os.path.join(path_id, str(self.img_cnt) + ".png"), rgb_img_64x64)
            ## /debug

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
            self.steps_after_reset += 1
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            response = None

        next_move_str = index_to_action(response['action_bits']['action']) # <- this needs to talk to Jetson over ZMQ and pass it the image
        #print("ACT: ", next_move_str)
        # along with the rest of the observation.
        # next_move_str has to be returned from Dreamer running on Jetson and then if it is STOP, then we need to prepare
        # a observation with is_last = True. If however this is the very first image after loading a scene, then we
        # need to set is_first = True.

        #print("AE AG : .socket = ", self.socket, " @@ response = ", response)
        if next_move_str == "STOP":
            self.stop_received()
        elif response['action_bits']['reset']:
            self.reset()
        else:
            self.normal_op()

        return next_move_str

class RandomRotationActionGen(ActionGenerator):
    '''
    A very simple action generator- it just generates a series of rotations to change the FPV of the agent to a random direction
    and then unloads the generated actions one by one when called.
    '''
    def __init__(self):
        self.regen_actions()

    def regen_actions(self):
        # which direction
        rotate_direction = bool(random.randint(0, 1))
        # how many steps (1 step = 45 degrees)
        step_cnt = random.randint(1, 4)
        self.actions = ['RotateRight' if rotate_direction else 'RotateLeft' for i in range(step_cnt)]

    def __call__(self, ai2_thor_image):
        if self.actions:
            next_move_str = self.actions.pop(0) # here we just extract commands that we generated earlier
        else:
            next_move_str = "STOP"
        return next_move_str

    # Overriding unused functions to do nothing
    def reset(self): pass
    def stop_received(self): pass
    def normal_op(self): pass
    def handshake(self): pass
    def set_image_receiver(self, image_receiver): pass

class SNPType(Enum):
    NONE = 0
    ROOM_CENTRE_FINDER = 1
    DOOR_FINDER = 2
    PERIMETER_WALKER = 3
    RANDOM_ROTATOR = 4

class SemanticNavigationClient:
    LLM_PORT = 5555
    DR_NAV_PORT = 5556
    RC_NAV_PORT = 5557
    PER_NAV_PORT = 5558
    # Images for VPR (Visual Place Recognition)
    IMGS_TO_KEEP = 100
    IMGS_TO_EMBED = 10
    IMG_HISTORY_FOR_IMM_VPR = 40 # how long in the past to look if we want to store immediate door transition (we're close to the door).
    IMG_HISTORY_FOR_EARLY_VPR = 60 # ditto, but for early VPR (when we're only approaching)
    DEBUG = True

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
        self.rr_action_gen = RandomRotationActionGen()
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
        self.reset_doors_in_current_transition_run()

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
        self.open_door_track_ids_last_x = []

    def reset_doors_in_current_transition_run(self):
        self.doors_in_current_transition_run = []

    def add_door_in_current_transition_run(self, track_id, bbox, door_pic):
        """
        When we transit to a different room, we want to have a good set of doors that lead us there. This collection
        will allow tracking them.
        :param track_id: YOLO track_id
        :param bbox: bbox from the original image might be useful to infer the size of the door or distance to it
        :param door_pic: cropped door
        :return:
        """
        self.doors_in_current_transition_run.append({'track_id': track_id, 'bbox': bbox, 'door_pic': door_pic})

    def reset_last_room_type_identifations(self):
        self.room_type_id_last10 = []
        self.room_detections_last10 = []

    def process_incoming_image(self, pil_image):
        '''
        Receive an image on every step during an SNP work - steps that we need to take for all SNPs.
        Basically pass it to YOLO to check what's in it and detect room type and so on.
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

        # if we have an open door, then remember that
        #self.detect_open_door_in_image(pil_image)
        if "OPENDOOR" in objs_in_image:
            self.open_door_incidence_last10.append(True)
            # if we have detected an OPENDOOR, then we also want to know the tracking IDs for these doors so that we can
            # later block them out if needed. We will be clearing this collection out together with self.fpv_images_last_x.
            door_track_ids = [{'track_id': item['track_id'], 'relative_distance': self.check_door_proximity(item['bbox'])[1]} for item in item_infos if item['name'] == 'OPENDOOR']
            track_id_to_add = sorted(door_track_ids, key=lambda x: x['relative_distance'])[0]
            if track_id_to_add['track_id'] > -1: # only add it if we've got a history of tracking it
                self.open_door_track_ids_last_x.append([track_id_to_add['track_id']])
                #print("AE: Adding door_track_ids in self.open_door_track_ids_last_x: ", track_id_to_add)
            else:
                self.open_door_track_ids_last_x.append([]) # append empty list to keep consistend with self.fpv_images_last_x

            # If we've spotted an OPENDOOR, then let's store a cropped image of it along with its tracker ID so that later
            # when we store a door transition, we have a good distribution of what this door looks like from different
            # angles and distances.
            # Important: This is different from self.open_door_track_ids_last_x collection in such a way that we will clear
            # self.doors_in_current_transition_run when transition completes, but self.open_door_track_ids_last_x is a
            # total running history of door track IDs which we only clear one by one when the buffer is full.
            # self.doors_in_current_transition_run is for the current transition only (there is always a transition BTW,
            # because sooner or later we will go through a door).
            potential_doors_to_add_to_current_transition = []

            for item in item_infos:
                if item['name'] == 'OPENDOOR':
                    track_id = item['track_id']
                    bbox = item['bbox']
                    _, relative_distance = self.check_door_proximity(bbox)
                    potential_doors_to_add_to_current_transition.append({'track_id': track_id, 'bbox': bbox, 'relative_distance': relative_distance})

            nearest_door = sorted(potential_doors_to_add_to_current_transition, key = lambda x: x['relative_distance'])[0]
            door_only_pic = self.crop_bbox_from_pil(pil_image, nearest_door['bbox'])
            if nearest_door['track_id'] > -1: # only add it if we've got a history of tracking it
                self.add_door_in_current_transition_run(nearest_door['track_id'], nearest_door['bbox'], door_only_pic)
                #print("AE: Adding track_id to self.doors_in_current_transition_run : ", nearest_door)

            # for item in item_infos:
            #     if item['name'] == 'OPENDOOR':
            #         track_id = item['track_id']
            #         bbox = item['bbox']
            #         door_only_pic = self.crop_bbox_from_pil(pil_image, bbox)
            #
            #         # TODO: Only add the nearest door if there are several.
            #         _, relative_distance = self.check_door_proximity(bbox)
            #
            #         self.add_door_in_current_transition_run(track_id, bbox, door_only_pic)
            #         print("AE: Adding track_id to self.doors_in_current_transition_run : ", track_id)
        else:
            self.open_door_incidence_last10.append(False)
            self.open_door_track_ids_last_x.append([])
            #print("AE: Adding door_track_ids: NONE")

        if len(self.open_door_incidence_last10) > 10:
            #self.open_door_incidence_last10 = self.open_door_incidence_last10[1:]
            self.open_door_incidence_last10.pop(0)

        # store FPVs
        self.fpv_images_last_x.append(pil_image)
        if len(self.fpv_images_last_x) > self.IMGS_TO_KEEP:
            #self.fpv_images_last_x = self.fpv_images_last_x[1:]
            self.fpv_images_last_x.pop(0)
            self.open_door_track_ids_last_x.pop(0)

        # If room transition spotted, then we want to manage objects seen in the previous room
        if room_transition_spotted:
            self.process_room_transition(skip_storing_data=self.DEBUG)

        # collect seen objects for this room type (or room)
        self.objs_in_current_room = self.objs_in_current_room.union(objs_in_image)

        # if we have a defined current room, then store that room's objects in the dict
        if (not self.is_room_nonsense(self.current_room_type)):
            self.objects_by_room[self.current_room_type] = self.objs_in_current_room

        return item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted

    def check_door_proximity(self, bbox, image_height=600, min_percentage=0.25):
        """
        Checks if a door is close enough to confidently identify/block,
        using a normalized vertical scale factor to bypass width variations.

        Args:
            bbox: List [xmin, ymin, xmax, ymax]
            image_height: The absolute height of the raw camera frame (e.g., 600)
            min_percentage: The minimum portion of the screen height the door must occupy
        """
        xmin, ymin, xmax, ymax = bbox

        # Calculate vertical height in pixels
        door_pixel_height = ymax - ymin

        # Calculate what percentage of the camera's field of view the door height occupies
        occupancy_ratio = door_pixel_height / image_height

        # If occupancy ratio is 0.25, the door takes up 25% of the frame vertically
        if occupancy_ratio >= min_percentage:
            return True, occupancy_ratio

        return False, occupancy_ratio

    def do_we_need_to_paint_this_door(self, stored_door_info):
        if ((stored_door_info['qry_results'] and
                len(stored_door_info['qry_results']) > 0 and
                stored_door_info['qry_results'][0]['similarity'] > 0.8)
            and (stored_door_info['qry_results'][0] == self.current_room_type)):
            return True
        else:
            #print("AE: Stored Door Info: ", stored_door_info)
            return False

    def is_room_nonsense(self, room_type):
        if (room_type == None
            or room_type == RoomType.NOT_KNOWN
            or room_type == RoomType.NOT_CLASSIFIED):
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

        result = (None, None)

        if sum(self.open_door_incidence_last10) > 5 and len(self.fpv_images_last_x) > 5 and not self.DEBUG:#self.IMGS_TO_EMBED:
        #if room_transition_spotted:
            #imgs_to_embed = self.fpv_images_last_x[5:]
            imgs_to_embed = self.fpv_images_last_x[-5:]  # get last images -- self.IMGS_TO_EMBED
            qry_result = self.qry_door_transition(np.stack(imgs_to_embed))
            # if qry_result and qry_result['success'] and len(qry_result['qry_results']) > 0:
            #     print("AE: IMG QUERY: ", qry_result['qry_results'][0], " imgs_cnt: ", len(imgs_to_embed))

            if qry_result and qry_result['success'] and len(qry_result['qry_results']) > 0 and qry_result['qry_results'][0]['similarity'] >= 0.92:
                best_match = qry_result['qry_results'][0]
                print(f"I am 100% sure I am walking from {best_match['room_from']} to {best_match['room_to']}, early_or_late: {best_match['early_or_late']}, conf = {qry_result['qry_results'][0]['similarity']}")
                #self.scene_navigator.interrupt_navigation(self.callback_from_interrupted_snp)

                ## Thhis is the original way to ablate doors
                # # Here we now need to track the past x images with doors in them and I guess find the prevalent track_id of opendoors objects in the recent
                # # history. And then mark that ID as the forbidden door so that we can paint the grey box over it.
                # most_common_track_id = self.most_common_door_track_id_in_recent_history()
                # # this is the forbidden door: most_common_track_id
                #
                # bbox_to_cover = None
                # for item in item_infos:
                #     if item['track_id'] == most_common_track_id and item['name'] == 'OPENDOOR':
                #         bbox_to_cover = item['bbox']
                # if bbox_to_cover == []: bbox_to_cover = None
                # result = (bbox_to_cover, best_match['early_or_late'])

        ## This is the new way (note the nesting level):
        if 'OPENDOOR' in objs_in_image and not self.DEBUG:
            for item in item_infos:
                if item['name'] == 'OPENDOOR':
                    bbox = item['bbox']
                    door_only_pic = self.crop_bbox_from_pil(pil_image, bbox)
                    # If we've spotted an OPENDOOR, then let's query if this door has been seen from other
                    # angles and distances.
                    # query if we already know this door. And if we do and it is leading where we don't want to go,
                    # then paint it.
                    stored_door_info = self.qry_door_images_of_transition(door_only_pic)
                    if (self.do_we_need_to_paint_this_door(stored_door_info)):
                        bbox_to_cover = item['bbox']
                        result = (bbox_to_cover, True)

        if room_transition_spotted:
            #print("TRANS DR: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)
            print("TRANS DR: ", self.prev_room_type, self.current_room_type)

        return result

    def most_common_door_track_id_in_recent_history(self):
        """
        Here we now need to track the past x images with doors in them and find the prevalent track_id of opendoors objects in the recent
        history.
        :return:
        """
        track_id_history_of_interest = self.open_door_track_ids_last_x[-10:]

        # empties_discarded = [item for item in self.open_door_track_ids_last_x if len(item) > 0]
        # track_id_history_of_interest = empties_discarded[-5:]

        # flatten our list of lists
        track_ids_flat = sum(track_id_history_of_interest, [])
        #print("AE: track_ids_flat == ", track_ids_flat)
        if len(track_ids_flat) > 0:
            track_id_counts = Counter(track_ids_flat)
            most_common_track_id, count = track_id_counts.most_common(1)[0]
            return most_common_track_id
        else:
            return None

    def callback_from_interrupted_snp(self):
        print("AE: SNP INTERRUPTED AND SCENE NAVIGATOR CALLED BACK. Current active: ", self.current_active_SNP)
        # if we interrupted a door walker, then we probably want to go back to the room centre, but we can't launch
        # that SNP directly from here because this interrupt function needs to exit so that self.scene_navigator.navigate_to_goal()
        # can complete and set self.current_active_SNP to NONE and only then we should launch the new SNP.
        if self.current_active_SNP == SNPType.DOOR_FINDER:
            # Instead of calling it, push the function reference to our queue
            print("AE: Enqueuing remedy actions...")
            # TODO: consider door finding task to be a sequence of perimeter SNP until we see a door + door finder
            #  instead of just door finder SNP alone. That should allow for discovery of other doors after we've reset
            #  the RSSM state with the is_first flag.
            #
            # TODO: Introduce two levels of transition recognition - the one that we already have (let's call it the
            #  immediate one) and another one- earlier (call it early one). If we have an immediate transition in front,
            #  then blat out whole screen. If the early one, then only blat out the door and still reset SNP.
            #self.remedy_commands.append(self.go_to_room_centre)
            self.remedy_commands.append(self.do_random_rotation)
            self.remedy_commands.append(self.go_to_next_room)
            self.dr_action_gen.reset()
        elif self.current_active_SNP == SNPType.ROOM_CENTRE_FINDER:
            self.rc_action_gen.reset()
        elif self.current_active_SNP == SNPType.PERIMETER_WALKER:
            self.per_action_gen.reset()

    def process_incoming_image_rc(self, pil_image):
        '''
        Receive an image on every step during RC SNP work and process it.
        :param pil_image:
        :return:
        '''
        # let's try to ID the room.
        item_infos, objs_in_image, instability_info, room_detection, room_transition_spotted = self.process_incoming_image(pil_image)

        if room_transition_spotted:
            #print("TRANS RC: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)
            print("TRANS RC: ", self.prev_room_type, self.current_room_type)

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
            #print("TRANS PER: ", self.room_type_id_last10, self.prev_room_type, self.current_room_type)
            print("TRANS PER: ", self.prev_room_type, self.current_room_type)

        # Not sure what else we might want to do in the perimeter finder - at least for now while I'm focussing on environment exploration.

        # Run perimeter finder for 100 steps only to promote room exploration before we resort to RC SNP
        if self.per_action_gen.steps_after_reset >= 50:
            self.scene_navigator.interrupt_navigation(self.callback_from_interrupted_snp)

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

    def process_room_transition_debug(self):
        most_common_door_track_id = self.most_common_door_track_id_in_recent_history()

        door_track_ids = [
            (item['track_id'], float(np.round(self.check_door_proximity(item['bbox'])[1], 2))) for item in
            self.doors_in_current_transition_run]

        print("AE: self.doors_in_current_transition_run : ",
              door_track_ids, " most_common_door_track_id: ",
              most_common_door_track_id, " self.open_door_track_ids_last_x[-10:]: ",
              self.open_door_track_ids_last_x[-10:])

        path_id = "debug_door_pics"
        os.makedirs(path_id, exist_ok=True)
        for item in self.doors_in_current_transition_run:
            img_np = np.array(item['door_pic'])
            prox = float(np.round(self.check_door_proximity(item['bbox'])[1], 2))
            prox = str(prox).replace('.', '_')
            cv2.imwrite(os.path.join(path_id, f"{item['track_id']}_{prox}.png"), img_np)

    def process_room_transition(self, skip_storing_data = False):
        """
        When room has changed (e.g. Door SNP has completed work or we have classified  a new room type),
        this function will do what needs doing- storing doors' images in various ways, etc.

        :param skip_storing_data: For debug purposes: skip storing images in Vector DB

        :return:
        """
        if skip_storing_data:
            if not(self.is_room_nonsense(self.prev_room_type)) and not(self.is_room_nonsense(self.current_room_type)):
                self.reset_seen_objs()
                self.reset_doors_in_current_transition_run()
        else:
            # We want to manage objects seen in the previous room
            # If previous room is defined, then reset objects seen in that room because we will store new objects
            # If it is not defined, then assume that we're discovering the room type for the first time and the
            # collected objects need not be erased, but collected for the new room type, which will happen outside this
            # if block.
            if not(self.is_room_nonsense(self.prev_room_type)) and not(self.is_room_nonsense(self.current_room_type)):
            # this might be a case of walking through an open plan living room into a kitchen (in which case we won't
            # have a door, or this might be a transition through a door. If it's through a door, then we want to save it
            #
            # For now let's detect all transitions regardless of doors.
            #if sum(self.open_door_incidence_last10) > 5 and len(self.fpv_images_last_x) > 5:
                #imgs_to_embed = self.fpv_images_last_x[:self.IMGS_TO_EMBED]  # Or save the mid-point transition images
                imgs_to_embed = self.fpv_images_last_x[-self.IMG_HISTORY_FOR_IMM_VPR:][:self.IMGS_TO_EMBED] # take first 10 images from the history of 40 back
                self.store_door_transition(np.stack(imgs_to_embed), self.prev_room_type, self.current_room_type, False)

                # now let's see if we have enough imagery for an early transition storage
                if len(self.fpv_images_last_x) >= self.IMG_HISTORY_FOR_EARLY_VPR:
                    imgs_to_embed = self.fpv_images_last_x[-self.IMG_HISTORY_FOR_EARLY_VPR:][:self.IMGS_TO_EMBED]  # take first 10 images from the history of 40 back
                    self.store_door_transition(np.stack(imgs_to_embed), self.prev_room_type, self.current_room_type, True)

                # Now that we've stored transition to a new room, let's also store the looks of the door that brought us there
                # The most commond door track ID in the recent history (something like last 10 images) should be the door that's
                # lead us to the new room.
                most_common_door_track_id = self.most_common_door_track_id_in_recent_history()
                if most_common_door_track_id is not None:
                    door_pics_infos_to_store = [(item['door_pic'], item['bbox']) for item in self.doors_in_current_transition_run if item['track_id'] == most_common_door_track_id]
                    # Store door pics to vector DB pertaining to this transition.
                    # TODO: Why we didn't get DOORPICS when transiting from BEDROOM to LIVING_ROOM?
                    if len(door_pics_infos_to_store) > 0:
                        self.store_door_images_of_transition(door_pics_infos_to_store, self.prev_room_type, self.current_room_type)
                    else:
                        # TODO: Remove this branch once we have confirmed that this condition is fixed and does not happen anymore
                        print("AE: self.doors_in_current_transition_run : ", [item['track_id'] for item in self.doors_in_current_transition_run], " most_common_door_track_id: ", most_common_door_track_id, " self.open_door_track_ids_last_x[-10:]: ", self.open_door_track_ids_last_x[-10:])
                        exit()

                self.reset_seen_objs()
                self.reset_doors_in_current_transition_run()

    def go_to_room_centre(self):
        """
        Use remote DreamerV3 model on Jetson to put the agent at the centre of the current room
        :return:
        """
        self.rc_action_gen.set_image_receiver(self.process_incoming_image_rc)
        self.scene_navigator.set_action_gen(self.rc_action_gen)
        self.current_active_SNP = SNPType.ROOM_CENTRE_FINDER
        print("AE: ROOM_CENTRE_FINDER started")
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE
        print("AE: ROOM_CENTRE_FINDER ended")

    def go_to_next_room(self):
        """
        Use remote DreamerV3 model on Jetson to go through the nearest door and into the next room
        :return:
        """
        self.dr_action_gen.set_image_receiver(self.process_incoming_image_dr)
        self.scene_navigator.set_action_gen(self.dr_action_gen)
        self.current_active_SNP = SNPType.DOOR_FINDER
        print("AE: DOOR_FINDER started")
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE
        # trigger here a transition to a new room - store door images that we may have just seen
        self.process_room_transition_debug()
        print("AE: DOOR_FINDER ended")

    def go_to_perimeter_of_room(self):
        """
        Use remote DreamerV3 model on Jetson to go through the nearest door and into the next room
        :return:
        """
        self.per_action_gen.set_image_receiver(self.process_incoming_image_per)
        self.scene_navigator.set_action_gen(self.per_action_gen)
        self.current_active_SNP = SNPType.PERIMETER_WALKER
        print("AE: PERIMETER_WALKER started")
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE
        print("AE: PERIMETER_WALKER ended")

    def do_random_rotation(self):
        """
        Just rotate in place (yaw) either left or right anything between 45 and 180 degrees.
        :return:
        """
        self.scene_navigator.set_action_gen(self.rr_action_gen)
        self.rr_action_gen.regen_actions()
        self.current_active_SNP = SNPType.RANDOM_ROTATOR
        self.scene_navigator.navigate_to_goal()
        self.current_active_SNP = SNPType.NONE

    def store_door_images_of_transition(self, door_pics_infos_to_store, room_from, room_to):
        """
        Store images of the actual doors when transiting from one room to another.
        Serializes each cropped image into its own byte block to handle varying dimensions.

        :param door_imgs: images of doors only for fast ID later
        :param room_from:
        :param room_to:
        :return:
        """
        serialized_pics = []
        door_bboxes_to_store = []

        #print("AE: door_pics_infos_to_store: ", door_pics_infos_to_store)

        for item in door_pics_infos_to_store:
            pil_img = item[0]
            bbox = item[1]

            # 1. Convert the PIL Image into a NumPy array (handling RGB/BGR properly)
            img_np = np.array(pil_img)
            #img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = img_np

            # 2. Compress the individual frame into a JPEG memory buffer
            success, encoded_img = cv2.imencode('.jpg', img_bgr)
            if success:
                # Store the raw binary bytes, its unique shape, and its bounding box bounds
                serialized_pics.append({
                    'bytes': img_np.tobytes(),
                    'shape': img_bgr.shape,  # (H, W, C)
                    'dtype': str(img_bgr.dtype)
                })
                door_bboxes_to_store.append(bbox)

        # Construct the network structure payload
        data = {
            'door_pics': serialized_pics,
            'room_from': room_from.name,
            'room_to': room_to.name,
            'door_bboxes': door_bboxes_to_store,
            'action': "store_door_pics_of_transition",
            'module': "path_comparator"
        }

        ## debug - Keeping your exact diagnostic loop working smoothly
        path_id = room_from.name + "_to_" + room_to.name + "_" + str(self.door_transitions_stored) + "_DOORPICS"
        os.makedirs(path_id, exist_ok=True)
        print(f"STORING {len(serialized_pics)} door pics. shapes: ", [item['shape'] for item in data['door_pics']])

        for cnt, item in enumerate(door_pics_infos_to_store, 1):
            img_np = np.array(item[0])
            #img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = img_np
            cv2.imwrite(os.path.join(path_id, f"{cnt}.png"), img_bgr)
        ## /debug

        # Send serialized object structure over ZMQ
        self.llm_socket.send_pyobj(data)

        try:
            response = self.llm_socket.recv_pyobj()
            return response
        except zmq.ZMQError as e:
            print(f"Error receiving response: {e}")
            return None

    def qry_door_images_of_transition(self, door_img):
        door_img = np.stack([door_img])

        # Serialize the images
        data = {
            'shape': door_img.shape,
            'dtype': str(door_img.dtype),
            'bytes': door_img.tobytes(),
            'action': "qry_door_pics_of_transition",
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

    def store_door_transition(self, path_imgs, room_from, room_to, early_or_late):
        """
        Send an a collection of images, representing a door entrance, to server.

        Args:

        :param path_imgs:
        :param room_from:
        :param room_to:
        :param early_or_late: is this an early (door still quite far) or late (already going through) VPR for a door transition
        :return:

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
            'early_or_late': early_or_late,
            'action': "store_door_transition",
            'module': "path_comparator"
        }

        ## debug
        self.door_transitions_stored += 1
        path_id = room_from.name + "_to_" + room_to.name + "_" + str(self.door_transitions_stored) + "_" + ('EARLY' if early_or_late else 'IMM')
        os.makedirs(path_id, exist_ok=True)
        cnt = 0
        #print("STORING ", len(path_imgs), " images. early_or_late = ", early_or_late)
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

    def crop_bbox_from_pil(self, pil_image, bbox):
        """
        Crops an image segment (usually a door) out of a PIL Image instance.

        Args:
            pil_image: PIL.Image object
            bbox: Flat list [xmin, ymin, xmax, ymax]
        Returns:
            cropped_pil_image: A new cropped PIL Image object
        """
        # Unpack and cast to integers
        xmin, ymin, xmax, ymax = map(int, bbox)

        # PIL handles boundary safety internally, returning blank space if out-of-bounds
        cropped_pil_image = pil_image.crop((xmin, ymin, xmax, ymax))

        return cropped_pil_image

    def detect_objects_in_image(self, img):
        """
        Send a collection of images, representing a reference path, to server.

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
    #agent.add_main_command(agent.go_to_room_centre)
    for i in range(rooms_to_traverse):
        agent.add_main_command(agent.go_to_room_centre)
        agent.add_main_command(agent.go_to_next_room)
        agent.add_main_command(agent.go_to_perimeter_of_room)

    agent.do_work()

    # TODO: Next step: implement not going through a visited door again during exploration.