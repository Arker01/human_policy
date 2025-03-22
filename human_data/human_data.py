from opentv.teleop.TeleVision import *
from opentv.teleop.Preprocessor import VuerPreprocessor

import h5py
import datetime
import time
import json
import cv2
import numpy as np
import pyzed.sl as sl
from argparse import ArgumentParser
from multiprocessing import Process, shared_memory, Queue, Manager, Event

np.set_printoptions(precision=2, suppress=True)

class ZED:
    def __init__(self, 
                 shm_name, 
                 img_shape, 
                 crop_size_h, 
                 crop_size_w, 
                 control_dict, 
                 toggle_recording) -> None:
        self.control_dict = control_dict
        self.toggle_recording = toggle_recording
        self.shm_name = shm_name
        self.img_shape = img_shape
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.zed = Process(target=self.zed_process)
        self.zed.daemon = True
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        indicator_pos_y = img_shape[0] // 10
        indicator_pos_x = img_shape[1] // 4
        self.record_position = (indicator_pos_x, indicator_pos_y)
        self.radius = 10
        self.color = (0, 255, 0)
        self.thickness = -1  # Filled circle
        self.input_color = (0, 0, 255)

        # num
        self.episode_position = (indicator_pos_x + 30, indicator_pos_y+ 10)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.txt_color = (255, 255, 255)
        self.txt_thickness = 2

        self.zed.start()
        
    def zed_process(self):
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {err}. Exiting.")
            return

        img_left = sl.Mat()
        img_right = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
        
        while True:
            if self.toggle_recording.is_set():
                if self.control_dict['is_recording']:
                    print("Stopped recording.")
                    zed.disable_recording()
                    self.control_dict['is_recording'] = False
                    self.color = (0, 255, 0)
                    self.video_writer.release()
                    # Prompt for episode description
                    meta_data = {
                        "video_frame_timestamps": self.timestamps  # Add timestamps array
                    }
                    with open(self.timestamp_path, 'w') as f:
                        json.dump(meta_data, f)
                else:
                    self.timestamps = []
                    self.video_path = self.control_dict['path'] + "_stereo.mp4"
                    self.svo_path = self.control_dict['path'] + ".svo"
                    self.timestamp_path = self.control_dict['path'] + "_vid_timestamps.json"
                    self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, 30, (2560, 720))  # Combined width
                    zed.enable_recording(sl.RecordingParameters(self.svo_path, sl.SVO_COMPRESSION_MODE.H264))
                    current_path = self.control_dict['path']
                    print(f"Started recording to: {current_path}")
                    self.control_dict['is_recording'] = True
                    self.color = (255, 0, 0)
                self.toggle_recording.clear()
            
            # start = time.time()
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(img_left, sl.VIEW.LEFT)
                zed.retrieve_image(img_right, sl.VIEW.RIGHT)
            
            left_img = img_left.get_data()
            right_img = img_right.get_data()

            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGRA2BGR)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGRA2BGR)

            if self.control_dict['is_recording']:
                try:
                    
                    # Combine left and right images horizontally
                    combined_frame = np.hstack((left_img, right_img))
                    self.video_writer.write(combined_frame)
                    self.timestamps.append(time.time())
                except Exception as e:
                    print(f"Error during video writing: {e}")

            # here we crop the image 
            left_rgb = cv2.cvtColor(img_left.get_data()[self.crop_size_h:, self.crop_size_w:-self.crop_size_w], cv2.COLOR_BGRA2RGB)
            right_rgb = cv2.cvtColor(img_right.get_data()[self.crop_size_h:, self.crop_size_w:-self.crop_size_w], cv2.COLOR_BGRA2RGB)

            cv2.circle(left_rgb, self.record_position, self.radius, self.color, self.thickness)
            cv2.circle(right_rgb, self.record_position, self.radius, self.color, self.thickness)

            cv2.putText(left_rgb, str(self.control_dict['episode']), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)
            cv2.putText(right_rgb, str(self.control_dict['episode']), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)

            rgb = np.hstack((left_rgb, right_rgb))
            np.copyto(img_array, rgb)

            # print(f"Time to process frame: {time.time() - start}")

    def update_output_path(self, new_output_path):
        self.config_queue.put({'output_path': new_output_path})

class GestureCheck:
    def __init__(self, path, freq, threshold=0.15) -> None:
        # add gesture check
        self.threshold = threshold
        self.freq = freq
        with open(path, 'rb') as f:
            self.gesture = np.load(f)
        self.init_buf()

        self.gesture_list = []
    
    def init_buf(self):
        self.gesture_window = np.zeros((self.freq*3, 1))
        self.gesture_cnt = 0

    def check(self, gesture):
        # post action gesture check
        verified = np.linalg.norm(gesture - self.gesture) <= self.threshold
        self.gesture_window = np.concatenate((np.array(verified).reshape((1,1)), self.gesture_window[:-1,:]))
        if self.gesture_window.sum() >= self.gesture_window.shape[0] * (9/10):
            self.init_buf()
            return True
        return False

    def add_gesture_to_list(self, new_gesture):
        # Append the gesture to the list
        self.gesture_list.append(new_gesture)
        # print("Gesture added to list.")
    
    def calculate_average_gesture(self):
        if not self.gesture_list:
            print("No gestures in list to calculate average.")
            return None
        # Calculate the average of all gestures in the list
        average_gesture = np.mean(self.gesture_list[100:-100], axis=0)
        print("Average gesture calculated.")
        return average_gesture
    
    def save_average_gesture(self, save_path="../data/ref_gestures/average_gesture.npy"):
        average_gesture = self.calculate_average_gesture()
        if average_gesture is not None:
            with open(save_path, 'wb') as f:
                np.save(f, average_gesture)
            print(f"Average gesture saved to {save_path}")

class Dataset:
    def __init__(self, path,) -> None:
        # add gesture check
        self.path = path
        self.data_dict = {'/obs/timestamp': [],
                          '/action/cmd/head_mat': [],
                          '/action/cmd/rel_left_wrist_mat': [],
                          '/action/cmd/rel_right_wrist_mat': [],
                          '/action/cmd/rel_left_hand_keypoints': [],
                          '/action/cmd/rel_right_hand_keypoints': []}   # 4*4 + 4*4 + 4*4 + 25*3 + 25*3

    def insert(self, 
               timestamp, 
               head_mat, 
               rel_left_wrist_mat, 
               rel_right_wrist_mat, 
               rel_left_hand_keypoints, 
               rel_right_hand_keypoints):
        self.data_dict['/obs/timestamp'].append(timestamp)
        self.data_dict['/action/cmd/head_mat'].append(head_mat)
        self.data_dict['/action/cmd/rel_left_wrist_mat'].append(rel_left_wrist_mat)
        self.data_dict['/action/cmd/rel_right_wrist_mat'].append(rel_right_wrist_mat)
        self.data_dict['/action/cmd/rel_left_hand_keypoints'].append(rel_left_hand_keypoints)
        self.data_dict['/action/cmd/rel_right_hand_keypoints'].append(rel_right_hand_keypoints)
    
    def save_to_hdf5(self, description, embodiment):
        with h5py.File(self.path + ".hdf5", 'w') as file:
            for key, value in self.data_dict.items():
                file.create_dataset(key, data=value)
        with open(self.path + "_meta.json", 'w') as f:
            json.dump({"description": description, 
                       "embodiment": embodiment,
                       "num_prop_frames": len(self.data_dict['/obs/timestamp']),
                       "time_start": self.data_dict['/obs/timestamp'][0],
                       "time_end": self.data_dict['/obs/timestamp'][-1],
                       "total_time": self.data_dict['/obs/timestamp'][-1] - self.data_dict['/obs/timestamp'][0]}, 
                       f)

class HumanDataCollection():
    def __init__(self, args, freq, path, stream_mode, description) -> None:
        self.path = path
        self.description = description

        self.episode = 0

        self.freq = freq

        self.dt = 1.0 / self.freq

        self.timestep = 0


        self.manager = Manager()
        
        self.processor = VuerPreprocessor()
        
        self.resolution = (720, 1280)
        self.crop_size_w = 160  # 1#(resolution[1] - resolution[0]) // 2
        self.crop_size_h = 0  # 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        # robot related params
        self.img_shape = (self.resolution_cropped[0], 2*self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        self.shm = shared_memory.SharedMemory(name="sim_image", create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        image_queue = Queue()

        # self.control_queue = Queue()
        self.if_record = not args.no_record
        self.manager = Manager()
        self.control_dict = self.manager.dict()
        self.control_dict['is_recording'] = False
        self.control_dict['path'] = ""
        self.control_dict['episode'] = 0
        self.toggle_recording = Event()
        self.toggle_streaming = Event()

        self.tv = OpenTeleVision(self.resolution_cropped, 
                            self.shm.name, 
                            image_queue,
                            self.toggle_streaming, 
                            cert_file="./cert.pem", 
                            key_file="./key.pem", 
                            )
        
        print("Resolution cropped: ", self.resolution_cropped)
        print("Shared memory name: ", self.shm.name)
        print("Stream mode: ", stream_mode)
        print("Using ngrok: ", args.http)

        self.cam = ZED(self.shm.name, 
                        self.img_shape, 
                        self.crop_size_h, 
                        self.crop_size_w, 
                        self.control_dict, 
                        self.toggle_recording)

        self.record_check = GestureCheck('../data/ref_gestures/record_gesture.npy', self.freq)
        self.drop_check = GestureCheck('../data/ref_gestures/drop_gesture.npy', self.freq)

    def step(self):
        processed_mat= self.processor.process(self.tv)

        if self.control_dict["is_recording"] and self.if_record:
            action_time = time.time()
            self.dataset.insert(action_time,
                                *processed_mat)
            # print(f"Recording time: {time.time() - start}")
        
        self.post_step_callback()

    def post_step_callback(self):
        new_gesture = self.processor.get_hand_gesture(self.tv)

        if self.record_check.check(new_gesture) and self.if_record:
            print("Record gesture detected, recording toggled")
            self.toggle_recording.set()
            if self.control_dict["is_recording"]:  # stop recording
                self.episode += 1
                description = self.description
                self.dataset.save_to_hdf5(description, "human_zed")
            os.makedirs(folder_name, exist_ok=True)  # start recording
            self.control_dict["path"] = os.path.join(self.path, f"episode_{self.episode}")#f"../data/{time.time()}.svo"
            self.control_dict["episode"] = self.episode
            self.dataset = Dataset(self.control_dict["path"])
        elif self.drop_check.check(new_gesture) and self.if_record and self.control_dict["is_recording"]:
            print("DROP!")
            self.toggle_recording.set()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ip', default='192.168.8.194')
    parser.add_argument('--des', default='')
    parser.add_argument('--no_record', default=False, action='store_true')
    parser.add_argument('--stream_mode', default='image', choices=['image', 'webrtc'])
    parser.add_argument('--http', default=False, action='store_true')
    parser.add_argument('--description', default='')
    args = parser.parse_args()

    if not args.no_record:
        time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.des == "":
            folder_name = "../data/recordings/" + time_str
        else:
            folder_name = "../data/recordings/" + args.des + "-" + time_str
    else:
        folder_name = None

    FREQ = 30
    STEP_TIME = 1/FREQ

    pipeline = HumanDataCollection(args, FREQ, path=folder_name, stream_mode=args.stream_mode, description=args.description)
    
    start = time.time()
    i = 0
    print("here")

    while True:
        # print(time.time()-start)
        start = time.time()  # 10e-6 s precision
        pipeline.step()
        duration = time.time() - start
        # print(f"freq: {1/duration}")
        if duration < STEP_TIME:
            time.sleep(STEP_TIME - duration)
        i+=1