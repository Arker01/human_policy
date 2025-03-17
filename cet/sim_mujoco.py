import os
import cv2
import json
import time
import yaml
import glfw
import numpy as np
from pathlib import Path
import os
from scipy.spatial.transform import Rotation as R
from multiprocessing import shared_memory

import mujoco as mj

from opentv.RobotController import RobotController
from cet.utils import policy2ctrl_cmd, safe_as_quat, safe_from_quat

ROBOT_POS = [0, 0, 1.6]
ROBOT_POS_OFFSET = [0, 0, 0]  # For table-top manipulation
ROBOT_POS = [ROBOT_POS[i] + ROBOT_POS_OFFSET[i] for i in range(3)]

class MujocoSim:
    def __init__(
                self,
                config_files,
                root_path=None,
                print_freq=False,
                teleop_control=False,
                task_id=0,
                tasktype=None,
                path = "../data/recordings",
                shm_name = None,
                img_shape = (720, 1280, 3),
                is_viewer=False,
                control_dict = None,
                toggle_recording = None,
                crop_size_w = 0,
                crop_size_h = 0,
                cfgs=None
                ):
        
        # Basic setup
        self.root_path = root_path
        self.teleop_control = teleop_control
        self.episode = 0
        self.print_freq = print_freq
        self.is_viewer = is_viewer
        self.tasktype = tasktype.lower()

        # Image resolution setup
        self.resolution = (720, 1280)
        self.crop_size_w = 0 # (resolution[1] - resolution[0]) // 2
        self.crop_size_h = 0
        self.resolution_cropped = (
            int((self.resolution[0]-self.crop_size_h)/1.5), 
            (self.resolution[1]-2*self.crop_size_w)//2
        )
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)

        # Teleoperation setup
        if self.teleop_control:
            self._init_teleop(path, shm_name, img_shape, control_dict, toggle_recording)

        # Initialize logs
        self.action_log = open("action_log.txt", "w")
        self.qpos_log = open("qpos_log.txt", "w")
        
        # Load configuration files
        self.cfgs = self._load_configs(config_files, cfgs)
        
        # Simulation Environment Setup
        self.num_envs = len(self.cfgs)
        self.envs = []
        self.robot_handles = []
        self.cur_env = task_id
        self.robot_asset_files = []
        self.dof_names = []
        self.num_dof = []
        self.initial_object_qpos = []
        
        # Indicator, UI, and Gesture Parameters
        self._init_ui_params()
        
        # Mujoco setup
        self._init_mujoco(task_id)
        
        # Create environments
        self.create_envs()
        
    def _init_teleop(self, path, shm_name, img_shape, control_dict, toggle_recording):
        """ Initialize teleoperation-related settings"""
        self.img_shape = img_shape
        self.path = path
        self.frame_rate = 100
        self.last_frame_time = time.time()
        self.start_time = time.time()
        self.is_recording = False
        self.control_dict = control_dict
        self.toggle_recording = toggle_recording
        
        # Video recording setup
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Shared memory setup
        try:
            self.existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray(
                                (self.img_shape[0], self.img_shape[1], 3), 
                                dtype=np.uint8, 
                                buffer=self.existing_shm.buf
                                )
        except FileNotFoundError:
            print(f"Shared memory with name {shm_name} not found.")
              
    def _load_configs(self, config_files, cfgs):
        """ Load robot configurations files """
        if cfgs is not None:
            return cfgs
        
        loaded_cfgs = []
        config_dir = Path(__file__).resolve().parent.parent / "configs"
        
        for config_file_name in config_files:
            config_file_path = config_dir / config_file_name
            with open(config_file_path, "r") as f:
                cfg = yaml.safe_load(f)['robot_cfg']
            loaded_cfgs.append(cfg)
                
        return loaded_cfgs
    
    def _init_ui_params(self):
        """ Initialize UI Parameters for indicators and text rendering"""
        indicator_pos_y = self.img_shape[0] // 10
        indicator_pos_x = self.img_shape[1] // 4
        
        # Recording indicator
        self.color = (0, 255, 0)
        self.record_position = (indicator_pos_x, indicator_pos_y)
        self.radius = 10 # Dot radius
        self.thickness = -1 # Filled circle
        
        # Gesture check
        self.gesture_color = (0, 0, 0)
        self.gesture_position = (indicator_pos_x, indicator_pos_y)
        self.gesture_thickness = 5

        # Text properties
        self.episode_position = (indicator_pos_x + 30, indicator_pos_y + 10)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.txt_color = (255, 255, 255)
        self.txt_thickness = 2
        
        # Image dimensions
        self.img_width = self.img_shape[1] // 2
        self.img_height = self.img_shape[0]

    def _init_mujoco(self, task_id):
        """ Initialize MuJoCo-related settings """
        robot_asset_root = str(Path(__file__).resolve().parent.parent / "assets")
        xml_path = self.cfgs[task_id]["xml_path"]
        xml_path = os.path.join(robot_asset_root, xml_path, self.tasktype + "_scene.xml")
        self.robot_asset_files.append(xml_path)

        # Load Mujoco model and data
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # Simulation settings
        self.model.opt.timestep = 1.0 / 70.0  # Increase frequency to 70Hz
        self.model.opt.gravity = np.array([0.0, 0.0, -9.81])
        self.model.opt.iterations = 80  # Increase solver iterations
        self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON  # Use Newton solver
        self.model.opt.tolerance = 1e-6  # Increase solver tolerance
        self.model.opt.noslip_tolerance = 1e-6  # No-slip tolerance
        self.model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICIT  # Implicit integration
        self.model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL  # Pyramidal friction cone
        self.model.opt.impratio = 5  # Implicit-to-explicit integration ratio

        # Increase joint damping for stability
        for i in range(self.model.nv):
            self.model.dof_damping[i] *= 2.0

        # Initialize GLFW hidden context for rendering
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        self.hidden_window = glfw.create_window(1, 1, "hidden", None, None)
        glfw.make_context_current(self.hidden_window)

        # Rendering context
        self.img_width = self.img_shape[1] // 2
        self.img_height = self.img_shape[0]
        self.viewport = mj.MjrRect(0, 0, self.img_width, self.img_height)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        
        # Initialize controller if needed
        if not self.teleop_control:
            self.controller = RobotController(
                asset_dir=self.root_path + "/assets", 
                config_dir=self.root_path + "/configs"
            )
            self.controller.load_config('h1_inspire_sim.yml') # Visualize sim data
    
    def create_envs(self):
        """Initialize multiple environments based on configurations files."""
        for env_idx in range(self.num_envs):
            cfg = self.cfgs[env_idx]

            # Extract robot joint and body indices
            self.robot_joint_ids = np.array(cfg.get('robot_indices', []))
            self.body_ids = np.array(cfg.get('body_indices', []))
            self.initial_object_qpos.append(self.data.qpos[self.robot_joint_ids.shape[0]:].copy())

            # Extract DOF properties
            dof = len(self.robot_joint_ids)
            robot_dof_props = self.get_dof_properties()
            self.hand_indices = cfg.get('left_hand_indices', []) + cfg.get('right_hand_indices', [])
            self.torso_indices = cfg.get('torso_indices', [])
            
            # Apply PD gains (if defined in cfg)
            # self._apply_pd_gains(cfg, robot_dof_props)
            # PD IS BEING SET EXPLICITLY IN STEP FUNCTION
            
            self.model.geom_friction = 0.6

            # Store joint names and DOF count
            joint_names = [self.model.joint(i).name for i in range(self.model.njnt)][:self.robot_joint_ids.shape[0]]
            self.dof_names.append(joint_names)   
            self.num_dof.append(self.robot_joint_ids.shape[0])      

        print('Done creating envs')
       
    def _apply_pd_gains(self, cfg, robot_dof_props):
        """Apply proportional-derivative (PD) gains to robot joints."""
        pd = cfg.get("pd", None)
        if pd:
            # Apply PD gains from config
            robot_dof_props["stiffness"][cfg.get("body_indices", [])] = pd["body_kp"]
            robot_dof_props["damping"][cfg.get("body_indices", [])] = pd["body_kd"]

            for side in ["left", "right"]:
                arm_indices = cfg.get(f"{side}_arm_indices", [])
                hand_indices = cfg.get(f"{side}_hand_indices", [])
                robot_dof_props["stiffness"][arm_indices] = pd["arm_kp"]
                robot_dof_props["damping"][arm_indices] = pd["arm_kd"]
                robot_dof_props["stiffness"][hand_indices] = pd["hand_kp"]
                robot_dof_props["damping"][hand_indices] = pd["hand_kd"]
        else:
            # Default PD values
            for i in range(len(self.robot_joint_ids)):
                if i in self.hand_indices:
                    robot_dof_props["stiffness"][i] = 500.0
                    robot_dof_props["damping"][i] = 10.0
                else:
                    robot_dof_props["stiffness"][i] = 60.0
                    robot_dof_props["damping"][i] = 20.0
                
    def get_dof_properties(self):
        """Get properties only for robot joints."""
        
        return {
            'lower': self.model.jnt_range[self.robot_joint_ids, 0],     # Joint lower limits
            'upper': self.model.jnt_range[self.robot_joint_ids, 1],     # Joint upper limits
            'velocity': self.model.actuator_ctrlrange[:, 1] if len(self.model.actuator_ctrlrange) > 0 else np.array([]),  # Max velocity limits
            'effort': self.model.actuator_forcerange[:, 1] if len(self.model.actuator_forcerange) > 0 else np.array([]),   # Max torque/force limits
            'stiffness': self.model.jnt_stiffness[self.robot_joint_ids],  # Joint stiffness
            'damping': self.model.dof_damping[self.robot_joint_ids],      # Joint damping
            'friction': self.model.dof_frictionloss[self.robot_joint_ids], # Joint friction
            'armature': self.model.dof_armature[self.robot_joint_ids],    # Joint armature (rotor inertia)
        }
 
    def setup_viewer(self, viewer):
        """ Initialize the MuJoCo viewer and configure recording cameras"""
        self.viewer = viewer
        
        # Set viewer position based on environment type
        env_name = self.cfgs[self.cur_env]["name"]
        self.viewer_pos = np.array([0.16, 0, 1.65] if "gr1" in env_name else [0.15, 0, 1.65])
        self.viewer_target = np.array([0.45, 0, 1.45])  # TODO: Adjust based on head_pos in YAML
        # FOR TESTING RW DATA:
        # self.viewer_target = np.array([0.45, 0, 1.30])

        # Compute camera angles
        direction = self.viewer_target - self.viewer_pos
        distance = np.linalg.norm(direction)  # Distance from pos to target
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arcsin(direction[2] / distance))

        # Apply settings to main viewer camera
        self.viewer.cam.distance = distance
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation   
        self.viewer.cam.lookat = self.viewer_target
        
        # Camera offsets for stereo recording
        self.stereo_cam_offsets = {"left": np.array([0, 0.033, 0]), "right": np.array([0, -0.033, 0])} # Hardcoded
        self.left_cam, self.right_cam = mj.MjvCamera(), mj.MjvCamera()

        for cam_name, cam in zip(["left", "right"], [self.left_cam, self.right_cam]):
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat = (self.viewer_target + self.stereo_cam_offsets[cam_name]).tolist()
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation

        print('Done setting up viewer')
    
    def sim_config(self):
        """Return simulation configuration for current environment"""
        return {"num_dof": self.num_dof[self.cur_env], 
                  "dof_names": self.dof_names[self.cur_env], 
                  "urdf_path": self.robot_asset_files[self.cur_env]}
    
    # Used by open-television
    @property
    def all_sim_configs(self):
        """Return all simulation configurations"""
        return [{"num_dof": self.num_dof[i], 
                  "dof_names": self.dof_names[i], 
                  "urdf_path": self.robot_asset_files[i]} for i in range(self.num_envs)]

    def step(self, cmd, head_rmat, viewer):
        """Perform a single simulation step"""
        self.step_camera(head_rmat)

        if self.print_freq:
            start = time.time()
            

        # Define control parameters based on robot type
        robot_name = self.cfgs[self.cur_env]['name']
        is_gr1 = robot_name in ['gr1', 'gr1_inspire', 'gr1_inspire_sim']
        is_h1 = robot_name in ['h1_inspire', 'h1_2_inspire_cmu', 'h1_inspire_sim']
        
        kp = np.full(self.robot_joint_ids.shape[0], 200)
        kd = np.full(self.robot_joint_ids.shape[0], 10)

        # For GR1, GR1_JAW, GR1_ACE, force the waist joint's roll, pitch, yaw to 0
        if is_gr1:
            cmd[self.body_ids] = 0
            kp[self.hand_indices] = 80
            kd[self.hand_indices] = 2
            kp[self.torso_indices] = 200
            kd[self.torso_indices] = 3

        # For H1_inspire, H1_2_inspire_cmu, H1_inspire_sim
        elif is_h1:
            kp[self.hand_indices] = 20
            kd[self.hand_indices] = 0.25
            self.model.jnt_stiffness[self.hand_indices] = 0
            # self.model.jnt_damping[self.hand_indices] = 1

        # Apply Joint-space PD control
        self.set_torque_servo(np.arange(self.model.nu), 1)
        self.data.ctrl[self.robot_joint_ids] = -kp*(self.data.qpos[self.robot_joint_ids] - cmd) - kd*self.data.qvel[self.robot_joint_ids]

        # Step simulation
        mj.mj_step(self.model, self.data)
        viewer.sync()

        # Handle recording
        if self.teleop_control:
            if self.toggle_recording.is_set():
                if self.control_dict['is_recording']:
                    print("Video recording stopped")
                    self.video_writer.release() 
                    with open(self.timestamp_path, 'w') as f:
                        json.dump({'video_frame_timestamps': self.timestamps}, f)
                    print(f"Data saved to {self.timestamp_path}")
                    self.color = (0, 255, 0)
                    self.control_dict['is_recording'] = False
                else:
                    os.makedirs(self.path, exist_ok=True)
                    print('start recording')
                    self.video_path = f"{self.path}/episode_{self.episode}_stereo.mp4"
                    self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, self.frame_rate, (int(self.img_shape[1]), self.img_shape[0]))
                    self.timestamp_path = f"{self.path}/episode_{self.episode}_vid_timestamps.json"
                    self.timestamps = []
                    self.obj_pose_log_path = f"{self.path}/episode_{self.episode}_obj_pose.json"
                    self.control_dict['is_recording'] = True
                    self.color = (255, 0, 0)
                    self.reset_env_randomize()
                    self.episode += 1
                self.toggle_recording.clear()  

        if self.print_freq:
            print(f"Time taken for step: {time.time() - start}")
    
    def step_init(self, action, viewer, init_state=False, init_qos = None):
        """Initializes and steps the simulation"""
        self.action_log.write(f"{action}\n")  
        qpos, head_rmat = self.generate_qpos(action, init_state, init_qos)
        self.qpos_log.write(f"{qpos}\n")
        self.step(qpos, head_rmat, viewer)

    def generate_qpos(self, action, init_state=False, init_qos=None):
        """Generate joint positions based on action and initial state"""
        '''
        h1_inspire joint configuration:
        left_arm_indices = [13, 14, 15, 16, 17, 18, 19]
        right_arm_indices = [32, 33, 34, 35, 36, 37, 38]
        left_hand_indices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        right_hand_indices = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        '''
        
        robot_name = self.cfgs[self.cur_env]['name']

        if not init_state:# or init_qos is None:
            if robot_name in ['h1_inspire', 'h1_2_inspire_cmu', 'h1_inspire_sim']:
                if action.shape[0] == 128:
                    ctrl_cmd = policy2ctrl_cmd(action[None, :])
                    self.controller.update(*ctrl_cmd)
                    return self.controller.qpos, self.controller.head_rot_mat
                
                qpos = np.zeros(51)
                
                # left arm actions
                qpos[13:20] = action[0:7]
                # left hand actions
                qpos[20:22] = action[7] # intermediate joints mimic proximals
                qpos[22:24] = action[8] 
                qpos[24:26] = action[9] 
                qpos[26:28] = action[10] 
                qpos[28] = action[11] # thumb joints
                qpos[29:32] = action[12] * np.array([1, 1.6, 2.4])
                
                # right arm actions
                qpos[32:39] = action[13:20]
                # right hand actions
                qpos[39:41] = action[20] 
                qpos[41:43] = action[21]
                qpos[43:45] = action[22]
                qpos[45:47] = action[23]
                qpos[47] = action[24]
                qpos[48:51] = action[25] * np.array([1, 1.6, 2.4])
                qpos = np.array(qpos, dtype=np.float32)
                
            elif robot_name in ['gr1_inspire', 'gr1_inspire_sim']:
                if action.shape[0] == 128:
                    ctrl_cmd = policy2ctrl_cmd(action[None, :])
                    self.controller.update(*ctrl_cmd)
                    qpos = self.controller.qpos
                elif action.shape[0] == 28:
                    qpos = np.zeros(56)

                    # left arm actions
                    qpos[18:25] = action[0:7]
                    # left hand actions
                    qpos[25:27] = action[7] # intermediate joints mimic proximals
                    qpos[27:29] = action[8] 
                    qpos[29:31] = action[9] 
                    qpos[31:33] = action[10] # intermediate joints mimic proximals
                    qpos[33] = action[11] # thumb joints
                    qpos[34:37] = action[12] * np.array([1, 1.6, 2.4])

                    # right arm actions
                    qpos[37:44] = action[13:20]
                    # right hand actions
                    qpos[44:46] = action[20] 
                    qpos[46:48] = action[21]
                    qpos[48:50] = action[22]
                    qpos[50:52] = action[23]
                    qpos[52] = action[24]
                    qpos[53:56] = action[25] * np.array([1, 1.6, 2.4])
            else:
                raise ValueError(f"Unknown robot: {robot_name}")

        #! different robot need different init state, later add to config file
        else:
            if robot_name in ['h1_inspire', 'h1_2_inspire_cmu', 'h1_inspire_sim']:
                init_state = np.zeros((51,))
                init_state[13] = -0.5
                init_state[14] = 1
                init_state[16] = -2
                init_state[32] = -0.5
                init_state[33] = -1
                init_state[35] = -2
                qpos = init_state
                qpos = np.array(qpos, dtype=np.float32)
            elif robot_name in ['gr1_inspire', 'gr1_inspire_sim']:
                init_state = np.zeros((56,))
                init_state[18] = -0.5
                init_state[19] = 1
                init_state[21] = -2
                init_state[37] = -0.5
                init_state[38] = -1
                init_state[40] = -2
                qpos = init_state
                qpos = np.array(qpos, dtype=np.float32)
            else:
                raise ValueError(f"Unknown robot: {robot_name}")
            
        return qpos, self.controller.head_rot_mat
    
    def step_camera(self, head_rmat):

        curr_viewer_target = self.viewer_target @ head_rmat.T
        curr_left_offset = self.stereo_cam_offsets["left"] @ head_rmat.T
        curr_right_offset = self.stereo_cam_offsets["right"] @ head_rmat.T

        l_rec_cam_pos, l_rec_cam_target = self.viewer_pos + curr_left_offset, curr_viewer_target + curr_left_offset
        r_rec_cam_pos, r_rec_cam_target = self.viewer_pos + curr_right_offset, curr_viewer_target + curr_right_offset

        direction = curr_viewer_target - self.viewer_pos
        distance = np.linalg.norm(direction)  # Distance from target to pos
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        elevation = np.degrees(np.arcsin(direction[2] / distance))
        
        # Update camera parameters
        for cam, target in zip([self.left_cam, self.right_cam], [l_rec_cam_target, r_rec_cam_target]):
            cam.lookat = target.tolist()
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation

        # Get camera images
        left_image = self.get_camera_image(cam_id=0)  # Use cam_id instead of fixedcamid
        right_image = self.get_camera_image(cam_id=1)

        # Convert images to contiguous arrays that OpenCV can modify
        left_image = np.ascontiguousarray(left_image)
        left_image_ = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = np.ascontiguousarray(right_image)
        right_image_ = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        combined_frame = np.hstack((left_image_, right_image_))

        # Handle video recording
        if self.teleop_control and self.control_dict['is_recording']:
            try:
                current_time = time.time()
                # if current_time - self.last_frame_time > 1.0 / self.frame_rate:
                    # print('video shape: ', combined_frame.shape)
                self.video_writer.write(combined_frame)
                self.last_frame_time = current_time
                self.timestamps.append(current_time)
            except Exception as e:
                print(f"Error during video writing: {e}")

        # UI Elements
        cv2.circle(left_image, self.record_position, self.radius, self.color, self.thickness)
        cv2.circle(right_image, self.record_position, self.radius, self.color, self.thickness)
        cv2.circle(left_image, self.gesture_position, self.radius, self.gesture_color, self.gesture_thickness)
        cv2.circle(right_image, self.gesture_position, self.radius, self.gesture_color, self.gesture_thickness)
        cv2.putText(left_image, str(self.episode), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)
        cv2.putText(right_image, str(self.episode), self.episode_position, self.font, self.font_scale, self.txt_color, self.txt_thickness)

        rgb = np.hstack((left_image, right_image))
        if self.teleop_control:
            np.copyto(self.img_array, rgb)
            if self.is_viewer:
                cv2.imshow('Left Camera', left_image_)
                cv2.imshow('Right Camera', right_image_)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.end()  
                    exit(0)                
 
    def get_camera_image(self, cam_id):
        """Captures and returns an image from the specified camera"""        
        cam = self.left_cam if cam_id == 0 else self.right_cam
        mj.mjv_updateScene(self.model, self.data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)

        rgb_img = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
        mj.mjr_render(self.viewport, self.scene, self.context)
        mj.mjr_readPixels(rgb_img, None, self.viewport, self.context)
            
        return np.flipud(rgb_img)
    
    def set_torque_servo(self, actuator_indices, flag): 
        self.model.actuator_gainprm[actuator_indices, 0] = flag
    
    def fetch_pos(self, body_name):
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        return self.model.body(body_id).pos

    def reset_env_randomize(self, record_obj_pose=True, randomize_lighting=False):
        """Randomize environment settings for new episode"""
        # Randomize lighting
        if randomize_lighting:
            self._randomize_lighting()

        # Hardcoded for tap -- 
        if self.tasktype == 'tap':
            self._tap_reset()
            return
               
        # Create a 7-digit array
        number_of_obj_dof = (self.data.qpos.shape[0] - self.robot_joint_ids.shape[0])
        number_of_objects = number_of_obj_dof // 7
        new_array = np.zeros(7 * number_of_objects)  # Initialize a 7-digit array with zeros

        if self.tasktype == 'microwave':
            microwave_body_pos = self.fetch_pos('microwave_object')
            if not hasattr(self, '_init_microwave_pos'):
                self._init_microwave_pos = microwave_body_pos.copy()
            # microwave_body.pos = np.array([0.79, 0.01, 1.33])
            x_offset = np.random.normal(0, 0.01)  # Gaussian noise for the second element
            y_offset = np.random.normal(0, 0.02)  # Gaussian noise for the third element
            microwave_body_pos[0] = self._init_microwave_pos[0] + x_offset
            microwave_body_pos[1] = self._init_microwave_pos[1] + y_offset
            new_array = np.zeros(8) # 1dof for microwave hinge, 7dof for object

        # Randomize object pose
        for i in range(number_of_objects):
            new_array[7*i+0] = np.random.normal(0, 0.02)  # Gaussian noise for the second element
            new_array[7*i+1] = np.random.normal(0, 0.02)  # Gaussian noise for the third element

            # Get initial quaternion for this object
            initial_quat = self.initial_object_qpos[self.cur_env][7*i+3:7*i+7]

            # add noise to orientation about z axis
            z = np.random.normal(0, np.pi)
            quat = safe_as_quat(R.from_rotvec(np.array([0, 0, z])))
            quat_rot = safe_from_quat(quat)
            quat_init = safe_from_quat(initial_quat)

            if self.tasktype == 'microwave':
                quat_init = safe_from_quat(self.initial_object_qpos[self.cur_env][7*i+4:7*i+8])
                self.initial_object_qpos[self.cur_env][7*i+4:7*i+8] = safe_as_quat((quat_rot * quat_init))
            else:
                self.initial_object_qpos[self.cur_env][7*i+3:7*i+7] = safe_as_quat((quat_rot * quat_init))
        
        if record_obj_pose:
            obj_pose = {'object_qpos': list(self.initial_object_qpos[self.cur_env] + new_array)}
            with open(self.obj_pose_log_path, "w") as f:
                json.dump(obj_pose, f)
            
        print('object_qpos: ', self.initial_object_qpos[self.cur_env] + new_array)

        # Sample from Gaussian distribution only for 1st and 2nd dof
        self.data.qpos[-number_of_obj_dof:] = self.initial_object_qpos[self.cur_env] + new_array
        self.data.qvel[-number_of_obj_dof:] = 0
        mj.mj_forward(self.model, self.data)
        
    def _tap_reset(self):
        tap_body_pos = self.fetch_pos('tap_object')
        if not hasattr(self, '_init_tap_pos'):
            self._init_tap_pos = tap_body_pos.copy()
        # tap_body.pos = np.array([0.8, 0, -0.3])
        x_offset = np.random.normal(0, 0.03)  # Gaussian noise for the second element
        y_offset = np.random.normal(0, 0.03)  # Gaussian noise for the third element
        tap_body_pos[0] = self._init_tap_pos[0] + x_offset
        tap_body_pos[1] = self._init_tap_pos[1] + y_offset
        mj.mj_forward(self.model, self.data)
        
    def _randomize_lighting(self):
        for i in range(self.model.nlight):
            # Store initial light positions if not already stored
            if not hasattr(self, '_init_light_pos'):
                self._init_light_pos = self.model.light_pos.copy()
            
            # Randomize light position relative to initial position
            pos_noise = np.random.uniform(-0.5, 0.5, 3)
            self.model.light_pos[i] = self._init_light_pos[i] + pos_noise
            
            # Randomize light color/intensity (RGB values between 0.5 and 1.0)
            self.model.light_ambient[i] = np.random.uniform(0.1, 0.3, 3)
            self.model.light_diffuse[i] = np.random.uniform(0.5, 1.0, 3)
            self.model.light_specular[i] = np.random.uniform(0.5, 1.0, 3)

    def end(self):
        if self.is_viewer:
            glfw.destroy_window(self.hidden_window)
        glfw.terminate()
