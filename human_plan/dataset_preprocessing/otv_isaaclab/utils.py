import os
import h5py
import numpy as np

from scipy.spatial.transform import Rotation as R

import cv2

from pathlib import Path

from tqdm import tqdm 

from human_plan.utils.visualization import (
   project_points,
   plot_points,
   plot_hand_sim,
   plot_hand_mano
)

from human_plan.utils.transformation import (
  homogeneous_coord,
  CAM_AXIS_TRANSFORM,
  ISAAC_LAB_CAMERA_FRAME_CHANGE
)

# Close-Drawer      Insert-And-Unload-Cans   Orient-Cube         Press-Gamepad-Blue       Push-Box                         Sort-Cans
#  egovla_sim_data   Insert-Cans              Orient-Pour-Balls   Press-Gamepad-Blue-Red  'Sim Data Collection List.xlsx'   Stack-Cubes
#  Flip-Mug          Open-Drawer              Pour-Balls          Press-Gamepad-Red       'Sim Data Info.docx'              Stack-Cubes-From-Drawer
LANGUAGE_MAPPING = {
  "Pour-Balls": "pour balls in cup into bowl",
  "Pick-Place-Safe-Drawer": "Pick up object, place it in drawer and close drawer",
  "Stack-Cubes": "stack one cube on another cube",
  "Dump-Apple": "dump aplle into bowl",
  "Push-Box": "push box to the marker",
  "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
  "Sort-Cans-Old": "Put sprite cans to the left box, and orange cans to the right box",
  "Orient-Cube": "Orient the cube to the rotation as show in the observation",
  "Insert-Cans": "Insert cans into the boxes",
  "Close-Drawer": "Close the opened drawer",
  "Open-Drawer": "Open the closed drawer", 
  "Stack-Cubes-From-Drawer": "Open the closed drawer, and stack on cube on another cube on the desk",
  "Insert-And-Unload-Cans": "Insert the left can into the slot and insert the right can into the slot, unload the left cans andd then unload the right cans",
  "Insert-And-Unload-Cans-Old": "Insert the right can into the slot and insert the left can into the slot, unload the right cans andd then unload the left cans",
  "Orient-Pour-Balls": "Reorient the mug and pour balls in cup into bowl",
  "Press-Gamepad-Blue": "Press the blue button on the gamepad",
  "Press-Gamepad-Red": "Press the red button on the gamepad",
  "Press-Gamepad-Blue-Red": "Press the blue button on the gamepad then press the red button",
  "Flip-Mug": "Flip the mug",
  "Unload-Cans": "unload the right cans and then unload the left cans",
  "Press-Gamepad-Red-Blue": "Press the red button on the gamepad then press the blue button", 
  "Stack-Single-Cube": "stack right cube on the middle cube",
  "Stack-Single-Cube-From-Drawer": "Open the closed drawer, and stack the cube on the cube in front of the drawer",
  "Stack-Can": "put can on the saucer",
  "Stack-Can-From-Drawer": "Put can on the saucer, and Close the drawer",
  "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
  "Open-Laptop": "open the laptop",
}

LANGUAGE_MAPPING_CATEGORY = {
  "Pour-Balls": 0,
  "Push-Box": 0,
  "Close-Drawer": 0,
  "Open-Drawer": 0, 
  "Flip-Mug": 0,
  "Stack-Can": 0,
  "Open-Laptop": 0,
  "Sort-Cans": 1,
  "Unload-Cans": 1,
  "Insert-Cans": 1,
  "Insert-And-Unload-Cans": 1,
  "Stack-Can-Into-Drawer": 1,
}

def get_sort_cans_language_instruction(data_dict):
  if data_dict["sort_success"] == 0.0:
    return "put right sprite cans to the left box"
  elif data_dict["sort_success"] == 1.0:
    return "put left sprite cans to the left box"
  elif data_dict["sort_success"] == 2.0:
    return "put right orange cans to the right box"
  elif data_dict["sort_success"] == 3.0:
    return "put right orange cans to the right box"

def get_insert_cans_language_instruction(data_dict):
  if data_dict["insert_success"] == 0.0:
    return "insert right can to the box"
  elif data_dict["insert_success"] == 1.0:
    return "insert left can to the box"

def get_unload_cans_language_instruction(data_dict):
  if data_dict["unload_success"] == 0.0:
    return "unload right can"
  elif data_dict["unload_success"] == 1.0:
    return "unload left can"

def get_insert_and_unload_cans_language_instruction(data_dict):
  if data_dict["insert_success"] == 0.0:
    return "insert left can to the box"
  elif data_dict["insert_success"] == 1.0:
    return "insert right can to the box"
  elif data_dict["unload_success"] == 0.0:
    return "unload left can"
  elif data_dict["unload_success"] == 1.0:
    return "unload right can"

# def get_stack_can_into_drawer_cans_language_instruction(data_dict):
#   return

LANGUAGE_MAPPING_NEW = {
  "Pour-Balls": "pour balls in cup into bowl",
  "Push-Box": "push box to the marker",
  "Close-Drawer": "Close the opened drawer",
  "Open-Drawer": "Open the closed drawer",
  "Flip-Mug": "Flip the mug",
  "Stack-Can": "put can on the saucer",
  "Open-Laptop": "open the laptop",
  "Sort-Cans": get_sort_cans_language_instruction,
  "Insert-Cans": get_insert_cans_language_instruction,
  "Insert-And-Unload-Cans": get_insert_and_unload_cans_language_instruction,
  "Unload-Cans": get_unload_cans_language_instruction,
  # "Stack-Can-Into-Drawer": get_stack_can_into_drawer_cans_language_instruction,
  "Stack-Can-Into-Drawer": "Put can on the saucer, and Close the drawer",
}

def extract_success_datadict(data_dict, index):
  single_step_dict = {}
  for key in data_dict["success_label"].keys():
    if "success" in key:
      single_step_dict[key] = data_dict["success_label"][key][index]
      # print(key)
      # print(data_dict[key])
  return single_step_dict

# LANGUAGE_MAPPING_ = {
#   "Sort-Cans": "sort_success",
#   "Insert-Cans": "insert_success",
#   "Unload-Cans": "unload_success",
#   "Insert-And-Unload-Cans": "insert_success",
# }



# Qunicy Data:
main_cam_trans = np.array([0.09, 0.0, 1.7])

# Isaac Lab Convention WXYZ
main_cam_quat = (0.66446, 0.24184, -0.24184, -0.664464)
main_cam_quat_xyzw = (0.24184, -0.24184, -0.664464, 0.66446)
# main_cam_quat = (0.9063077870366499, 0.0, 0.42261826174069944, 0.0)
# main_cam_quat_xyzw = (0.0, 0.42261826174069944, 0.0, 0.9063077870366499, )
main_cam_rotmat = R.from_quat(main_cam_quat_xyzw)
main_cam_rotmat = main_cam_rotmat.as_matrix()

main_cam_transformation = np.eye(4)
main_cam_transformation[:3, :3] = ISAAC_LAB_CAMERA_FRAME_CHANGE @ main_cam_rotmat
main_cam_transformation[:3, -1] = main_cam_trans

main_intrinsics = np.array([
  [488.6662,   0.0000, 640.0000],
  [  0.0000, 488.6662, 360.0000],
  [  0.0000,   0.0000,   1.0000]
])

def to_pose(ee_pose):
  seq_len = ee_pose.shape[0]
  ee_pose_out = np.zeros((seq_len, 4, 4))
  # WXYZ to XYZW
  # print(ee_pose.shape)
  rot = np.concatenate([
    ee_pose[:, 4:],
    ee_pose[:, 3:4]
  ], axis=1)
  # print(rot.shape)
  rot = R.from_quat(rot)
  ee_pose_out[:, :3, :3] = rot.as_matrix()
  ee_pose_out[:, -1, -1] = 1
  ee_pose_out[:, :3, -1] = ee_pose[:, :3]
  # print(ee_pose_out[0])
  # print(ee_pose_out.shape)
  return ee_pose_out



from human_plan.dataset_preprocessing.utils.mano_utils import (
  obtain_mano_parameters_otv_inspire_hand,
  obtain_mano_parameters_otv_inspire_hand_mano_rot,
  obtain_mano_parameters_otv_inspire_hand_full_optimization,
  mano_left,
  mano_right,
  mano_to_inspire_mapping
)

def find_hdf5_files(task_root):
  hdf5_files = []
  for dirpath, _, filenames in os.walk(task_root):
    for filename in filenames:
      if filename.endswith(".hdf5"):
        hdf5_files.append(os.path.relpath(
          os.path.join(dirpath, filename))
        )
  return hdf5_files

def get_all_seqs(
    dataset_root, seq_skip=1, return_task_list=False
):
  task_list = sorted(
    [p for p in os.listdir(dataset_root) if not os.path.isfile(os.path.join(dataset_root, p))]
  )
  print(task_list)
  
  seq_list = []
  for task in task_list:
    hdf5_files =sorted(
      [f for  f in os.listdir(os.path.join(dataset_root, task)) if f.endswith(".hdf5")]
    )
    # hdf5
    # seq_list[task] = hdf5_files
    hdf5_files = hdf5_files[::seq_skip]
    seq_list += [(task, hdf5_file) for hdf5_file in hdf5_files]
  # print(seq_list)
  if return_task_list:
    return seq_list, task_list
  return seq_list

TASKS_WITH_50 = [
  "Stack-Can-From-Drawer",
  "Close-Drawer",
]

def filter_seqs_by_room(seq_list, room_idx):
  filtered_seq_list = []
  for task, hdf5_file in seq_list:
    # example: episode_270.hdf5
    seq_id = int(hdf5_file[8:-5])
    split_sum = 300
    if task in TASKS_WITH_50:
      split_sum = 150
    if seq_id // split_sum + 1 == room_idx:
      filtered_seq_list.append((task, hdf5_file))

  return filtered_seq_list

def filter_seqs_by_table(seq_list, table_idx):
  filtered_seq_list = []
  for task, hdf5_file in seq_list:
    # example: episode_270.hdf5
    seq_id = int(hdf5_file[8:-5])

    split_sum = 300
    if task in TASKS_WITH_50:
      split_sum = 150

    with_in_room_id = seq_id % split_sum

    if with_in_room_id // (split_sum // 3) + 1 == table_idx:
      filtered_seq_list.append((task, hdf5_file))

  return filtered_seq_list

def get_single_task_seqs(
    dataset_root, task_name, seq_skip=1
):
  task_list = [
    p for p in os.listdir(dataset_root) if (not os.path.isfile(os.path.join(dataset_root, p)) and p == task_name)
  ]
  print(task_list)
  seq_list = []
  for task in task_list:
    hdf5_files =[f for  f in os.listdir(os.path.join(dataset_root, task)) if f.endswith(".hdf5")]
    # hdf5
    hdf5_files = hdf5_files[::seq_skip]
    seq_list += [(task, hdf5_file) for hdf5_file in hdf5_files]
  return seq_list

def robot_action_to_mano_action(
  left_finger_tip, right_finger_tip,
  left_ee_pose, right_ee_pose
):
  mano_kps3d_right = obtain_mano_parameters_otv_inspire_hand_mano_rot(
    mano_right, right_finger_tip, right_ee_pose, is_right=True
  )

  mano_kps3d_left = obtain_mano_parameters_otv_inspire_hand_mano_rot(
    mano_left, left_finger_tip, left_ee_pose, is_right=False
  )
  return {
    "left_hand_pose": {
      "mano_kps3d": mano_kps3d_left["mano_kp_predicted"],
      "mano_parameters": mano_kps3d_left["optimized_mano_parameters"],
      "mano_rot": mano_kps3d_left["optimized_mano_rot"],
      "mano_trans": mano_kps3d_left["optimized_mano_trans"],
      "relative_transformation": mano_kps3d_left["optimized_mano_t_pose"]
    }, 
    "right_hand_pose": {
      "mano_kps3d": mano_kps3d_right["mano_kp_predicted"],
      "mano_parameters": mano_kps3d_right["optimized_mano_parameters"],
      "mano_rot": mano_kps3d_right["optimized_mano_rot"],
      "mano_trans": mano_kps3d_right["optimized_mano_trans"],
      "relative_transformation": mano_kps3d_right["optimized_mano_t_pose"]
    }
  }


def to_cam_frame(points):
  points = CAM_AXIS_TRANSFORM @ np.linalg.inv(main_cam_transformation) @ homogeneous_coord(
    points
  )[..., np.newaxis]
  points = points[..., :3, 0]
  points = points.astype(np.float32)
  return points

def pose_to_cam_frame(poses):
  poses = CAM_AXIS_TRANSFORM @ np.linalg.inv(main_cam_transformation) @ poses.reshape(-1, 4, 4)
  poses = poses.astype(np.float32)

  # print(poses.shape)
  return poses

def obtain_relative_transformation(
  transformation,
):
  """
  Compute the relative transformations between consecutive transformation matrices.

  Parameters:
  transformations (list of np.ndarray): A list of 4x4 transformation matrices.

  Returns:
  list of np.ndarray: A list of 4x4 relative transformation matrices.
  """
  # if len(transformations) < 2:
  #     raise ValueError("At least two transformations are required to compute relative transformations.")

  # relative_transforms = [np.eye(4)]
  # num_transformations = transformation.shape[0]

  T_inv = np.linalg.inv(transformation)
  T_relative = T_inv[:-1] @ transformation[1:]
  T_relative = np.concatenate([
    np.eye(4).reshape(1, 4, 4),
    T_relative
  ], axis=0)
  # for i in range(num_transformations - 1):
  #     T1_inv = np.linalg.inv(transformations[i])
  #     T_relative = np.dot(T1_inv, transformations[i + 1])
  #     relative_transforms.append(T_relative)

  return T_relative

def load_episode_data(
    dataset_root,
    task_name,
    seq_name, include_depth=False,
    clip_starting=20,
    clip_ending=None,
    filter_by_success=False,
):
  # Construct the file path for the episode's HDF5 file
  episode_path = os.path.join(
    dataset_root,
    task_name,
    seq_name
  )
  print(episode_path)

  # Dictionary to store loaded data
  data = {}

  # Open the HDF5 file in read mode
  with h5py.File(episode_path, 'r') as data_file:
    # Check if 'sim' attribute is present
    data['sim'] = data_file.attrs.get('sim', None)
    print(data_file.keys())
    print(data_file.attrs.keys())
    # print(data['sim'].keys())
    # Parse the 'observations' group
    obs_group = data_file.get('observations')
    # print(obs_group.keys())
    if "success" in obs_group:
      print(obs_group["success"][:])
      first_success_index = np.where(obs_group["success"][:] == 1)[0]
      if first_success_index.size > 0:
        first_success_index = first_success_index[0]
      else:
        if filter_by_success:
          return None
      print(first_success_index)  # Output: 3
    else:
      assert not filter_by_success, "No success data found in the file."
    
    if filter_by_success:
      # Clip the data to only include frames after the first success
      clip_ending = first_success_index
    if obs_group is not None:
      # Load image data if it exists
      image_group = obs_group.get('images')
      if image_group is not None:
        data['images'] = {}
        data['images']['main'] = image_group['main'][clip_starting:clip_ending]
        if include_depth and 'main_d' in image_group:
          data['images']['main_d'] = image_group['main_d'][()]
      # print(obs_group.keys())
      # Load other datasets in 'observations'
      data['observations'] = {
          'qpos': obs_group['qpos'][clip_starting:clip_ending],
          'qvel': obs_group['qvel'][clip_starting:clip_ending],
          # 'left_ee_pose': obs_group['left_ee_pose'][clip_starting:clip_ending],
          # 'right_ee_pose': obs_group['right_ee_pose'][clip_starting:clip_ending],
          'left_finger_tip_pos': obs_group['left_finger_tip_pos'][clip_starting:clip_ending],
          'right_finger_tip_pos': obs_group['right_finger_tip_pos'][clip_starting:clip_ending],
          # 'inside_drawer': obs_group['inside_drawer'][clip_starting:clip_ending],
          # 'inside_safe': obs_group['inside_safe'][clip_starting:clip_ending],
          # 'drawer_close': obs_group['drawer_close'][clip_starting:clip_ending],
          # 'safe_close': obs_group['safe_close'][clip_starting:clip_ending],
          # 'left_hand_contact_force': obs_group['left_hand_contact_force'][clip_starting:clip_ending],
          # 'right_hand_contact_force': obs_group['right_hand_contact_force'][clip_starting:clip_ending],
          # 'object_pose': obs_group['object_pose'][clip_starting:clip_ending]
      }
      if "success" in obs_group:
        data["success_label"] = {}
        for key in obs_group.keys():
          if "success" in key:
            print(key)
            data["success_label"][key] = obs_group[key][clip_starting:clip_ending]

      if 'left_ee_pose' in obs_group:
        data['observations']['left_ee_pose'] = obs_group['left_ee_pose'][clip_starting:clip_ending]
        data['observations']['right_ee_pose'] = obs_group['right_ee_pose'][clip_starting:clip_ending]
      else:
        data['observations']['left_ee_pose'] = obs_group['left_target_ee_pose'][clip_starting:clip_ending]
        data['observations']['right_ee_pose'] = obs_group['right_target_ee_pose'][clip_starting:clip_ending]
      # EE Pose
      data["left_ee_pose_cam_frame"] = pose_to_cam_frame(
        to_pose(data["observations"]["left_ee_pose"])
      )
      data["right_ee_pose_cam_frame"] = pose_to_cam_frame(
        to_pose(data["observations"]["right_ee_pose"])
      )

      # Finger tip pos
      data["left_finger_tip_pos_cam_frame"] = to_cam_frame(
        data["observations"]["left_finger_tip_pos"]
      )
      data["right_finger_tip_pos_cam_frame"] = to_cam_frame(
        data["observations"]["right_finger_tip_pos"]
      )
      mano_action = robot_action_to_mano_action(
        left_finger_tip=data["left_finger_tip_pos_cam_frame"], 
        right_finger_tip=data["right_finger_tip_pos_cam_frame"],
        left_ee_pose=data["left_ee_pose_cam_frame"],
        right_ee_pose=data["right_ee_pose_cam_frame"]
      )

      left_relative_transformation = obtain_relative_transformation(
        mano_action["left_hand_pose"]["relative_transformation"]
      )

      right_relative_transformation = obtain_relative_transformation(
        mano_action["right_hand_pose"]["relative_transformation"]
      )


      data["right_mano_parameters"] = mano_action["right_hand_pose"]["mano_parameters"]
      data["right_mano_kps3d"] = mano_action["right_hand_pose"]["mano_kps3d"]
      data["right_mano_rot"] = mano_action["right_hand_pose"]["mano_rot"]
      data["right_mano_trans"] = mano_action["right_hand_pose"]["mano_trans"]

      data["left_mano_parameters"] = mano_action["left_hand_pose"]["mano_parameters"]
      data["left_mano_kps3d"] = mano_action["left_hand_pose"]["mano_kps3d"]
      data["left_mano_rot"] = mano_action["left_hand_pose"]["mano_rot"]
      data["left_mano_trans"] = mano_action["left_hand_pose"]["mano_trans"]

      # print(data["right_ee_pose_cam_frame"][:, :3, -1].shape)
      data["right_mano_ee_2d"] = project_points(data["right_ee_pose_cam_frame"][:, :3, -1], main_intrinsics)
      # print(data["right_mano_ee_2d"].shape)
      data["left_mano_ee_2d"] = project_points(data["left_ee_pose_cam_frame"][:, :3, -1], main_intrinsics)

      data["left_mano_ee_relative_transformation"] = left_relative_transformation
      data["right_mano_ee_relative_transformation"] = right_relative_transformation

      data["left_mano_ee_relative_trans"] = left_relative_transformation[:, :3, -1]
      data["right_mano_ee_relative_trans"] = right_relative_transformation[:, :3, -1]

      left_relative_rotation = R.from_matrix(left_relative_transformation[:, :3, :3])
      left_relative_rotation = left_relative_rotation.as_rotvec()
      right_relative_rotation = R.from_matrix(right_relative_transformation[:, :3, :3])
      right_relative_rotation = right_relative_rotation.as_rotvec()

      data["left_mano_ee_relative_rot"] = left_relative_rotation
      data["right_mano_ee_relative_rot"] = right_relative_rotation

    # Load 'action' dataset
    data['action'] = data_file['action'][clip_starting:clip_ending]
  return data


def parse_single_seq_image(
    dataset_root,
    task_name,
    seq_name,
    image_w,
    image_h,
    clip_starting=20
    # frame_skip,
    # future_len,
):
  episode_data = load_episode_data(
    dataset_root,
    task_name,
    seq_name,
    clip_starting=clip_starting
  )
  sequence_len = episode_data["images"]["main"].shape[0]

  seq_data = []

  # for idx in range(0, sequence_len, frame_skip):
  for idx in range(0, sequence_len):
    frame = episode_data["images"]["main"][idx]
    # frame = episode_data["images"]["main"][idx][:, :640, :]
    frame = cv2.resize(frame, (image_w, image_h))
    current_data = {
      "seq_name": f"{task_name}/{seq_name}",
      "frame_count": idx,
      "rgb_obs": frame
    }
    seq_data.append(current_data)
  return seq_data



OTV_SIM_IMAGE_WIDTH = 1280
OTV_SIM_IMAGE_HEIGHT = 720

def parse_single_seq_hand(
    dataset_root,
    task_name,
    seq_name,
    sample_skip,
    future_len,
    frame_skip=4,
    clip_starting=20,
    clip_ending=None,
    start_idx=0,
    filter_by_success=False,
    use_per_step_instruction=False,
):
  episode_data = load_episode_data(
    dataset_root,
    task_name,
    seq_name,
    clip_starting=clip_starting,
    clip_ending=clip_ending,
    filter_by_success=filter_by_success
  )
  if episode_data is None:
    # Only if the filter_by_success is True and the augmented data failed
    return []
  sequence_len = episode_data["images"]["main"].shape[0]

  seq_data = []
  # Let's just do frame_skip == 1, only consider sample skip here
  for idx in range(start_idx, sequence_len, sample_skip):
    future_end_idx = idx + future_len * frame_skip

    sample = {}
    # for hand in ["left", "right"]:
    #   sample[]

    language_label = None
    if use_per_step_instruction:
      single_step_sucess_dict = extract_success_datadict(episode_data, idx)
      if isinstance(LANGUAGE_MAPPING_NEW[task_name], str):
        language_label = LANGUAGE_MAPPING_NEW[task_name]
      else:
        language_label = LANGUAGE_MAPPING_NEW[task_name](single_step_sucess_dict)
    else:
      language_label = LANGUAGE_MAPPING[task_name]


    sample = dict(
      # World Frame
      # language_label = LANGUAGE_MAPPING[task_name],
      language_label = language_label,
      seq_name = f"{task_name}/{seq_name}",
      frame_count = idx,

      raw_width = OTV_SIM_IMAGE_WIDTH,
      raw_height = OTV_SIM_IMAGE_HEIGHT,

      curent_qpos = episode_data["observations"]["qpos"][idx, :],

      current_left_ee_pose = episode_data["observations"]["left_ee_pose"][idx, :],
      current_right_ee_pose = episode_data["observations"]["right_ee_pose"][idx, :],

      current_left_finger_tip_pos = episode_data["observations"]["left_finger_tip_pos"][idx, :],
      current_right_finger_tip_pos = episode_data["observations"]["right_finger_tip_pos"][idx, :],

      future_left_ee_pose = episode_data["observations"]["left_ee_pose"][idx:future_end_idx:frame_skip, :],
      future_right_ee_pose = episode_data["observations"]["right_ee_pose"][idx:future_end_idx:frame_skip, :],

      future_left_finger_tip_pos = episode_data["observations"]["left_finger_tip_pos"][idx:future_end_idx:frame_skip, :],
      future_right_finger_tip_pos = episode_data["observations"]["right_finger_tip_pos"][idx:future_end_idx:frame_skip, :],

      # Cam Frame

      current_left_ee_cam_pose = episode_data["left_ee_pose_cam_frame"][idx, :],
      current_right_ee_cam_pose = episode_data["right_ee_pose_cam_frame"][idx, :],

      current_left_finger_tip_cam_pos = episode_data["left_finger_tip_pos_cam_frame"][idx, :],
      current_right_finger_tip_cam_pos = episode_data["right_finger_tip_pos_cam_frame"][idx, :],

      future_left_ee_cam_pose = episode_data["left_ee_pose_cam_frame"][idx:future_end_idx:frame_skip, :],
      future_right_ee_cam_pose = episode_data["right_ee_pose_cam_frame"][idx:future_end_idx:frame_skip, :],

      future_left_finger_tip_cam_pos = episode_data["left_finger_tip_pos_cam_frame"][idx:future_end_idx:frame_skip, :],
      future_right_finger_tip_cam_pos = episode_data["right_finger_tip_pos_cam_frame"][idx:future_end_idx:frame_skip, :],

      # Cam Frame 

      current_left_mano_rot = episode_data["left_mano_rot"][idx, :],
      current_right_mano_rot = episode_data["right_mano_rot"][idx, :],

      current_left_mano_trans = episode_data["left_mano_trans"][idx, :],
      current_right_mano_trans = episode_data["right_mano_trans"][idx, :],

      current_left_mano_parameters = episode_data["left_mano_parameters"][idx, :],
      current_right_mano_parameters = episode_data["right_mano_parameters"][idx, :],

      current_left_mano_kps3d = episode_data["left_mano_kps3d"][idx, :],
      current_right_mano_kps3d = episode_data["right_mano_kps3d"][idx, :],

      current_left_mano_ee_2d = episode_data["left_mano_ee_2d"][idx, :],
      current_right_mano_ee_2d = episode_data["right_mano_ee_2d"][idx, :],

      future_left_mano_rot = episode_data["left_mano_rot"][idx:future_end_idx:frame_skip, :],
      future_right_mano_rot = episode_data["right_mano_rot"][idx:future_end_idx:frame_skip, :],

      future_left_mano_trans = episode_data["left_mano_trans"][idx:future_end_idx:frame_skip, :],
      future_right_mano_trans = episode_data["right_mano_trans"][idx:future_end_idx:frame_skip, :],

      future_left_mano_parameters = episode_data["left_mano_parameters"][idx:future_end_idx:frame_skip, :],
      future_right_mano_parameters = episode_data["right_mano_parameters"][idx:future_end_idx:frame_skip, :],

      future_left_mano_kps3d = episode_data["left_mano_kps3d"][idx:future_end_idx:frame_skip, :],
      future_right_mano_kps3d = episode_data["right_mano_kps3d"][idx:future_end_idx:frame_skip, :],

      future_left_mano_ee_2d = episode_data["left_mano_ee_2d"][idx:future_end_idx:frame_skip, :],
      future_right_mano_ee_2d = episode_data["right_mano_ee_2d"][idx:future_end_idx:frame_skip, :],

      future_left_mano_ee_relative_trans = episode_data["left_mano_ee_relative_trans"][idx:future_end_idx:frame_skip, :],
      future_right_mano_ee_relative_trans = episode_data["right_mano_ee_relative_trans"][idx:future_end_idx:frame_skip, :],

      future_left_mano_ee_relative_rot = episode_data["left_mano_ee_relative_rot"][idx:future_end_idx:frame_skip, :],
      future_right_mano_ee_relative_rot = episode_data["right_mano_ee_relative_rot"][idx:future_end_idx:frame_skip, :],
      # data["right_mano_ee_relative_rot"] = right_relative_rotation
    )
    flag_shape = list(sample["future_right_mano_ee_2d"].shape)
    flag_shape[-1] = 1

    sample["action"] = episode_data["action"][idx]

    sample["future_left_flag"] = np.ones(flag_shape)
    sample["future_right_flag"] = np.ones(flag_shape)

    seq_data.append(sample)
  return seq_data

def visualize_single_seq_otv(
    single_seq_data,
    save_root,
    seq_name
):
    output_path = os.path.join(
       save_root, seq_name
    )
    Path(output_path).mkdir(parents=True, exist_ok=True)

    seq_length = single_seq_data["images"]["main"].shape[0]
    for idx in tqdm(range(0, seq_length, 20)):
       # TO CV2 convention
      #  print()
      # rgb_obs = single_seq_data["images"]["main"][idx][..., :640, ::-1]
      rgb_obs = single_seq_data["images"]["main"][idx][..., ::-1]
      # 3
      left_ee = single_seq_data["left_ee_pose_cam_frame"][idx, :3, -1].reshape(1, 3)
      right_ee = single_seq_data["right_ee_pose_cam_frame"][idx, :3, -1].reshape(1, 3)
      left_tip = single_seq_data["left_finger_tip_pos_cam_frame"][idx]
      right_tip = single_seq_data["right_finger_tip_pos_cam_frame"][idx]
      # # 3
      left_hand = np.concatenate([
         left_ee, left_tip
      ], axis=0)

      right_hand = np.concatenate([
         right_ee, right_tip
      ], axis=0)

      # left_hand_mano = single_seq_data["left_mano_kps3d"][idx, mano_to_inspire_mapping]
      # right_hand_mano = single_seq_data["right_mano_kps3d"][idx, mano_to_inspire_mapping]

      left_hand_mano = single_seq_data["left_mano_kps3d"][idx]
      right_hand_mano = single_seq_data["right_mano_kps3d"][idx]

      # print("-" * 20)
      left_hand_proj = project_points(left_hand, main_intrinsics)
      right_hand_proj = project_points(right_hand, main_intrinsics)


      left_hand_mano_proj = project_points(left_hand_mano, main_intrinsics)
      right_hand_mano_proj = project_points(right_hand_mano, main_intrinsics)

      rgb_obs = np.array(rgb_obs, dtype=np.uint8)
      # Ensure the array is contiguous
      rgb_obs = np.ascontiguousarray(rgb_obs)
      rgb_obs_mano = rgb_obs.copy()

      rgb_obs = plot_hand_sim(
         left_hand_proj, rgb_obs, (0, 0, 255)
      )
      rgb_obs = plot_hand_sim(
         right_hand_proj, rgb_obs, (0, 255, 0)
      )

      rgb_obs_mano = plot_hand_mano(
        left_hand_mano_proj, rgb_obs_mano, (0, 0, 255)
      )
      rgb_obs_mano = plot_hand_mano(
        right_hand_mano_proj, rgb_obs_mano, (0, 255, 0)
      )
      # print(np.concatenate([
      #     rgb_obs, rgb_obs_mano
      #   ], axis=1).shape)
      # print(rgb_obs)
      cv2.imwrite(
        os.path.join(output_path, "demo_robot_{}.jpg".format(idx)), 
        rgb_obs
      )

      cv2.imwrite(
        os.path.join(output_path, "demo_mano_{}.jpg".format(idx)), 
        rgb_obs_mano
      )

if __name__ == "__main__":
  dataset_root = "/mnt/data3/data/OTV_AUG_v2/"
  # seq_name = "Pick-Place-Safe-Drawer/episode_0.hdf5"
  seq_name = "Insert-And-Unload-Cans/episode_0.hdf5"
  seq_data =load_episode_data(
    dataset_root,
    "Insert-And-Unload-Cans",
    "episode_0.hdf5",
    clip_starting=0,
    filter_by_success=True
  )

  save_root = "playground/dataset_vis/OTV/sim_fix_retarget"

  visualize_single_seq_otv(
    seq_data,
    save_root,
    seq_name
  )