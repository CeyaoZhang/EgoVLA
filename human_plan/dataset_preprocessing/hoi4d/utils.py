from manopth.manolayer import ManoLayer
import pickle
import torch
import torch.nn as nn
import os
import cv2
import zipfile
import subprocess
import natsort
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

from human_plan.dataset_preprocessing.utils.transformations import (
    transform_to_current_frame,
    transform_to_world_frame,
    project_set
)

object_mapping = [
    '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
    'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
    'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]

sequence_list_file = "sequence_list.txt"

HAND_POSE_PREFIX = "handpose/refinehandpose_right"
CAM_POSE_PREFIX = "HOI4D_annotations"
CAM_POSE_SUFFIX = "3Dseg/output.log"


def get_all_seqs():
  with open(
      os.path.join(os.path.dirname(__file__), sequence_list_file), "r"
  ) as f:
    sequence_lists = f.readlines()
  sequence_lists = [seq.strip() for seq in sequence_lists]
  sequence_lists = sorted(sequence_lists)
  return sequence_lists

def repeat_array(
    arr, repeat
):
  arr = np.repeat(
    arr, repeat, axis=0
  )
  return arr

def expand_first_dim(arr, scale_factor=6, kind='cubic'):
    """
    Expands the first dimension of a multi-dimensional array using interpolation.
    
    Parameters:
    - arr: np.ndarray (original array with shape (N, ...))
    - scale_factor: int (factor to expand the first dimension)
    - kind: str (interpolation type: 'linear', 'cubic', 'quadratic', etc.)
    
    Returns:
    - np.ndarray (new array with first dimension expanded)
    """
    original_shape = arr.shape
    N = original_shape[0]  # First dimension
    new_N = N * scale_factor  # New size for the first dimension

    x_old = np.linspace(0, 1, N)  # Original indices
    x_new = np.linspace(0, 1, new_N)  # New indices for interpolation

    # Reshape to (N, -1) to interpolate over flattened dimensions
    reshaped_arr = arr.reshape(N, -1)  # Flatten all other dimensions
    interpolated = np.zeros((new_N, reshaped_arr.shape[1]))

    from scipy.interpolate import interp1d

    for i in range(reshaped_arr.shape[1]):  # Iterate over all remaining dimensions
        f = interp1d(x_old, reshaped_arr[:, i], kind=kind, axis=0, fill_value="extrapolate")
        interpolated[:, i] = f(x_new)

    # Reshape back to original higher-dimensional format
    new_shape = (new_N,) + original_shape[1:]  # Adjust only the first dimension
    return interpolated.reshape(new_shape)

# Merge Hand / Image / Language Label

def parse_single_seq_image(
    dataset_root: str,
    seq_name: str,
    frame_skip: int,
    image_w: int,
    image_h: int,
    frame_repeat: int = 1,
):
  raw_video_path = os.path.join(
      dataset_root, seq_name, "align_rgb/image.mp4"
  )
  if not os.path.isfile(raw_video_path):
    return []

  # cnvert to 30 FPS
  # adjustfps_video_path = os.path.join(
  #     dataset_root, seq_name, "align_rgb/image_30fps.mp4"
  # )
  # # Run FFMPEG to align FPS
  # text_command = f'ffmpeg -y -hide_banner -loglevel warning -nostats -i {raw_video_path} -filter:v "fps=30" {adjustfps_video_path}'
  # subprocess.call(text_command, shell=True)

  # Capture the video from the file
  # cap = cv2.VideoCapture(adjustfps_video_path)
  cap = cv2.VideoCapture(raw_video_path)
  original_fps = cap.get(cv2.CAP_PROP_FPS)
  # print(original_fps)
  frame_count = 0
  seq_data = []

  while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # If frame is read correctly ret is True
    if not ret:
      break

    if frame_count % frame_skip != 0:
      continue

    # if image_w != 454 or image_h != 256:
    frame = cv2.resize(frame, (image_w, image_h))
    for repeat_idx in range(frame_repeat):
      current_data = {
          "seq_name": seq_name,
          "frame_count": frame_count + repeat_idx,
          "rgb_obs": frame,
      }
      seq_data.append(current_data)
    frame_count += frame_repeat
  # When everything done, release the capture
  cap.release()
  return seq_data



def get_3d_pos_hand(hand_info):
  manolayer = ManoLayer(
      mano_root='mano_v1_2/models',
      use_pca=False, ncomps=45, flat_hand_mean=True, side='right'
  )

  theta = nn.Parameter(torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0))
  beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0))
  trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0))
  hand_verts, hand_joints = manolayer(theta, beta)
  kps3d = hand_joints / 1000.0 + trans.unsqueeze(1)  # in meters
  hand_transformed_verts = hand_verts / 1000.0 + trans.unsqueeze(1)

  # theta shape (1, 48), first three dim -> axis angle 
  theta_np = np.array(hand_info['poseCoeff'])
  wrist_rot = theta_np[:3]
  # print(theta.shape)
  r = R.from_rotvec(wrist_rot)
  # Convert to rotation matrix
  wrist_rot = r.as_matrix()

  theta_hand_rotation = theta_np[3:].reshape(15, 3)
  hand_r = R.from_rotvec(theta_hand_rotation)
  hand_theta = hand_r.as_matrix()
  # print(hand_theta.shape)
  return kps3d, trans, hand_theta, wrist_rot

# Example: ZY20210800004/H4/C8/N12/S71/s02/T1
# ZY2021080000* refers to the camera ID.
# H* refers to human ID.
# C* refers to object class.
# mapping = [
#     '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
#     'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
#     'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
# ]
# N* refers to object instance ID.
# S* refers to the room ID.
# s* refers to the room layout ID.
# T* refers to the task ID.


def parse_sequence_info(seq_name):
  info_list = seq_name.split("/")

  info_dict = {}
  id_names = ["camera", "human", "object",
              "object_instance", "room", "room_layout", "task"]
  for id_name, data in zip(id_names, info_list):
    info_dict[id_name] = data

  return info_dict


def obtain_pose(full_sequence):

  full_sequence = [line.decode('utf-8').strip() for line in full_sequence]
  poses = []
  num_pose = len(full_sequence) // 5
  for i in range(num_pose):
    pose_data = full_sequence[i * 5: (i + 1) * 5]
    pose_data = [line.split() for line in pose_data[1:]]
    pose_data = np.array(pose_data, dtype=float)
    poses.append(pose_data)
  return poses


TASK_DEFINITION_FILE = os.path.join(
    os.path.dirname(__file__), "task_definitions.csv"
)
df = pd.read_csv(TASK_DEFINITION_FILE)

def obtain_cam_intrinsics(dataset_root, cam_id):
  camera_intrinsics_path = os.path.join(
      dataset_root, "camera_params", cam_id, "intrin.npy"
  )
  cam_intrinsics = np.load(camera_intrinsics_path)
  return cam_intrinsics

def get_task_label(seq_info_dict):
  row = df[df['Category ID'] == seq_info_dict["object"]]
  task_description = row[seq_info_dict["task"]].values[0]

  return {
      "verb": task_description,
      "noun": object_mapping[int(seq_info_dict["object"][1:])]
  }

# Merge Hand / Image / Language Label

HOI4D_WIDTH = 1920
HOI4D_HEIGHT = 1080

def parse_single_seq_hand(
    hand_pose_zip_ref,
    annotation_zip_ref,
    hand_zip_file_list,
    cam_zip_file_list,
    dataset_root: str,
    seq_name: str,
    frame_skip: int,
    future_len: int,
    sample_skip: int,
    frame_repeat: int = 1,
):

  subdir_path = seq_name + "/"
  hand_pose_subdir_path = os.path.join(
      "handpose/refinehandpose_right", subdir_path
  )
  cam_pose_path = os.path.join(
      CAM_POSE_PREFIX, subdir_path, CAM_POSE_SUFFIX
  )

  pose_files = [
      file for file in hand_zip_file_list if file.startswith(hand_pose_subdir_path) and not file.endswith('/')
  ]

  if len(pose_files) == 0:
    return []

  sequence_info_dict = parse_sequence_info(seq_name)
  task_language_label = get_task_label(sequence_info_dict)
  cam_intrinsics = obtain_cam_intrinsics(dataset_root, sequence_info_dict["camera"])
  pose_files = [(int(pf.split("/")[-1].split(".")[0]), pf) for pf in pose_files]
  pose_files = sorted(pose_files)
  pose_dict = {}
  for idx, pose_file in pose_files:
    pose_dict[idx] = pose_file

  # pose_files = natsort.natsorted(pose_files)

  hand_pose = []
  hand_trans = []
  hand_kps_2D = []
  hand_pose_theta = []
  hand_rot = []
  hand_valid = []
  # print(pose_files[:20])
  max_idx = pose_files[-1][0]
  for pose_file_idx in range(0, max_idx, frame_skip):
    # pose_files[::frame_skip]:
    if pose_file_idx not in pose_dict:
      hand_pose.append(np.zeros((21, 3), dtype=np.float32))
      hand_trans.append(np.zeros((3), dtype=np.float32))
      hand_kps_2D.append(np.zeros((21, 2), dtype=np.int64))
      hand_pose_theta.append(
        # np.squeeze(pose_theta.detach().cpu().numpy())
        # pose_theta
        np.zeros((15, 3, 3), dtype=np.float32)
      )
      hand_rot.append(
        # np.squeeze(wrist_rot.detach().cpu().numpy())
        # wrist_rot
        np.eye(3, dtype=np.float32)
      )
      hand_valid.append(0)
      continue
    pose_file = pose_dict[pose_file_idx]
    with hand_pose_zip_ref.open(pose_file) as file:
      # Load the file content using pickle
      hand_info = pickle.load(file)
      hand_kps_3d, wrist_trans, pose_theta, wrist_rot = get_3d_pos_hand(
          hand_info
      )
      hand_pose.append(np.squeeze(hand_kps_3d.detach().cpu().numpy()))
      hand_trans.append(np.squeeze(wrist_trans.detach().cpu().numpy()))
      hand_kps_2D.append(np.squeeze(hand_info["kps2D"]))
      hand_pose_theta.append(
        # np.squeeze(pose_theta.detach().cpu().numpy())
        pose_theta
      )
      hand_rot.append(
        # np.squeeze(wrist_rot.detach().cpu().numpy())
        wrist_rot
      )
      hand_valid.append(1)

  with annotation_zip_ref.open(cam_pose_path) as file:
    data = file.readlines()
    pose_datas = obtain_pose(data)

  min_len = min(
    len(hand_pose),
    len(pose_datas)
  )

  cam_pose = np.stack(pose_datas)[:min_len]
  hand_pose = np.stack(hand_pose)[:min_len]
  hand_trans = np.stack(hand_trans)[:min_len]
  hand_kps_2D = np.stack(hand_kps_2D)[:min_len]
  hand_pose_theta = np.stack(hand_pose_theta)[:min_len]
  hand_rot = np.stack(hand_rot)[:min_len]
  right_hand_valid = np.array(hand_valid)

  cam_pose = expand_first_dim(cam_pose, scale_factor=frame_repeat)
  hand_pose = expand_first_dim(hand_pose, scale_factor=frame_repeat)
  hand_trans = expand_first_dim(hand_trans, scale_factor=frame_repeat)
  hand_kps_2D = expand_first_dim(hand_kps_2D, scale_factor=frame_repeat)
  hand_pose_theta = expand_first_dim(hand_pose_theta, scale_factor=frame_repeat)
  hand_rot = expand_first_dim(hand_rot, scale_factor=frame_repeat)
  right_hand_valid = repeat_array(right_hand_valid, frame_repeat)

  min_len = cam_pose.shape[0]

  seq_data = []

  # right_hand_valid = np.ones(hand_kps_2D.shape[0])

  right_inframe_mask = np.bitwise_and(
      np.bitwise_and(hand_kps_2D[:, 0, 0] < 1920, hand_kps_2D[:, 0, 0] > 0),
      np.bitwise_and(hand_kps_2D[:, 0, 1] < 1080, hand_kps_2D[:, 0, 1] > 0),
      right_hand_valid
  )

  world_frame_hand_pose = transform_to_world_frame(
      cam_pose=cam_pose,
      cam_frame_pos=hand_pose
  )
  world_frame_wrist_3d = transform_to_world_frame(
      cam_pose=cam_pose,
      cam_frame_pos=hand_trans.reshape(-1, 1, 3)
  )

  world_frame_wrist_rot = transform_to_world_frame(
      cam_pose=cam_pose,
      cam_frame_pos=hand_rot.reshape(-1, 3, 3)
  )

  # for frame_count in range(0, len(pose_files), frame_skip * sample_skip):
  for sample_idx in range(0, min_len, sample_skip):
    frame_count = sample_idx * frame_skip
    # sample
    current_cam_pose = cam_pose[sample_idx]
    inv_cam_pose = np.linalg.inv(current_cam_pose)

    # Camera Frame
    # n x 20 x 3
    future_hand_pose = hand_pose[sample_idx:sample_idx + future_len]
    # print(future_hand_pose.shape)
    # print(inv_cam_pose.shape)
    future_hand_pose = transform_to_current_frame(
      inv_cam_pose=inv_cam_pose,
      world_frame_pos=world_frame_hand_pose[sample_idx:sample_idx + future_len]
    )

    # future_right_wrist_3d = hand_trans[sample_idx:sample_idx + future_len]
    future_right_wrist_3d = transform_to_current_frame(
      inv_cam_pose=inv_cam_pose,
      world_frame_pos=world_frame_wrist_3d[sample_idx:sample_idx + future_len]
    )

    # future_right_wrist_rot = hand_trans[sample_idx:sample_idx + future_len]
    future_right_wrist_rot = transform_to_current_frame(
      inv_cam_pose=inv_cam_pose,
      world_frame_pos=world_frame_wrist_rot[sample_idx:sample_idx + future_len]
    )
    
    current_right_wrist_2d = hand_kps_2D[sample_idx, 0]

    future_right_wrist_2d, future_right_wrist_inframe_mask = project_set(
      future_right_wrist_3d, cam_intrinsics,
      HOI4D_WIDTH, HOI4D_HEIGHT
    )

    # current_right_kp_wrist_2d, current_right_kp_wrist_inframe_mask = project_set(
    #   hand_[sample_idx, 0, :], cam_intrinsics,
    #   HOI4D_WIDTH, HOI4D_HEIGHT
    # )
    future_right_kp_wrist_2d, future_right_kp_wrist_inframe_mask = project_set(
      future_hand_pose[:, 0, :], cam_intrinsics,
      HOI4D_WIDTH, HOI4D_HEIGHT
    )

    right_valid = (
      right_inframe_mask[sample_idx]== 1
    ) and (
      np.sum(future_right_kp_wrist_inframe_mask) > (future_right_kp_wrist_inframe_mask.shape[0] * 0.9)
    ) and (
      np.sum(future_right_wrist_inframe_mask) > (future_right_wrist_inframe_mask.shape[0] * 0.9)
    )

    if not right_valid:
      continue
    print(seq_name, frame_count)
    current_data = {
        "seq_name": seq_name,
        "frame_count": frame_count,
        "raw_width": HOI4D_WIDTH,
        "raw_height": HOI4D_HEIGHT,
        # Language label
        "language_label_verb": task_language_label["verb"],
        "language_label_noun": task_language_label["noun"],
        # Current
        "current_right_wrist_3d": hand_trans[sample_idx],
        "current_right_hand_kp": hand_pose[sample_idx],
        "current_right_wrist_2d": current_right_wrist_2d,
        "current_right_flag": right_inframe_mask[sample_idx],
        # As KP
        "current_right_kp_wrist_3d": hand_pose[sample_idx, 0, :],
        "current_right_kp_wrist_2d": current_right_wrist_2d,
        "current_right_kp_flag": right_inframe_mask[sample_idx],

        # Future
        "future_right_wrist_3d": future_right_wrist_3d,
        "future_right_hand_kp": future_hand_pose,
        "future_right_wrist_2d": future_right_wrist_2d,
        "future_right_flag": future_right_wrist_inframe_mask,

        # Future
        "future_right_kp_wrist_3d": future_hand_pose[:, 0, :],
        "future_right_kp_wrist_2d": future_right_kp_wrist_2d,
        "future_right_kp_flag": future_right_kp_wrist_inframe_mask,

        # MANO hand joint rot
        "current_right_pose_theta": hand_pose_theta[sample_idx], 
        "future_right_pose_theta": hand_pose_theta[sample_idx:sample_idx + future_len],
        
        # Rotation
        "current_right_wrist_rot": hand_rot[sample_idx],
        "future_right_wrist_rot": future_right_wrist_rot,

        "current_kp_2d": hand_kps_2D[sample_idx]

    }
    seq_data.append(current_data)
  # for hand_data
  return seq_data
