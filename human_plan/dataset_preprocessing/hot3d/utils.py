import os
import human_plan
import os
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from tqdm import tqdm
import cv2
import numpy as np
from projectaria_tools.core.stream_id import StreamId  # @manual

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.calibration import (  # @manual
    CameraCalibration,
    DeviceCalibration,
    distort_by_calibration,
    FISHEYE624,
    get_linear_camera_calibration,
    LINEAR,
)

from human_plan.dataset_preprocessing.utils.transformations import (
  transform_to_current_frame,
  transform_to_current_frame_pose,
  matrix_to_trans_rot
)

def get_all_seqs(dataset_root):
  all_seq_names = os.listdir(dataset_root)
  all_seq_names = [seq for seq in all_seq_names if seq != "assets"]
  return all_seq_names


def get_aria_seqs(dataset_root):  
  all_seqs = get_all_seqs(dataset_root)
  aria_data_list_file = "aria_seqs.txt"
  aria_seqs_path = os.path.join(
    os.path.dirname(__file__), aria_data_list_file
  )
  with open(aria_seqs_path, 'r') as f:
    aria_seqs = f.readlines()
  aria_seqs = [seq.strip() for seq in aria_seqs]
  aria_seqs = sorted([seq for seq in all_seqs if seq in aria_seqs])
  return aria_seqs


def get_quest_seqs(dataset_root):  
  all_seqs = get_all_seqs(dataset_root)
  quest_data_list_file = "quest_seqs.txt"
  quest_seqs_path = os.path.join(
    os.path.dirname(__file__), quest_data_list_file
  )
  with open(quest_seqs_path, 'r') as f:
    quest_seqs = f.readlines()
  quest_seqs = [seq.strip() for seq in quest_seqs]
  quest_seqs = sorted([seq for seq in all_seqs if seq in quest_seqs])
  return quest_seqs


def get_hot3d_data_provider(dataset_root, seq_name):
  # home = os.path.expanduser("~")
  sequence_path = os.path.join(dataset_root, seq_name)
  object_library_path = os.path.join(dataset_root, "assets")
  # mano_hand_model_path = os.path.join(home, "Downloads")

  mano_hand_model_path = "mano_v1_2/models"

  if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
      print("Invalid input sequence or library path.")
      print("Please do update the path to VALID values for your system.")
      return None

  # Init the object library
  object_library = load_object_library(object_library_folderpath=object_library_path)
  # Init the HANDs model
  # If None, the UmeTrack HANDs model will be used
  if mano_hand_model_path is not None:
      mano_hand_model = MANOHandModel(mano_hand_model_path)

  # Initialize hot3d data provider
  hot3d_data_provider = Hot3dDataProvider(
      sequence_folder=sequence_path,
      object_library=object_library,
      mano_hand_model=mano_hand_model,
  )
  # print(f"data_provider statistics: {hot3d_data_provider.get_data_statistics()}")

  return hot3d_data_provider


# Merge Hand / Image / Language Label
def parse_single_seq_image_aria(
    hot3d_data_provider,
    seq_name,
    # annotations: Dict,
    frame_skip: int,
    image_w: int,
    image_h: int,
    # video_fps: int=30
):
  video_fps = 30

  # Getting the device data provider (alias)
  device_data_provider = hot3d_data_provider.device_data_provider

  # Retrieve the list of image stream supported by this sequence
  # It will return the RGB and SLAM Left/Right image streams
  image_stream_ids = device_data_provider.get_image_stream_ids()
  # print(image_stream_ids)
  # rgb_stream_id = [stream_id for stream_id in image_stream_ids if "214" in stream_id]
  rgb_stream_id = device_data_provider._vrs_data_provider.get_stream_id_from_label("camera-rgb")

  _, native_camera_calibration = device_data_provider.get_camera_calibration(
    rgb_stream_id, camera_model=FISHEYE624
  )
  _, pinhole_camera_calibration = device_data_provider.get_camera_calibration(
    rgb_stream_id, camera_model=LINEAR
  )
  # assert len(rgb_stream_id == 1)
  # rgb_stream_id = rgb_stream_id[0]

  # Retrieve a list of timestamps for the sequence (in nanoseconds)
  timestamps = device_data_provider.get_sequence_timestamps()

  # How to iterate over timestamps using a slice to show one timestamp every 200
  timestamps_slice = slice(None, None, frame_skip)
  # Loop over the timestamps of the sequence and visualize corresponding data
  seq_data = []

  for idx, timestamp_ns in tqdm(enumerate(timestamps[timestamps_slice])):
    frame_count = idx * frame_skip

    image_data = device_data_provider.get_image(timestamp_ns, rgb_stream_id)
    image_data = distort_by_calibration(
      image_data, pinhole_camera_calibration, native_camera_calibration
    )
    # RGB order
    image_data = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)
    image_data = cv2.resize(image_data, (image_w, image_h))
    # print(image_data)
    current_data = {
      "seq_name": seq_name,
      "frame_count": frame_count,
      "rgb_obs": image_data,
    }
    seq_data.append(current_data)
  return seq_data


# Merge Hand / Image / Language Label
def parse_single_seq_image_quest3(
    hot3d_data_provider,
    seq_name,
    # annotations: Dict,
    frame_skip: int,
    image_w: int,
    image_h: int,
    # video_fps: int=30
):
  # Getting the device data provider (alias)
  device_data_provider = hot3d_data_provider.device_data_provider
  # Retrieve the list of image stream supported by this sequence
  # It will return the RGB and SLAM Left/Right image streams
  left_slam_stream_id = StreamId("1201-1")

  _, native_camera_calibration = device_data_provider.get_camera_calibration(
    left_slam_stream_id, camera_model=FISHEYE624
  )
  _, pinhole_camera_calibration = device_data_provider.get_camera_calibration(
    left_slam_stream_id, camera_model=LINEAR
  )

  # Retrieve a list of timestamps for the sequence (in nanoseconds)
  timestamps = device_data_provider.get_sequence_timestamps()

  # How to iterate over timestamps using a slice to show one timestamp every 200
  timestamps_slice = slice(None, None, frame_skip)
  # Loop over the timestamps of the sequence and visualize corresponding data
  seq_data = []

  for idx, timestamp_ns in tqdm(enumerate(timestamps[timestamps_slice])):
    frame_count = idx * frame_skip

    image_data = device_data_provider.get_image(timestamp_ns, left_slam_stream_id)
    if image_data is None:
      continue
    image_data = distort_by_calibration(
      image_data, pinhole_camera_calibration, native_camera_calibration
    )
    image_data = np.repeat(
      image_data[:, :, np.newaxis], 3, axis=-1
    )
    # RGB order
    image_data = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)
    image_data = cv2.resize(image_data, (image_w, image_h))
    # print(image_data)
    current_data = {
      "seq_name": seq_name,
      "frame_count": frame_count,
      "rgb_obs": image_data,
    }
    seq_data.append(current_data)
  return seq_data


def rotate_clockwise_batch(points, width, height):
  """
  Rotate a batch of points by 90 degrees clockwise using NumPy arrays.

  :param points: NumPy array of shape (N, 2), where each row is (x, y) coordinates of a point.
  :param width: The width of the image.
  :param height: The height of the image.
  :return: NumPy array of shape (N, 2) representing the new (x', y') coordinates after rotation.
  """
  # Extract x and y columns
  x = points[:, 0]
  y = points[:, 1]

  # Apply the 90-degree clockwise rotation formula
  new_x = height - y - 1
  new_y = x

  # Combine the results into a new array
  rotated_points = np.column_stack((new_x, new_y))

  return rotated_points

def project_set(points, camera_intrinsics, width, height):
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])
  # print(points.shape, points.dtype)
  points, _ = cv2.projectPoints(
    # points,
    points.astype(dtype=np.float64),
    rvec, tvec, camera_intrinsics, np.array([])
  )
  points = np.array(points).reshape(-1, 2)
  # points = rotate_clockwise_batch(points, width, height)
  # !!! Rotate would cause the width and height swap
  points = rotate_clockwise_batch(points, height, width)

  mask_x = np.bitwise_and(
    points[..., 0] < width, points[..., 0] >= 0
  )
  mask_y = np.bitwise_and(
    points[..., 1] < height, points[..., 1] >= 0
  )
  mask = np.bitwise_and(
    mask_x,
    mask_y
  )
  return points, mask


ARIA_WIDTH = 1408
ARIA_HEIGHT= 1408

QUEST_WIDTH = 1024
QUEST_HEIGHT = 1280

# Merge Hand / Image / Language Label
def parse_single_seq_hand(
    hot3d_data_provider,
    seq_name,
    is_aria: bool,
    # annotations: Dict,
    frame_skip: int,
    sample_skip: int,
    future_len: int,
    his_len: int,
    image_mapping_dict = None
):
  video_fps = 30
  # Getting the device data provider (alias)
  device_data_provider = hot3d_data_provider.device_data_provider

  if is_aria:
    c_w, c_h = ARIA_WIDTH, ARIA_HEIGHT
  else:
    c_w, c_h = QUEST_WIDTH, QUEST_HEIGHT

  if is_aria:
    rgb_stream_id = device_data_provider._vrs_data_provider.get_stream_id_from_label("camera-rgb")
  else:
    rgb_stream_id = StreamId("1201-1")
  T_device_camera, rgb_intrinsics = device_data_provider.get_camera_calibration(rgb_stream_id)

  focal_length = rgb_intrinsics.get_focal_lengths() # fx, fy
  principal_point = rgb_intrinsics.get_principal_point()  # cx, cy
  camera_intrinsics = np.array([
    [focal_length[0], 0, principal_point[0]],
    [0, focal_length[1], principal_point[1]],
    [0, 0, 1]
  ])

  # Alias over the HAND pose data provider
  hand_data_provider = hot3d_data_provider.mano_hand_data_provider if hot3d_data_provider.mano_hand_data_provider is not None else hot3d_data_provider.umetrack_hand_data_provider
  # Retrieve a list of timestamps for the sequence (in nanoseconds)
  timestamps = device_data_provider.get_sequence_timestamps()

  # Alias over the HEADSET/Device pose data provider
  device_pose_provider = hot3d_data_provider.device_pose_data_provider
  if device_pose_provider is None:
    return []
  
  if hand_data_provider is None:
    return []

  camera_pose = []
  left_wrist_pose_world = []
  left_wrist_rot_world = []
  right_wrist_pose_world = []
  right_wrist_rot_world = []
  left_hand_pose_world = []
  right_hand_pose_world = []
  
  left_hand_joint_angles = []
  right_hand_joint_angles = []
  

  camera_valid = []
  left_hand_valid = []
  right_hand_valid = []

  # How to iterate over timestamps using a slice to show one timestamp every 200
  timestamps_slice = slice(None, None, frame_skip)
  # Loop over the timestamps of the sequence and visualize corresponding data

  for idx, timestamp_ns in tqdm(enumerate(timestamps[timestamps_slice]), total=(len(timestamps) // frame_skip)):
    headset_pose3d_with_dt = None
    headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp_ns,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )

    if headset_pose3d_with_dt is None:
      camera_pose.append(np.zeros((4,4)))
      camera_valid.append(0)

    headset_pose3d = headset_pose3d_with_dt.pose3d
    T_world_camera = headset_pose3d.T_world_device @ T_device_camera

    camera_pose.append(T_world_camera.to_matrix())
    camera_valid.append(1)

    # HAND
    hand_poses_with_dt = None
    hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
      timestamp_ns=timestamp_ns,
      time_query_options=TimeQueryOptions.CLOSEST,
      time_domain=TimeDomain.TIME_CODE,
    )

    if hand_poses_with_dt is None:
      left_hand_valid.append(0)
      right_hand_valid.append(0)
      left_wrist_pose_world.append(np.eye(4))
      right_wrist_pose_world.append(np.eye((4)))
      left_hand_pose_world.append(np.zeros((20, 3)))
      right_hand_pose_world.append(np.zeros((20, 3)))

      left_hand_joint_angles.append(np.zeros((15, 3, 3)))
      right_hand_joint_angles.append(np.zeros((15, 3, 3)))
      continue

    hand_pose_collection = hand_poses_with_dt.pose3d_collection

    found_left = False
    found_right = False
    for hand_pose_data in hand_pose_collection.poses.values():
      # Retrieve the handedness of the hand (i.e Left or Right)
      # handedness_label = hand_pose_data.handedness_label()
      T_world_wrist = hand_pose_data.wrist_pose
      hand_landmarks = hand_data_provider.get_hand_landmarks(
        hand_pose_data
      )
      # Accumulate HAND poses translations as list, to show a LINE strip HAND trajectory
      if hand_pose_data.is_left_hand():
        # left_wrist_pose_world.append(T_world_wrist.translation())
        left_wrist_pose_world.append(T_world_wrist.to_matrix())
        # left_wrist_rot_world.append(T_world_wrist.rotation().to_matrix())
        left_hand_pose_world.append(hand_landmarks.cpu().numpy())
        left_hand_valid.append(1)

        # print(hand_pose_data.joint_angles)
        # print(np.array(hand_pose_data.joint_angles).shape)
        # exit()

        # HOT3D use 15 dim PCA
        left_hand_joint_angles.append(np.array(hand_pose_data.joint_angles).reshape(15))
        found_left = True
      elif hand_pose_data.is_right_hand():
        # right_wrist_pose_world.append(T_world_wrist.translation())
        right_wrist_pose_world.append(T_world_wrist.to_matrix())
        # right_wrist_rot_world.append(T_world_wrist.rotation().to_matrix())
        right_hand_pose_world.append(hand_landmarks.cpu().numpy())
        right_hand_valid.append(1)
        # HOT3D use 15 dim PCA
        right_hand_joint_angles.append(np.array(hand_pose_data.joint_angles).reshape(15))
        found_right = True

    if not found_left:
        left_wrist_pose_world.append(np.zeros((4, 4)))
        # left_wrist_rot_world.append(np.eye(3))
        left_hand_pose_world.append(np.zeros((20, 3)))
        left_hand_valid.append(0)
        left_hand_joint_angles.append(np.zeros((15)))

    if not found_right:
        right_wrist_pose_world.append(np.zeros((4, 4)))
        # right_wrist_rot_world.append(np.eye(3))
        right_hand_pose_world.append(np.zeros((20, 3)))
        right_hand_valid.append(0)
        right_hand_joint_angles.append(np.zeros((15)))

  camera_pose = np.stack(camera_pose)
  camera_valid = np.stack(camera_valid)

  left_wrist_pose_world = np.stack(left_wrist_pose_world)
  left_hand_pose_world = np.stack(left_hand_pose_world)
  left_hand_valid = np.stack(left_hand_valid)
  left_hand_joint_angles = np.stack(left_hand_joint_angles)

  right_wrist_pose_world = np.stack(right_wrist_pose_world)
  right_hand_pose_world = np.stack(right_hand_pose_world)
  right_hand_valid = np.stack(right_hand_valid)
  right_hand_joint_angles = np.stack(right_hand_joint_angles)

  seq_data = []
  timestamps_slice = slice(None, None, frame_skip * sample_skip)
  for idx, timestamp_ns in tqdm(
    enumerate(timestamps[timestamps_slice]), 
    total=(len(timestamps) // frame_skip // sample_skip)
  ):
    frame_count = idx * frame_skip * sample_skip
    if frame_count not in image_mapping_dict[seq_name]:
      continue

    sample_idx = idx * sample_skip
    current_camera_pose = camera_pose[sample_idx]

    inv_cam_pose = np.linalg.inv(current_camera_pose)

    # print(inv_cam_pose.shape)
    # print(left_wrist_pose_world[sample_idx].shape)
    current_left_wrist_pose = transform_to_current_frame_pose(
      inv_cam_pose,
      left_wrist_pose_world[sample_idx].reshape(1, 1, 4, 4)
    )
    current_left_wrist_3d, current_left_wrist_rot = matrix_to_trans_rot(
      current_left_wrist_pose
    )
    current_left_hand_kp = transform_to_current_frame(
      inv_cam_pose,
      left_hand_pose_world[sample_idx]
    )

    current_right_wrist_pose = transform_to_current_frame_pose(
      inv_cam_pose,
      right_wrist_pose_world[sample_idx].reshape(1, 1, 4, 4)
    )
    current_right_wrist_3d, current_right_wrist_rot = matrix_to_trans_rot(
      current_right_wrist_pose
    )
    current_right_hand_kp = transform_to_current_frame(
      inv_cam_pose,
      right_hand_pose_world[sample_idx]
    )

    # print(inv_cam_pose.shape)
    # print(left_wrist_pose_world[sample_idx:sample_idx+future_len].shape)
    future_left_wrist_pose = transform_to_current_frame_pose(
      inv_cam_pose,
      left_wrist_pose_world[sample_idx:sample_idx+future_len].reshape(-1, 1, 4, 4)
    )
    future_left_wrist_3d, future_left_wrist_rot = matrix_to_trans_rot(
      future_left_wrist_pose
    )
    future_left_hand_kp = transform_to_current_frame(
      inv_cam_pose,
      left_hand_pose_world[sample_idx:sample_idx+future_len]
    )

    future_right_wrist_pose = transform_to_current_frame_pose(
      inv_cam_pose,
      right_wrist_pose_world[sample_idx:sample_idx+future_len].reshape(-1, 1, 4, 4)
    )
    future_right_wrist_3d, future_right_wrist_rot = matrix_to_trans_rot(
      future_right_wrist_pose
    )

    future_right_hand_kp = transform_to_current_frame(
      inv_cam_pose,
      right_hand_pose_world[sample_idx:sample_idx+future_len]
    )

    current_left_wrist_2d, current_left_inframe_mask = project_set(
      current_left_wrist_3d, camera_intrinsics,
      c_w, c_h
    )
    current_right_wrist_2d, current_right_inframe_mask = project_set(
      current_right_wrist_3d, camera_intrinsics,
      c_w, c_h
    )
    future_left_wrist_2d, future_left_inframe_mask = project_set(
      future_left_wrist_3d, camera_intrinsics,
      c_w, c_h
    )
    future_right_wrist_2d, future_right_inframe_mask = project_set(
      future_right_wrist_3d, camera_intrinsics,
      c_w, c_h
    )

    current_mano_left_wrist_2d, current_mano_left_inframe_mask = project_set(
      current_left_hand_kp[:, 5, :], camera_intrinsics,
      c_w, c_h
    )
    current_mano_right_wrist_2d, current_mano_right_inframe_mask = project_set(
      current_right_hand_kp[:, 5, :], camera_intrinsics,
      c_w, c_h
    )

    future_mano_left_wrist_2d, future_mano_left_inframe_mask = project_set(
      future_left_hand_kp[:, 5, :], camera_intrinsics,
      c_w, c_h
    )
    future_mano_right_wrist_2d, future_mano_right_inframe_mask = project_set(
      future_right_hand_kp[:, 5, :], camera_intrinsics,
      c_w, c_h
    )

    if future_left_inframe_mask.shape[0] < future_len:
      break 
    # his_end=max(0, sample_idx-future_len)
    # his_left_wrist_3d = np.einsum(
    #   "ij, nkj-> nki",
    #   inv_cam_pose, left_wrist_pose_world[sample_idx:his_end:-1]
    # )
    # his_right_wrist_3d = np.einsum(
    #   "ij, nkj-> nki",
    #   inv_cam_pose, right_wrist_pose_world[sample_idx:his_end:-1]
    # )
    
    left_valid = (
      current_left_inframe_mask == 1
    ) and (
      current_mano_left_inframe_mask == 1
    ) and (
      np.sum(left_hand_valid[sample_idx:sample_idx+future_len]) > (left_hand_valid[sample_idx:sample_idx+future_len].shape[0] * 0.9)
    ) and (
      np.sum(future_left_inframe_mask) > (future_left_inframe_mask.shape[0] * 0.9)
    ) and (
      np.sum(future_mano_left_inframe_mask) > (future_mano_left_inframe_mask.shape[0] * 0.9)
    )

    right_valid = (
      current_right_inframe_mask == 1
    ) and (
      current_mano_right_inframe_mask == 1
    ) and (
      np.sum(right_hand_valid[sample_idx:sample_idx+future_len]) > (right_hand_valid[sample_idx:sample_idx+future_len].shape[0] * 0.9)
    ) and (
      np.sum(future_right_inframe_mask) > (future_right_inframe_mask.shape[0] * 0.9)
    ) and (
      np.sum(future_mano_right_inframe_mask) > (future_mano_right_inframe_mask.shape[0] * 0.9)
    )

    if not left_valid and not right_valid:
      continue

    current_data = {
      "seq_name": seq_name,
      "frame_count": frame_count,
      "raw_height": c_h,
      "raw_width": c_w,
      "camera_intrinsics": camera_intrinsics,

      # Current
      "current_left_wrist_3d": current_left_wrist_3d,
      "current_right_wrist_3d": current_right_wrist_3d,
      "current_left_hand_kp": current_left_hand_kp,
      "current_right_hand_kp": current_right_hand_kp,
      "current_left_wrist_2d": current_left_wrist_2d,
      "current_right_wrist_2d": current_right_wrist_2d,
      "current_left_flag": left_hand_valid[sample_idx],
      "current_right_flag": right_hand_valid[sample_idx],

      # Future
      "future_left_wrist_3d": future_left_wrist_3d,
      "future_right_wrist_3d": future_right_wrist_3d,
      "future_left_hand_kp": future_left_hand_kp,
      "future_right_hand_kp": future_right_hand_kp,
      "future_left_wrist_2d": future_left_wrist_2d,
      "future_right_wrist_2d": future_right_wrist_2d,
      "future_left_flag": left_hand_valid[sample_idx:sample_idx+future_len],
      "future_right_flag": right_hand_valid[sample_idx:sample_idx+future_len],

      #
      "current_left_in_frame_flag:": current_left_inframe_mask,
      "current_right_in_frame_flag:": current_right_inframe_mask,
      "future_left_in_frame_flag": future_left_inframe_mask,
      "future_right_in_frame_flag": future_right_inframe_mask,

      # use wrist in mano hand as key point
      "current_mano_left_wrist_3d": current_left_hand_kp[..., 5, :],
      "current_mano_left_wrist_2d": current_mano_left_wrist_2d,
      "current_mano_left_inframe_mask" : current_mano_left_inframe_mask,
      "current_mano_right_wrist_3d": current_right_hand_kp[..., 5, :],
      "current_mano_right_wrist_2d": current_mano_right_wrist_2d, 
      "current_mano_right_inframe_mask": current_mano_right_inframe_mask,
      # 
      "future_mano_left_wrist_3d": future_left_hand_kp[..., 5, :],
      "future_mano_left_wrist_2d": future_mano_left_wrist_2d,
      "future_mano_left_inframe_mask": future_mano_left_inframe_mask,
      "future_mano_right_wrist_3d": future_right_hand_kp[..., 5, :],
      "future_mano_right_wrist_2d": future_mano_right_wrist_2d,
      "future_mano_right_inframe_mask": future_mano_right_inframe_mask,

      # Hand rotation # Mano Formulation
      "current_left_wrist_rot": current_left_wrist_rot,
      "current_right_wrist_rot": current_right_wrist_rot,
      "future_left_wrist_rot": future_left_wrist_rot,
      "future_right_wrist_rot": future_right_wrist_rot,

      # Joint Angle -> PCA -> 15 Dim
      "current_left_hand_joint_pca": left_hand_joint_angles[sample_idx],
      "current_right_hand_joint_pca": right_hand_joint_angles[sample_idx],
      "future_left_hand_joint_pca": left_hand_joint_angles[sample_idx:sample_idx+future_len],
      "future_right_hand_joint_pca": right_hand_joint_angles[sample_idx:sample_idx+future_len],
      # "future_left_wrist_rot": future_left_wrist_rot,
      # "future_right_wrist_rot": future_right_wrist_rot,

    }
    seq_data.append(current_data)
  # for hand_data 
  return seq_data
