#
# Section 0: DataProvider initialization
#
# Take home message:
# - Device data, such as Image data stream is indexed with a stream_id
# - Intrinsics and Extrinsics calibration relative to the device coordinates is available for each CAMERA/stream_id
#
# Data Requirements:
# - a sequence
# - the object library
# Optional:
# - To use the Mano hand you need to have the LEFT/RIGHT *.pkl hand models (available)
import human_plan
import os
from Hot3dDataProvider import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel

# home = os.path.expanduser("~")
SEQ_ID="P0003_c701bd11"
hot3d_dataset_path = "/mnt/data3/data/HOT3D/dataset"
sequence_path = os.path.join(hot3d_dataset_path, SEQ_ID)
object_library_path = os.path.join(hot3d_dataset_path, "assets")
# mano_hand_model_path = os.path.join(home, "Downloads")

save_path = os.path.join("playground/hot3d/", SEQ_ID)
from pathlib import Path
Path(save_path).mkdir(parents=True, exist_ok=True)
mano_hand_model_path = "mano_v1_2/models"

if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
    print("Invalid input sequence or library path.")
    print("Please do update the path to VALID values for your system.")
    raise
#
# Init the object library
#
object_library = load_object_library(object_library_folderpath=object_library_path)

#
# Init the HANDs model
# If None, the UmeTrack HANDs model will be used
#
# mano_hand_model = None
if mano_hand_model_path is not None:
    mano_hand_model = MANOHandModel(mano_hand_model_path)

#
# Initialize hot3d data provider
#
hot3d_data_provider = Hot3dDataProvider(
    sequence_folder=sequence_path,
    object_library=object_library,
    mano_hand_model=mano_hand_model,
)
print(f"data_provider statistics: {hot3d_data_provider.get_data_statistics()}")

# Section 1: Device calibration and Image data

from tqdm import tqdm

#
# Retrieve some statistics about the "IMAGE" VRS recording
#

# Getting the device data provider (alias)
device_data_provider = hot3d_data_provider.device_data_provider

# Retrieve the list of image stream supported by this sequence
# It will return the RGB and SLAM Left/Right image streams
image_stream_ids = device_data_provider.get_image_stream_ids()
print(image_stream_ids)
# Retrieve a list of timestamps for the sequence (in nanoseconds)
timestamps = device_data_provider.get_sequence_timestamps()

print(f"Sequence: {os.path.basename(os.path.normpath(sequence_path))}")
print(f"Device type is {hot3d_data_provider.get_device_type()}")
print(f"Image stream ids: {image_stream_ids}")
print(f"Number of timestamp for this sequence: {len(timestamps)}")
print(
    f"Duration of the sequence: {(timestamps[-1] - timestamps[0]) / 1e9} (seconds)"
)  # Timestamps are in nanoseconds


# Init a rerun context to visualize the sequence file images
# rr.init("Device images")
# rec = rr.memory_recording()

import cv2
import numpy as np

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

# Alias over the HEADSET/Device pose data provider
device_pose_provider = hot3d_data_provider.device_pose_data_provider

# Alias over the HAND pose data provider
hand_data_provider = hot3d_data_provider.mano_hand_data_provider if hot3d_data_provider.mano_hand_data_provider is not None else hot3d_data_provider.umetrack_hand_data_provider

from data_loaders.loader_hand_poses import LEFT_HAND_INDEX, RIGHT_HAND_INDEX
import matplotlib.pyplot as plt # Used to display consistent colored Bounding Boxes contours

hand_box2d_data_provider = hot3d_data_provider.hand_box2d_data_provider
hand_uids = [LEFT_HAND_INDEX, RIGHT_HAND_INDEX]
hand_box2d_colors = None
if hand_box2d_data_provider is not None:
    color_map = plt.get_cmap("viridis")
    hand_box2d_colors = color_map(
        np.linspace(0, 1, len(hand_uids))
    )
else:
    print("This section expect to have valid bounding box data")
    

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

for stream_id in image_stream_ids:
  print(type(stream_id))
  
  # Retrieve the camera calibration (intrinsics and extrinsics) for a given stream_id
  [extrinsics, intrinsics] = device_data_provider.get_camera_calibration(stream_id)
  # print(intrinsics.label)
  print(intrinsics)
  print(extrinsics)
  # We will show in next section how to visualize the position of the camera in the world frame
    
# rgb_stream_id = 
rgb_stream_id = device_data_provider._vrs_data_provider.get_stream_id_from_label("camera-rgb")
rgb_extrinsics, rgb_intrinsics = device_data_provider.get_camera_calibration(rgb_stream_id)
# Camera calibration parameters

focal_length = rgb_intrinsics.get_focal_lengths() # fx, fy
principal_point = rgb_intrinsics.get_principal_point()  # cx, cy
distortion_coeffs = np.array(rgb_intrinsics.projection_params()[3:])

# print(principal_point, focal_length[0],  principal_point[0])
camera_intrinsics = np.array([
  [focal_length[0], 0, principal_point[0]],
  [0, focal_length[1], principal_point[1]],
  [0, 0, 1]
])

print(camera_intrinsics)
# camera_intrinsic = 

pose_translations = []

# Accumulate HAND poses translations as list, to show a LINE strip HAND trajectory
left_hand_pose_translations = []
right_hand_pose_translations = []

# How to iterate over timestamps using a slice to show one timestamp every 200
timestamps_slice = slice(None, None, 200)
# Loop over the timestamps of the sequence and visualize corresponding data
for timestamp_ns in tqdm(timestamps[timestamps_slice]):
  headset_pose3d_with_dt = None
  if device_pose_provider is None:
      continue
  headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
      timestamp_ns=timestamp_ns,
      time_query_options=TimeQueryOptions.CLOSEST,
      time_domain=TimeDomain.TIME_CODE,
  )

  if headset_pose3d_with_dt is None:
      continue

  headset_pose3d = headset_pose3d_with_dt.pose3d
  T_world_device = headset_pose3d.T_world_device
  T_world_camera = T_world_device @ rgb_extrinsics

  pose_translations.append(T_world_device.translation()[0])

  # HAND
  hand_poses_with_dt = None
  if hand_data_provider is None:
      continue

  hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
    timestamp_ns=timestamp_ns,
    time_query_options=TimeQueryOptions.CLOSEST,
    time_domain=TimeDomain.TIME_CODE,
  )

  if hand_poses_with_dt is None:
    continue

  hand_pose_collection = hand_poses_with_dt.pose3d_collection

  for hand_pose_data in hand_pose_collection.poses.values():
    # Retrieve the handedness of the hand (i.e Left or Right)
    handedness_label = hand_pose_data.handedness_label()
    T_world_wrist = hand_pose_data.wrist_pose
    joint_angles = hand_pose_data.joint_angles
    hand_landmarks = hand_data_provider.get_hand_landmarks(
      hand_pose_data
    )

    T_camera_wrist = T_world_camera.inverse() @ T_world_wrist
    print(T_camera_wrist.translation()[0])

    T_camera_wrist_raw = T_camera_wrist
    T_camera_wrist = T_camera_wrist.translation()[0]
    # T_camera_wrist = T_camera_wrist[:3, -1]
    # print(T_camera_wrist)
    # print(T_camera_wrist[:3, -1])
    import torch
    hand_landmarks = np.einsum(
      "ij, kj -> ki",
      T_world_camera.inverse().to_matrix(),
      np.concatenate([
      hand_landmarks, np.ones((hand_landmarks.shape[0], 1))
      ], axis=-1),
    )
    hand_landmarks = hand_landmarks[:, :3]
    # print(hand_landmarks)
    # print(hand_landmarks.shape)

    # Accumulate HAND poses translations as list, to show a LINE strip HAND trajectory
    if hand_pose_data.is_left_hand():
        # print(T_world_wrist.translation().shape)
        # left_hand_pose_translations.append(T_camera_wrist.translation()[0])
        # left_hand_pose_translations.append(T_camera_wrist[:3, -1])
        # left_hand_pose_current = T_camera_wrist[:3, -1]
        left_hand_pose_translations.append(T_camera_wrist)
        left_hand_pose_current = T_camera_wrist
        left_hand_pose_world = T_world_wrist
        left_hand_pose_current_raw = T_camera_wrist_raw
        left_hand_landmarks = hand_landmarks
    elif hand_pose_data.is_right_hand():
        # print(T_world_wrist.translation().shape)
        # right_hand_pose_translations.append(T_camera_wrist.translation()[0])
        # right_hand_pose_translations.append(T_camera_wrist[:3, -1])
        # right_hand_pose_current = T_camera_wrist[:3, -1]
        right_hand_pose_translations.append(T_camera_wrist)
        right_hand_pose_current = T_camera_wrist
        right_hand_pose_world = T_world_wrist
        right_hand_pose_current_raw = T_camera_wrist_raw
        right_hand_landmarks = hand_landmarks


  if T_camera_wrist_raw is None:
    continue
  # for stream_id in image_stream_ids:
  # rgb_stream_id 
  # print(rgb_stream_id)
    # Retrieve the image stream label as string
  image_stream_label = device_data_provider.get_image_stream_label(rgb_stream_id)
  # Retrieve the image data for a given timestamp
  image_data = device_data_provider.get_undistorted_image(timestamp_ns, rgb_stream_id)
  # image_data = device_data_provider.get_image(timestamp_ns, rgb_stream_id)
  # Visualize the image data (it's a numpy array)
  # log_image(label=f"img/{image_stream_label}", image=image_data)
  if len(image_data.shape) == 3:
      image_data = image_data[:, :, ::-1]

  box2d_collection_with_dt = (
      hand_box2d_data_provider.get_bbox_at_timestamp(
          stream_id=rgb_stream_id,
          timestamp_ns=timestamp_ns,
          time_query_options=TimeQueryOptions.CLOSEST,
          time_domain=TimeDomain.TIME_CODE,
      )
  )
  if box2d_collection_with_dt is None:
      continue
  if (
      box2d_collection_with_dt is None
      and box2d_collection_with_dt.box2d_collection or None
  ):
      continue

  # We have valid data, returned as a collection
  # i.e for each hand_uid, we retrieve its BBOX and visibility
  for hand_uid in hand_uids:
      hand_name = "left" if hand_uid == LEFT_HAND_INDEX else "right"
      axis_aligned_box2d = box2d_collection_with_dt.box2d_collection.box2ds[hand_uid]
      bbox = axis_aligned_box2d.box2d
      visibility_ratio = axis_aligned_box2d.visibility_ratio
      if bbox is None:
          continue

      # print([bbox.left, bbox.top], [bbox.width, bbox.height])
      # print((int(bbox.left), int(bbox.top)), (int(bbox.left + bbox.width), int(bbox.top + bbox.height)))
      image_data = cv2.rectangle(np.array(image_data), 
        (int(bbox.left), int(bbox.top)), (int(bbox.left + bbox.width), int(bbox.top + bbox.height)), color=(0,0,255), thickness=20)

  # from projectaria_tools.core.calibration.
  print(left_hand_pose_current_raw.translation()[0], right_hand_pose_current_raw.translation()[0])
  left_raw_proj_points = rgb_intrinsics.project(left_hand_pose_current_raw.translation()[0])
  right_raw_proj_points = rgb_intrinsics.project(right_hand_pose_current_raw.translation()[0])
  print("WTF")
  print(left_hand_pose_current)
  print(right_hand_pose_current)
  # left_raw_proj_points = rgb_intrinsics.project(np.array(left_hand_pose_current))
  # right_raw_proj_points = rgb_intrinsics.project(np.array(right_hand_pose_current))
  print("Raw Projection:", left_raw_proj_points)
  # Put an empty camera pose for image.
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])

  left_points, _ = cv2.projectPoints(
      # hand_points[:3], rvec, tvec, img_intrinsics, np.array([]))
      left_hand_pose_current, rvec, tvec, camera_intrinsics, np.array([])
  )
  left_points = np.squeeze(np.array(left_points)).reshape(-1, 2)
  left_points = rotate_clockwise_batch(left_points, 1408, 1408)
  right_points, _ = cv2.projectPoints(
      # hand_points[:3], rvec, tvec, img_intrinsics, np.array([]))
      right_hand_pose_current, rvec, tvec, camera_intrinsics, np.array([])
  )
  right_points = np.squeeze(np.array(right_points)).reshape(-1, 2)
  right_points = rotate_clockwise_batch(right_points, 1408, 1408)
  print("Left Pose:", left_hand_pose_current, "World", left_hand_pose_world.translation()[0])
  print("Right Pose:", right_hand_pose_current, "World", right_hand_pose_world.translation()[0])
  

  image_data = cv2.rotate(np.array(image_data), cv2.ROTATE_90_CLOCKWISE)
  image_data = cv2.circle(
      np.array(image_data), 
      (int(round(left_points[0, 0])), int(round(left_points[0, 1]))),
      radius=10, color=(0, 0, 255), thickness=20
  )
  image_data = cv2.circle(
      np.array(image_data), 
      (int(round(right_points[0, 0])), int(round(right_points[0, 1]))),
      radius=10, color=(0, 255, 0), thickness=20
  )
  for hp in left_hand_landmarks:
    hpoints, _ = cv2.projectPoints(
        hp, rvec, tvec, camera_intrinsics, np.array([])
    )
    
    hpoints = np.squeeze(np.array(hpoints)).reshape(-1, 2)
    hpoints = rotate_clockwise_batch(hpoints, 1408, 1408)
    image_data = cv2.circle(
        np.array(image_data), 
        (int(round(hpoints[0, 0])), int(round(hpoints[0, 1]))),
        radius=5, color=(0, 0, 128), thickness=10
    )
  for hp in right_hand_landmarks:
    hpoints, _ = cv2.projectPoints(
        hp, rvec, tvec, camera_intrinsics, np.array([])
    )
    hpoints = np.squeeze(np.array(hpoints)).reshape(-1, 2)
    hpoints = rotate_clockwise_batch(hpoints, 1408, 1408)
    image_data = cv2.circle(
        np.array(image_data), 
        (int(round(hpoints[0, 0])), int(round(hpoints[0, 1]))),
        radius=5, color=(0, 0, 128), thickness=10
    )
  # if left_raw_proj_points is not None:
  #   image_data = cv2.circle(
  #       np.array(image_data), 
  #       (int(round(left_raw_proj_points[0])), int(round(left_raw_proj_points[1]))),
  #       radius=10, color=(255, 0, 255), thickness=20
  #   )
  # if right_raw_proj_points is not None:
  #   image_data = cv2.circle(
  #       np.array(image_data), 
  #       (int(round(right_raw_proj_points[0])), int(round(right_raw_proj_points[1]))),
  #       radius=10, color=(255, 255, 0), thickness=20
  #   )
  # print(np.array(image_data))
  
  if image_data is not None:
    cv2.imwrite(os.path.join(save_path, f"{timestamp_ns}_{rgb_stream_id}.jpg"), np.array(image_data))
  # print(image_data)

head_pose_trans = np.array(
   pose_translations
)
print(head_pose_trans.shape)

left_pose_trans = np.array(
   left_hand_pose_translations
)
print(left_pose_trans.shape)


right_pose_trans = np.array(
   right_hand_pose_translations
)
print(right_pose_trans.shape)

#
# Retrieve Camera calibration (intrinsics and extrinsics) for a given stream_id
#

# Showing the rerun window
# rr.notebook_show()


