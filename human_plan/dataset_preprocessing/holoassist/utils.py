from pathlib import Path
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import copy
import pickle
import math
from manopth.manolayer import ManoLayer

LEFT_HAND_SUFFIX = "Hands/Left_sync.txt"
RIGHT_HAND_SUFFIX = "Hands/Right_sync.txt"

HEAD_SUFFIX = "Head/Head_sync.txt"

IMU_SUFFIX = "IMU/Accelerometer_sync.txt"

POSE_SUFFIX = "Video/Pose_sync.txt"

INSTRICS_SUFFIX = "Video/Intrinsics.txt"

AXIS_TRANSFORM = np.linalg.inv(np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
]))

from human_plan.dataset_preprocessing.utils.language_response import sample_text_response

hand_pose_connectivity = np.array([
  # Thumb
  [2, 3], [3, 4], [4, 5],
  # Index
  [7, 8], [8, 9], [9, 10],
  # Middle
  [12, 13], [13, 14], [14, 15],
  # Ring
  [17, 18], [18, 19], [19, 20],
  # Pinky
  [22, 23], [23, 24], [24, 25]
]).reshape(15, 2)


def convert_kp_pose_to_mano_rotation(hand_kp_pose):
  children_pose = hand_kp_pose[:, hand_pose_connectivity[:, 1], :, :]
  parent_pose = hand_kp_pose[:, hand_pose_connectivity[:, 0], :, :]
  relative_pose = np.linalg.inv(parent_pose) @ children_pose
  return relative_pose[:, :, :3, :3]


holoassist_to_mano_joint_mapping = np.array([
  1, # wrist 
  2, 3, 4, 5, # Thumb
  7, 8, 9, 10, # Index
  12, 13, 14, 15, # middle
  17, 18, 19, 20, # Ring
  22, 23, 24, 25 # Index
])

import numpy as np


def parse_pose_data(img_pose_path):
  img_pose_array = np.loadtxt(img_pose_path)[:, 2:].reshape(-1, 4, 4)
  return img_pose_array


def parse_hand_data(hand_array, cam_pose):
  # 471 DIM
  # First Dim -> time in second
  # Second Dim -> originating time tag
  start = 2
  active_flag = hand_array[:, start]

  start += 1
  # 26 joint, 16 for 4x4 pose per joint
  hand_pose = hand_array[:, start:start + 26 * 16]
  hand_pose = hand_pose.reshape(-1, 26, 4, 4)
  # Only The Translation
  hand_translation = hand_pose[:, :, :, -1:]
  start += 26 * 16

  joint_valid = hand_array[:, start: start + 26]
  start += 26

  track_state = hand_array[:, start: start + 26]

  hand_translation_cam_frame = AXIS_TRANSFORM[np.newaxis, np.newaxis, :, :] @ \
    np.linalg.inv(cam_pose)[:, np.newaxis, :, :] @ \
    hand_translation

  # AXIS_TRANSFORM -> 1, 1, 4, 4
  # np.linalg.inv(cam_pose)[:, np.newaxis, :, :] -> n, 1, 4, 4
  # hand pose -> n, 26, 4, 4
  hand_pose_cam_frame = AXIS_TRANSFORM[np.newaxis, np.newaxis, :, :] @ \
    np.linalg.inv(cam_pose)[:, np.newaxis, :, :] @ \
    hand_pose
  # print(hand_translation.shape)
  # print(hand_translation_cam_frame.shape, hand_pose_cam_frame[..., -1:].shape)
  assert np.allclose(
    hand_translation_cam_frame, hand_pose_cam_frame[..., -1:],
    equal_nan=True
  ), f"{hand_translation_cam_frame - hand_pose_cam_frame[..., -1:]} {np.max(hand_translation_cam_frame - hand_pose_cam_frame[..., -1:])} {np.min(hand_translation_cam_frame - hand_pose_cam_frame[..., -1:])} {np.isfinite(hand_translation_cam_frame), np.isfinite(hand_pose_cam_frame[..., -1:])}"

  hand_translation_cam_frame = np.squeeze(hand_translation_cam_frame[:, :, :3])
  hand_rot_cam_frame = np.squeeze(hand_pose_cam_frame[:, :, :3, : 3])
  # hand_pose_cam_frame = 
  return {
      "active_flag": active_flag,
      "hand_pose": hand_pose,
      "hand_trans_cam_frame": hand_translation_cam_frame,
      "hand_rot_cam_frame": hand_rot_cam_frame,
      "hand_pose_cam_frame": hand_pose_cam_frame,
      "valid_state": joint_valid,
      "track_state": track_state
  }


def parse_single_seq_hand(
    seq_name: str,
    dataset_root: str,
):

  left_file_path = os.path.join(
      dataset_root, seq_name, "Export_py", LEFT_HAND_SUFFIX
  )
  right_file_path = os.path.join(
      dataset_root, seq_name, "Export_py", RIGHT_HAND_SUFFIX
  )

  pose_file_path = os.path.join(
      dataset_root, seq_name, "Export_py", POSE_SUFFIX
  )
  cam_pose = parse_pose_data(pose_file_path)

  video_path = os.path.join(
      dataset_root, seq_name, "Export_py/Video_pitchshift.mp4"
  )

  if not os.path.isfile(video_path):
    return {}

  seq_data = {
      "cam_pose": cam_pose,
      "left_hand_pose": parse_hand_data(
        np.loadtxt(left_file_path), cam_pose
      ),
      "right_hand_pose": parse_hand_data(
        np.loadtxt(right_file_path), cam_pose
      ),
  }
  return seq_data

from typing import Dict, List

def parse_single_seq_language(
    seq_name: str,
    annotations: Dict,
    seq_data: List,
    # frame_skip: int = 30,
    video_fps: int = 30,
):
  if seq_name not in annotations:
    return

  seq_annotation = annotations[seq_name]

  num_frames = len(seq_data)
  current_time = 0

  events = seq_annotation["events"]
  events = sorted(
      events, key=lambda x: x['start']
  )
  seq_data_with_language = []
  # prefix = "You are doing some manipulation, instructions are as follows: "
  # prefix = "What should the robot do to "
  prefix = ""

  padding_time = 1

  # filter_top10 = False
  # top10_verb = ['approach', 'lift', 'unscrew', 'place', 'withdraw', 'screw', 'rotate', 'align', 'insert', 'disassemble', 'grab']
  # top10_noun = ['joy_con_controller', 'nightstand', 'hex_socket_head', 'hexagon_cap_nut', 'screwdriver', 'screw', 'allen_wrench', 'leg', 'hexagonal_wrench', 'tray']

  # filter_top3 = False
  # top3_verb = ['place', 'grab', 'insert']
  # top3_noun = ['leg', 'allen_wrench', 'hex_socket_head']

  for current_frame in range(num_frames):
    # current_time = current_frame * frame_skip / video_fps
    current_time = seq_data[current_frame]["frame_count"] / video_fps + padding_time
    current_language_input = copy.deepcopy(prefix)
    current_language_input_empty = "finish the task"
    # Event sorted by time
    for event in events:
      if current_time < event["start"] or \
        current_time > event["end"]:
        # if current_time > event["end"]:
        continue

      if  event["end"] - event["start"] < 1:
        # if current_time > event["end"]:
        continue

      current_language_input_short = copy.deepcopy(prefix)
      current_language_input_full = copy.deepcopy(current_language_input)
      label = event["label"]

      verb = ""
      noun = ""
      adj = ""
      adv = ""

      if label == "Coarse grained action":
        current_language_input += f" in long term, {str(event['attributes']['Verb'])} {str(event['attributes']['Noun'])}"
      elif label == "Fine grained action":
        current_language_input_full += f" in short term: {str(event['attributes']['Verb'])}"
        verb = str(event['attributes']['Verb'])
        # if filter_top10 and verb not in top10_verb:
        #   verb = "manipulate"
        # if filter_top3 and verb not in top3_verb:
        #   verb = "manipulate"
        current_language_input_short += f"{verb}"

      # if label == "Fine grained action":
        # current_language_input += f" {str(event['attributes']['Verb'])}"
        if 'Adjective' in event['attributes'] and event['attributes']['Adjective'] != 'none':
          current_language_input_full += (" " + str(event['attributes']['Adjective']))
          adj = str(event['attributes']['Adjective'])
          # current_language_input_short += (" " + str(event['attributes']['Adjective']))
        if event['attributes']['Noun'] != 'none':
          noun = str(event['attributes']['Noun'])
          # if filter_top10 and noun not in top10_noun:
          #   noun = "object"
          # if filter_top3 and noun not in top3_noun:
          #   noun = "object"
          noun = noun.replace("_", " ")
          noun = noun.split(" ")[-1]
          current_language_input_full += (" " + noun)
          current_language_input_short += (" " + noun)
        if 'adverbial' in event['attributes'] and event['attributes']['adverbial'] != 'none':
          current_language_input_full += (" " + str(event['attributes']['adverbial']))
          adv = str(event['attributes']['adverbial'])
          # current_language_input_short += (" " + str(event['attributes']['adverbial']))

      # current_language_input += "? A:"
      if len(current_language_input_short) == 0:
        continue
      new_sample = copy.deepcopy(seq_data[current_frame])
      new_sample["language_label"] = current_language_input_full
      new_sample["language_label_short"] = current_language_input_short
      new_sample["language_label_empty"] = current_language_input_empty
      new_sample["language_label_verb"] = verb
      new_sample["language_label_noun"] = noun
      new_sample["language_label_adj"] = adj
      new_sample["language_label_adj"] = adv
      new_sample["language_response"] = sample_text_response()
      seq_data_with_language.append(new_sample)
    # # current_language_input += "? A:"
    # seq_data[current_frame]["language_label"] = current_language_input
    # seq_data[current_frame]["language_label_short"] = current_language_input_short
    # seq_data[current_frame]["language_label_empty"] = current_language_input_empty
    # seq_data[current_frame]["language_response"] = sample_text_response()
  return seq_data_with_language


def transform_points_to_current_frame(hand_pose, cam_pose):

  hand_pose = hand_pose.reshape(-1, 26, 4, 4)
  # Only The Translation
  hand_translation = hand_pose[:, :, :, -1:]

  assert np.prod(cam_pose.shape) == 16, "Cam Pose for a single step"
  hand_translation_cam_frame = AXIS_TRANSFORM[np.newaxis, np.newaxis, :, :] @ \
    np.linalg.inv(cam_pose)[np.newaxis, np.newaxis, :, :] @ \
    hand_translation

  hand_pose_cam_frame = AXIS_TRANSFORM[np.newaxis, np.newaxis, :, :] @ \
    np.linalg.inv(cam_pose)[np.newaxis, np.newaxis, :, :] @ \
    hand_pose

  hand_translation_cam_frame = np.squeeze(hand_translation_cam_frame[:, :, :3])
  hand_pose_cam_frame = np.squeeze(hand_pose_cam_frame)
  # print(hand_translation_cam_frame.shape,hand_pose_cam_frame.shape, np.squeeze(hand_pose_cam_frame[..., :3, -1:]).shape)
  assert np.allclose(
    hand_translation_cam_frame, np.squeeze(hand_pose_cam_frame[..., :3, -1:]),
    equal_nan=True
  )
  # hand_rot_cam_frame = np.squeeze(hand_pose_cam_frame[:, :, :3, : 3])
  return hand_translation_cam_frame, hand_pose_cam_frame

def get_hand_data(hand_dict: dict, current_cam_pose, idx: int, idx_len: int = 1, frame_skip: int = 1):
  transformed_trans, transformed_pose = transform_points_to_current_frame(
    hand_dict["hand_pose"][idx: idx + idx_len:frame_skip], current_cam_pose
  )
  # relative_rotation = convert_kp_pose_to_mano_rotation(
  #   hand_dict["hand_pose"][idx: idx + idx_len:frame_skip]
  # print(hand_dict["mano_parameters"].shape, idx, idx + idx_len, frame_skip)
  return {
      "active_flag": hand_dict["active_flag"][idx: idx + idx_len:frame_skip],
      "hand_pose": hand_dict["hand_pose"][idx: idx + idx_len:frame_skip],
      "hand_trans_cam_frame": hand_dict["hand_trans_cam_frame"][idx: idx + idx_len:frame_skip],
      "valid_state": hand_dict["valid_state"][idx: idx + idx_len:frame_skip],
      "track_state": hand_dict["track_state"][idx: idx + idx_len:frame_skip],
      "combined_valid_state": hand_dict["valid_state"][idx: idx + idx_len:frame_skip] * \
        hand_dict["track_state"][idx: idx + idx_len:frame_skip],
      "transformed_hand_trans_cam_frame": transformed_trans,
      "transformed_hand_pose_cam_frame": transformed_pose,
      # mano_rotation
      "mano_parameters":hand_dict["mano_parameters"][idx: idx + idx_len:frame_skip]
  }


def convert_annotation(annotations):
  result_dict = {}
  for seq_annotation in annotations:
    result_dict[seq_annotation["video_name"]] = seq_annotation
  return result_dict

# Merge Hand / Image / Language Label
def parse_single_seq(
    seq_name: str,
    dataset_root: str,
    annotations: Dict,
    frame_skip: int,
    future_len: int,
    image_w: int,
    image_h: int,
    video_fps: int=30
):
  video_path = os.path.join(
      dataset_root, seq_name, "Export_py/Video_pitchshift.mp4"
  )
  if not os.path.isfile(video_path):
    return []

  # Test Sequence is not available
  if seq_name not in annotations:
    return []

  hand_data = parse_single_seq_hand(seq_name, dataset_root)
  # print(hand_data)
  full_len = hand_data["left_hand_pose"]["active_flag"].shape[0]

  with open(os.path.join(
    dataset_root, seq_name, "processed_hand_data.pkl"
  ), "wb") as f:
    pickle.dump(hand_data, f)
  # Capture the video from the file
  cap = cv2.VideoCapture(video_path)

  frame_count = -1
  
  seq_data = []
  while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frame_count += 1
    # If frame is read correctly ret is True
    if not ret:
      break

    if frame_count % frame_skip != 0:
      continue

    if frame_count + future_len > full_len:
      break
    
    if image_w != 898 or image_h != 256:
      frame = cv2.resize(frame, (image_w, image_h))

    current_data = {
        "rgb_obs": frame,
        "frame_count": frame_count,
        "current_left_hand_pose": get_hand_data(hand_data["left_hand_pose"], frame_count),
        "current_right_hand_pose": get_hand_data(hand_data["right_hand_pose"], frame_count),
        "future_left_hand_pose": get_hand_data(hand_data["left_hand_pose"], frame_count, future_len),
        "future_right_hand_pose": get_hand_data(hand_data["right_hand_pose"], frame_count, future_len),
    }
    if np.sum(current_data["current_left_hand_pose"]["valid_state"]) + \
      np.sum(current_data["current_right_hand_pose"]["valid_state"]) < 13:
      # At least half need to be visible
      continue
    # if np.sum(current_data["current_right_hand_pose"]["valid_state"]) < 13:
    #   # At least half need to be visible
    #   continue
    seq_data.append(current_data)
  # When everything done, release the capture
  cap.release()
  # print("CAP release?")
  parse_single_seq_language(
    seq_name,
    annotations,
    seq_data,
    # frame_skip,
    video_fps
  )

  return seq_data


def update_dict(
  src_dict, tgt_dict, prefix
):
  for key in tgt_dict.keys():
    src_dict[prefix + "/" + key] = tgt_dict[key]


# Merge Hand / Image / Language Label
def parse_single_seq_image(
    seq_name: str,
    dataset_root: str,
    # annotations: Dict,
    frame_skip: int,
    image_w: int,
    image_h: int,
    # video_fps: int=30
):
  video_path = os.path.join(
      dataset_root, seq_name, "Export_py/Video_pitchshift.mp4"
  )
  if not os.path.isfile(video_path):
    return []

  # Capture the video from the file
  cap = cv2.VideoCapture(video_path)

  frame_count = -1
  
  seq_data = []

  while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frame_count += 1
    # If frame is read correctly ret is True
    if not ret:
      break

    if frame_count % frame_skip != 0:
      continue

    # if frame_count + future_len > full_len:
      # break
    
    if image_w != 454 or image_h != 256:
      frame = cv2.resize(frame, (image_w, image_h))

    current_data = {
      "seq_name": seq_name,
      "frame_count": frame_count,
      "rgb_obs": frame,
    }
    seq_data.append(current_data)
  # When everything done, release the capture
  cap.release()
  # print("CAP release?")
  # parse_single_seq_language(
  #   seq_name,
  #   # annotations,
  #   seq_data,
  #   # frame_skip,
  #   video_fps
  # )
  return seq_data


from human_plan.visualize.holoassist.utils import (
  read_intrinsics_txt,
  project_one_hand
)

def count_valid_one_hand(
  trans, cam_intrinsics, width, height
):
  c_trans = trans.copy().reshape(26, 3).astype(dtype=np.float64)
  # Put an empty camera pose for image.
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])

  points, _ = cv2.projectPoints(
      # hand_points[:3], rvec, tvec, img_intrinsics, np.array([]))
      c_trans, 
      rvec, tvec, cam_intrinsics, np.array([])
  )
  # print(points.shape)
  # print(trans.shape, len(points), points[0])
  valid_count = 0
  inframe_valid_mask = np.zeros(26)
  projected_2d_point = np.zeros((26, 2))
  # print(points)
  if np.isnan(points).any():
    return 0, inframe_valid_mask, projected_2d_point
  for idx, p in enumerate(points):
    # print(p)
    # print(int(p[0][0]), int(p[0][1]), width, height)
    if int(p[0][0]) < width and int(p[0][0]) > 0 and \
      int(p[0][1]) < height and int(p[0][1]) > 0:
      # print(p)
      # print(int(p[0][0]), int(p[0][1]), width, height)
      valid_count += 1
      inframe_valid_mask[idx] = 1
      projected_2d_point[idx, 0] = p[0][0] / width
      projected_2d_point[idx, 1] = p[0][1] / height

  return valid_count, inframe_valid_mask, projected_2d_point


from human_plan.dataset_preprocessing.utils.mano_utils import (
  obtain_mano_parameters_holoassist,
  mano_left,
  mano_right,
)

def obtain_mano_parameters_seq(hand_infos):
  mano_kps3d_right = obtain_mano_parameters_holoassist(
    mano_right, hand_infos["right_hand_pose"], is_right=True
  )

  mano_kps3d_left = obtain_mano_parameters_holoassist(
    mano_left, hand_infos["left_hand_pose"], is_right=False
  )
  hand_infos["left_hand_pose"]["mano_parameters"] = mano_kps3d_left["optimized_mano_parameters"]
  hand_infos["right_hand_pose"]["mano_parameters"] = mano_kps3d_right["optimized_mano_parameters"]

# Merge Hand / Image / Language Label
def parse_single_seq_hand_only(
    seq_name: str,
    dataset_root: str,
    annotations: Dict,
    frame_skip: int,
    sample_skip:int,
    future_len: int,
    # video_fps: int=30,
    image_mapping_dict = None
):
  video_path = os.path.join(
      dataset_root, seq_name, "Export_py/Video_pitchshift.mp4"
  )
  # print(video_path)
  if not os.path.isfile(video_path):
    return []
  video = cv2.VideoCapture(video_path)
  video_fps = video.get(cv2.CAP_PROP_FPS)
  video.release()

  intrinsics_path = os.path.join(
      dataset_root, seq_name, 
      "Export_py", "Video/Intrinsics.txt"
  )
  cam_intrinsics, width, height = read_intrinsics_txt(
      intrinsics_path
  )

  hand_data = parse_single_seq_hand(seq_name, dataset_root)
  full_len = hand_data["left_hand_pose"]["active_flag"].shape[0]

  obtain_mano_parameters_seq(hand_data)

  frame_count = -1
  seq_data = []

  for frame_count in range(0, full_len, frame_skip * sample_skip):
    if frame_count + future_len > full_len:
      break

    if frame_count not in image_mapping_dict[seq_name]:
      break
    # .reshape(-1, 4, 4)
    current_data = {
      "seq_name": seq_name,
      "cam_intrinsics": cam_intrinsics,
      "raw_width": width,
      "raw_height": height,
      "frame_count": frame_count,
      "current_cam_pose": hand_data["cam_pose"][frame_count],
      "future_cam_pose": hand_data["cam_pose"][frame_count: frame_count + future_len: frame_skip],
    }
    assert cam_intrinsics is not None
    # print(cam_intrinsics)
    update_dict(
      current_data, 
      get_hand_data(hand_data["left_hand_pose"], hand_data["cam_pose"][frame_count], frame_count),
      prefix="current_left_hand_pose", 
    )
    update_dict(
      current_data,
      get_hand_data(hand_data["right_hand_pose"], hand_data["cam_pose"][frame_count], frame_count),
      prefix="current_right_hand_pose", 
    )
    update_dict(
      current_data,
      get_hand_data(hand_data["left_hand_pose"], hand_data["cam_pose"][frame_count], frame_count, future_len, frame_skip),
      prefix="future_left_hand_pose", 
    )
    update_dict(
      current_data,
      get_hand_data(hand_data["right_hand_pose"], hand_data["cam_pose"][frame_count], frame_count, future_len, frame_skip),
      prefix="future_right_hand_pose", 
    )

    his_end = max(0, frame_count - future_len)
    # print(frame_count, his_end, -frame_skip)
    update_dict(
      current_data,
      get_hand_data(hand_data["left_hand_pose"], hand_data["cam_pose"][frame_count], frame_count, his_end - frame_count, -frame_skip),
      prefix="history_left_hand_pose",
    )
    update_dict(
      current_data,
      get_hand_data(hand_data["right_hand_pose"], hand_data["cam_pose"][frame_count], frame_count, his_end - frame_count, -frame_skip),
      prefix="history_right_hand_pose", 
    )
    current_data["history_cam_pose"] = hand_data["cam_pose"][frame_count: his_end: -frame_skip]

    current_left_valid_count, current_left_inframe_mask, projected_2d_points_left = count_valid_one_hand(
      current_data["current_left_hand_pose/hand_trans_cam_frame"],
      cam_intrinsics, width, height
    )

    current_right_valid_count, current_right_inframe_mask, projected_2d_points_right = count_valid_one_hand(
      current_data["current_right_hand_pose/hand_trans_cam_frame"],
      cam_intrinsics, width, height
    )
    # print(current_left_valid_count, current_right_valid_count)
    # skip_sample = False
    valid_data_flag = False
    if current_data["current_left_hand_pose/combined_valid_state"].reshape(26)[1] == 1 and \
      current_left_inframe_mask[1] == 1 and \
      np.sum(current_data["current_left_hand_pose/combined_valid_state"]) >= 20 and \
      current_left_valid_count >= 20:
      # At least half need to be visible
      valid_data_flag = True

    if current_data["current_right_hand_pose/combined_valid_state"].reshape(26)[1] == 1 and \
      current_right_inframe_mask[1] == 1 and \
      np.sum(current_data["current_right_hand_pose/combined_valid_state"]) >= 20 and \
      current_right_valid_count >= 20:
      # At least half need to be visible
      valid_data_flag = True

    if not valid_data_flag:
      continue

    current_data["current_left_hand_inframe_mask"] = current_left_inframe_mask
    current_data["current_right_hand_inframe_mask"] = current_right_inframe_mask
    # projected point has been normalized 
    current_data["current_left_hand_2d_point"] = projected_2d_points_left
    current_data["current_right_hand_2d_point"] = projected_2d_points_right

    # validation_future_len = 31
    validation_future_len = future_len // frame_skip

    future_left_inframe_mask = []
    future_right_inframe_mask = []
    future_left_2d_points = []
    future_right_2d_points = []

    future_trans_left_2d_points = []
    future_trans_right_2d_points = []

    left_total_valid_count = 0
    left_future_valid_flag = False
    for i in range(validation_future_len):
      f_left_valid_count, f_left_inframe_mask, f_left_2d = count_valid_one_hand(
        current_data["future_left_hand_pose/transformed_hand_trans_cam_frame"][i],
        cam_intrinsics, width, height
      )
      f_trans_left_valid_count, f_trans_left_inframe_mask, f_trans_left_2d = count_valid_one_hand(
        current_data["future_left_hand_pose/transformed_hand_trans_cam_frame"][i],
        cam_intrinsics, width, height
      )

      if current_data["future_left_hand_pose/combined_valid_state"][i].reshape(26)[1] and \
        f_left_inframe_mask[1] == 1 and f_trans_left_inframe_mask[1] == 1 and \
        np.sum(current_data["future_left_hand_pose/combined_valid_state"][i]) >= 20 and \
        f_left_valid_count >= 20 and f_trans_left_valid_count >= 20:
        left_total_valid_count += 1

      future_left_inframe_mask.append(
        f_left_inframe_mask * f_trans_left_inframe_mask
      )
      future_left_2d_points.append(
        f_left_2d
      )
      future_trans_left_2d_points.append(
        f_trans_left_2d
      )

    # if total_valid_count < math.ceil(validation_future_len / 2):
    if left_total_valid_count >= validation_future_len * 0.9:
      # print("Skip - not enought total valid count")
      left_future_valid_flag = True

    right_total_valid_count = 0
    right_future_valid_flag = False
    for i in range(validation_future_len):
      f_right_valid_count, f_right_inframe_mask, f_right_2d = count_valid_one_hand(
        current_data["future_right_hand_pose/transformed_hand_trans_cam_frame"][i],
        cam_intrinsics, width, height
      )
      f_trans_right_valid_count, f_trans_right_inframe_mask, f_trans_right_2d = count_valid_one_hand(
        current_data["future_right_hand_pose/transformed_hand_trans_cam_frame"][i],
        cam_intrinsics, width, height
      )

      if current_data["future_right_hand_pose/combined_valid_state"][i].reshape(26)[1] and \
        f_right_inframe_mask[1] == 1 and f_trans_right_inframe_mask[1] == 1 and \
        np.sum(current_data["future_right_hand_pose/combined_valid_state"][i]) >= 20 and \
        f_right_valid_count >= 20 and f_trans_right_valid_count >= 20:
        right_total_valid_count += 1

      future_right_inframe_mask.append(
        f_right_inframe_mask * f_trans_right_inframe_mask
      )
      future_right_2d_points.append(
        f_right_2d
      )
      future_trans_right_2d_points.append(
        f_trans_right_2d
      )

    # if total_valid_count < math.ceil(validation_future_len / 2):
    if right_total_valid_count >= validation_future_len * 0.9:
      # print("Skip - not enought total valid count")
      right_future_valid_flag = True

    if not (left_future_valid_flag or right_future_valid_flag):
      # print("Skip - not enought total valid count")
      continue

    future_left_inframe_mask = np.stack(
      future_left_inframe_mask
    )
    future_right_inframe_mask = np.stack(
      future_right_inframe_mask
    )
    future_left_2d_points = np.stack(
      future_left_2d_points
    )
    future_right_2d_points = np.stack(
      future_right_2d_points
    )
    future_trans_left_2d_points = np.stack(
      future_trans_left_2d_points
    )
    future_trans_right_2d_points = np.stack(
      future_trans_right_2d_points
    )
    current_data["future_left_hand_inframe_mask"] = future_left_inframe_mask
    current_data["future_right_hand_inframe_mask"] = future_right_inframe_mask
    current_data["future_left_hand_2d"] = future_left_2d_points
    current_data["future_right_hand_2d"] = future_right_2d_points
    current_data["future_trans_left_hand_2d"] = future_trans_left_2d_points
    current_data["future_trans_right_hand_2d"] = future_trans_right_2d_points

    seq_data.append(current_data)
    # print(current_data["future_right_hand_pose/hand_trans_cam_frame"][0].shape)
  # When everything done, release the capture

  seq_data = parse_single_seq_language(
    seq_name,
    annotations,
    seq_data,
    # frame_skip,
    video_fps
  )
  seq_data = [d for d in seq_data if len(d["language_label"]) > 2]
  return seq_data


# Merge Hand / Image / Language Label
def parse_single_seq_single_layer_with_depth(
    seq_name: str,
    dataset_root: str,
    depth_model: object,
    annotations: Dict,
    frame_skip: int,
    future_len: int,
    image_w: int,
    image_h: int,
    video_fps: int=30
):
  video_path = os.path.join(
      dataset_root, seq_name, "Export_py/Video_pitchshift.mp4"
  )
  if not os.path.isfile(video_path):
    return []

  hand_data = parse_single_seq_hand(seq_name, dataset_root)
  full_len = hand_data["left_hand_pose"]["active_flag"].shape[0]

  # Capture the video from the file
  cap = cv2.VideoCapture(video_path)

  frame_count = -1
  
  seq_data = []
  
  # depth_seq_root = os.path.join(
  #   dataset_root, seq_name, "Export_py", "AhatDepth"
  # )
  # depth_files = os.listdir(depth_seq_root)

  # depth_id_list, depth_id_dict = get_id_list_and_id_file_mapping(depth_files)

  # Find closest image
  while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frame_count += 1
    # If frame is read correctly ret is True
    if not ret:
      break

    if frame_count % frame_skip != 0:
      continue

    if frame_count + future_len > full_len:
      break
    
    if image_w != 454 or image_h != 256:
      frame = cv2.resize(frame, (image_w, image_h))

    # depth_file_name = find_closest_depth(
    #   depth_id_list, frame_count, depth_id_dict
    # )
    
    depth = depth_model.infer_image(frame) # HxW raw depth map in numpy
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = (depth * 255).astype(np.uint8)

    current_data = {
        "rgb_obs": frame,
        "depth_obs": depth,
        "frame_count": frame_count,
    }
    update_dict(
      current_data, get_hand_data(hand_data["left_hand_pose"], frame_count), prefix="current_left_hand_pose", 
    )
    update_dict(
      current_data, get_hand_data(hand_data["right_hand_pose"], frame_count), prefix="current_right_hand_pose", 
    )
    update_dict(
      current_data, get_hand_data(hand_data["left_hand_pose"], frame_count, future_len), prefix="future_left_hand_pose", 
    )
    update_dict(
      current_data, get_hand_data(hand_data["right_hand_pose"], frame_count, future_len), prefix="future_right_hand_pose", 
    )
    # "current_left_hand_pose": get_hand_data(hand_data["left_hand_pose"], frame_count),
    # "current_right_hand_pose": get_hand_data(hand_data["right_hand_pose"], frame_count),
    # "future_left_hand_pose": get_hand_data(hand_data["left_hand_pose"], frame_count, future_len),
    # "future_right_hand_pose": get_hand_data(hand_data["right_hand_pose"], frame_count, future_len),

    if np.sum(current_data["current_left_hand_pose/valid_state"]) + \
      np.sum(current_data["current_right_hand_pose/valid_state"]) < 13:
      # At least half need to be visible
      continue
    # if np.sum(current_data["current_right_hand_pose"]["valid_state"]) < 13:
    #   # At least half need to be visible
    #   continue
    seq_data.append(current_data)
  # When everything done, release the capture
  cap.release()
  # print("CAP release?")
  parse_single_seq_language(
    seq_name,
    annotations,
    seq_data,
    # frame_skip,
    video_fps
  )
  return seq_data