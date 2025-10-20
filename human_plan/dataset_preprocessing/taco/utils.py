from human_plan.utils.visualization import project_points
from pathlib import Path
from human_plan.utils.visualization import plot_hand_mano, plot_points
from human_plan.utils.mano.forward import (
    mano_forward,
    # mano_right
)
from human_plan.utils.mano.constants import (
    LEFT_PELVIS,
    RIGHT_PELVIS
)
from human_plan.utils.mano.model import (
    mano_left,
    mano_right
)
from human_plan.utils.hand_dof import convert_full_mano_to_pca_dof
from manopth.manolayer import ManoLayer
import pickle
import torch
import torch.nn as nn
import os
import cv2
import zipfile
import subprocess
# import natsort
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

from human_plan.dataset_preprocessing.utils.transformations import (
    transform_to_current_frame,
    transform_to_world_frame,
    project_set,
)


TACO_WIDTH = 1920
TACO_HEIGHT = 1080


def get_all_seqs(dataset_root: str):
  video_root = os.path.join(
      dataset_root,
      "Egocentric_RGB_Videos"
  )
  sequence_lists = []
  for task in os.listdir(video_root):
    task_folder = os.path.join(video_root, task)
    for demo in os.listdir(task_folder):
      sequence_lists.append(os.path.join(task, demo))
  return sequence_lists


# Merge Hand / Image / Language Label
def parse_single_seq_image(
    dataset_root: str,
    seq_name: str,
    frame_skip: int,
    image_w: int,
    image_h: int,
):

  raw_video_path = os.path.join(
      dataset_root,
      "Egocentric_RGB_Videos",
      seq_name,
      "color.mp4"
  )
  if not os.path.isfile(raw_video_path):
    return []

  # Capture the video from the file
  cap = cv2.VideoCapture(raw_video_path)
  # original_fps = cap.get(cv2.CAP_PROP_FPS)
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

    frame = cv2.resize(frame, (image_w, image_h))
    current_data = {
        "seq_name": seq_name,
        "frame_count": frame_count,
        "rgb_obs": frame,
    }
    seq_data.append(current_data)
  # When everything done, release the capture
  cap.release()
  return seq_data


def pkl2hand_info(folder: str, hand: str, frame_skip: int):
  import pickle as pkl
  from os.path import join

  with open(join(folder, f"{hand}_hand.pkl"), "rb") as f:
    hand_info = pkl.load(f)

  hand_poses = []
  hand_trans = []

  keys = list(hand_info.keys())
  keys = sorted(keys)
  for k in keys:
    hand_poses.append(hand_info[k]["hand_pose"])
    hand_trans.append(hand_info[k]["hand_trans"])

  hand_poses = np.stack(hand_poses)
  hand_trans = np.stack(hand_trans)

  with open(join(folder, f"{hand}_hand_shape.pkl"), "rb") as f:
    mano_beta = pkl.load(f)["hand_shape"]
    mano_beta = mano_beta.reshape(10).detach().cpu().numpy()
  print(hand_poses.shape)
  print(hand_poses[::frame_skip].shape)
  return hand_poses[::frame_skip], hand_trans[::frame_skip], mano_beta


taco_manolayer_left = ManoLayer(
    mano_root="mano_v1_2/models",
    use_pca=False,
    ncomps=45,
    flat_hand_mean=True,
    side="left",
    center_idx=0,
)


taco_manolayer_right = ManoLayer(
    mano_root="mano_v1_2/models",
    use_pca=False,
    ncomps=45,
    flat_hand_mean=True,
    side="right",
    center_idx=0,
)


def convert_mano_representation_to_default_version(
    hand_info_tuple: tuple,
    side: str = "right",
):
  # TACO using the following configuration:
  # manolayer = ManoLayer(
  #     mano_root="mano_v1_2/models",
  #     use_pca=False,
  #     ncomps=45,
  #     flat_hand_mean=True,
  #     side=side,
  #     center_idx=0,
  # )

  # Flat hand mean interfears with the default translation
  # center_idx cause an offset in the wrist position
  mano_model = mano_right if side == "right" else mano_left

  theta = torch.from_numpy(
      np.float32(hand_info_tuple[0])
  )[:, 3:]

  global_rot = torch.from_numpy(
      np.float32(hand_info_tuple[0]).copy()
  )[:, :3]

  hand_pca = convert_full_mano_to_pca_dof(
      theta.detach().cpu().numpy(),
      mano_model.hand_mean.detach().cpu().numpy(),
      mano_model.np_hand_components
  )

  pelvis = RIGHT_PELVIS if side == "right" else LEFT_PELVIS
  global_trans = torch.Tensor(hand_info_tuple[1]) - pelvis

  return hand_pca, global_trans, global_rot


def get_3d_pos_single_hand(
        hand_info_tuple: tuple,
        side: str = "right"
):
  manolayer = ManoLayer(
      mano_root="mano_v1_2/models",
      use_pca=False,
      ncomps=45,
      flat_hand_mean=True,
      side=side,
      center_idx=0,
  )
  theta = torch.from_numpy(
      np.float32(hand_info_tuple[0])
  )

  trans = torch.from_numpy(
      np.float32(hand_info_tuple[1])
  )
  betas = torch.from_numpy(
      hand_info_tuple[2]
  ).unsqueeze(0).to(torch.float32)
  betas = betas.repeat(theta.shape[0], 1)  # (N_frame, 10)
  _, hand_joints_pred = manolayer(theta, betas)
  hand_joints_pred = hand_joints_pred / 1000.0
#   print(hand_joints_pred.shape, trans.shape)
  kps3d = hand_joints_pred + trans.unsqueeze(1)  # (N_frame, 21, 3)

  # theta shape (1, 48), first three dim -> axis angle
  theta_np = np.array(theta)
  wrist_rot = theta_np[:, :3]

  r = R.from_rotvec(wrist_rot)
  # Convert to rotation matrix
  wrist_rot = r.as_matrix()

  return kps3d, trans, wrist_rot


def get_3d_pos_single_hand_raw(
    hand_component: torch.Tensor,
    global_trans: torch.Tensor,
    global_rot: torch.Tensor,
    side: str = "right"
):
  mano_model = mano_right if side == "right" else mano_left

  hand_component = torch.Tensor(hand_component)
  global_trans = torch.Tensor(global_trans)
  global_rot = torch.Tensor(global_rot)

  hand_kp_3d = mano_forward(
      mano_model,
      hand_component,
      global_rot,
      global_trans
  )

  return hand_kp_3d, global_trans, global_rot


def load_cam_intrinsics(file_path):
  with open(file_path, "r") as f:
    lines = f.readlines()
    fx, _, cx = map(float, lines[0].split())
    _, fy, cy = map(float, lines[1].split())
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
  return K

# Merge Hand / Image / Language Label

def obtain_single_seq_hand(
    dataset_root: str,
    seq_name: str,
    frame_skip: int,
):
  # TODO generate language instrcution based on task triplets
  phrases = seq_name.split("/")[-2].strip("()").split(", ")

  task_language_label = {}

  verb, tool, obj = phrases
  task_language_label["verb"] = verb
  task_language_label["noun"] = obj
  task_language_label["short"] = "Use {} to {} {}".format(
      tool,
      verb,
      obj
  )

  # TODO get camera intrinsics
  cam_params_dir = os.path.join(
      dataset_root,
      "Egocentric_Camera_Parameters",
      seq_name
  )

  cam_intrinsics = load_cam_intrinsics(
      os.path.join(cam_params_dir, "egocentric_intrinsic.txt")
  )
  # load camera matrix
#   print(cam_extrinsics)
  cam_extrinsics = np.load(
      os.path.join(cam_params_dir, "egocentric_frame_extrinsic.npy")
  )[::frame_skip]

  hand_pose = []
  hand_trans = []
  hand_rot = []

  hand_pose_converted = []
  hand_trans_converted = []
  hand_rot_converted = []
  hand_theta_converted = []
  # for hand_file in hand_file_list:

  hand_directory = os.path.join(
      dataset_root,
      "Hand_Poses",
      seq_name
  )
  for hand_side in ["left", "right"]:
    hand_info = pkl2hand_info(hand_directory, hand_side, frame_skip)
    # separates by hands
    hand_kps_3d, wrist_trans, wrist_rot = get_3d_pos_single_hand(
        hand_info,
        hand_side
    )

    hand_component, global_trans, global_rot = convert_mano_representation_to_default_version(
        hand_info,
        side=hand_side
    )

    hand_kps_3d_converted, global_trans_converted, global_rot_converted = get_3d_pos_single_hand_raw(
        hand_component,
        global_trans,
        global_rot,
        side=hand_side
    )

    hand_pose.append(
        np.squeeze(hand_kps_3d.detach().cpu().numpy())
    )
    hand_trans.append(
        np.squeeze(wrist_trans.detach().cpu().numpy())
    )

    hand_rot.append(wrist_rot)

    hand_pose_converted.append(
        np.squeeze(hand_kps_3d_converted.detach().cpu().numpy())
    )
    hand_trans_converted.append(
        np.squeeze(global_trans_converted.detach().cpu().numpy())
    )
    hand_rot_converted.append(
        np.squeeze(global_rot_converted.detach().cpu().numpy())
    )

    hand_theta_converted.append(
        np.squeeze(hand_component)
    )

  assert len(hand_pose[0]) == len(cam_extrinsics)
  min_len = len(cam_extrinsics)

  # TACO actually stored the inverse of the camera pose
  cam_pose = np.stack(cam_extrinsics)[:min_len]  # (seq_len, 4, 4)
  # World Frame
  hand_pose = np.stack(hand_pose)[:min_len]  # (2, seq_len, 21, 3)
  # World Frame
  hand_trans = np.stack(hand_trans)[:min_len]  # (2, seq_len, 3)
  # World Frame
  hand_rot = np.stack(hand_rot)[:min_len]  # (2, seq_len, 3, 3)

  hand_pose_converted = np.stack(hand_pose_converted)[
      :min_len]  # (2, seq_len, 21, 3)
  # World Frame
  hand_trans_converted = np.stack(hand_trans_converted)[
      :min_len]  # (2, seq_len, 3)
  # World Frame
  hand_rot_converted = np.stack(hand_rot_converted)[
      :min_len]  # (2, seq_len, 3, 3)

  converted_r = R.from_rotvec(hand_rot_converted.reshape(2 * min_len, 3))
  # Convert to rotation matrix
  hand_rot_converted = converted_r.as_matrix().reshape(2, min_len, 3, 3)


  # print("Hand Trans Converted Shape:", hand_trans_converted.shape)
  # print("Hand Rot Converted Shape:", hand_rot_converted.shape)

  hand_theta_converted = np.stack(hand_theta_converted)[
      :min_len
  ]

  return (
      task_language_label,
      cam_pose,
      hand_pose, hand_trans, hand_rot,
      hand_pose_converted, hand_trans_converted, hand_rot_converted,
      hand_theta_converted,
      cam_intrinsics,
  )


def parse_single_seq_hand(
    dataset_root: str,
    seq_name: str,
    frame_skip: int,
    future_len: int,
    sample_skip: int,
    image_mapping_dict: dict
):
  task_language_label, \
    cam_pose, \
    hand_kp, ee_trans, ee_rot, \
    hand_kp_converted, ee_trans_converted, ee_rot_converted, \
    hand_component, \
    cam_intrinsics = obtain_single_seq_hand(
      dataset_root,
      seq_name,
      frame_skip,
  )

  print("Hand KP Shape:", hand_kp.shape)
  print("Hand Component Shape:",hand_component.shape)

  seq_data = []
  # sample skip: number of skipped future frames

  min_len = cam_pose.shape[0]
  for sample_idx in range(0, min_len, sample_skip):
    frame_count = sample_idx * frame_skip
    # sample
    current_cam_pose = cam_pose[sample_idx]
    # TACO stored inv of camera pose
    inv_cam_pose = current_cam_pose
    current_data = {}
    is_valid = True
    if not frame_count in image_mapping_dict[seq_name]:
        print(frame_count, seq_name)
        continue
    for hid, hand_side in enumerate(["left", "right"]):

      # Camera Frame
      # n x 20 x 3
      future_hand_kp = transform_to_current_frame(
          inv_cam_pose=inv_cam_pose,
          world_frame_pos=hand_kp_converted[hid][
            sample_idx: sample_idx + future_len
          ]
      )

      future_ee_trans_3d = transform_to_current_frame(
          inv_cam_pose=inv_cam_pose,
          world_frame_pos=ee_trans_converted[hid][
            sample_idx: sample_idx + future_len
          ],
      ).reshape(-1, 3)

      future_wrist_trans_3d = transform_to_current_frame(
          inv_cam_pose=inv_cam_pose,
          world_frame_pos=ee_trans[hid][
            sample_idx: sample_idx + future_len
          ],
      ).reshape(-1, 3)

      future_ee_rot = transform_to_current_frame(
          inv_cam_pose=inv_cam_pose,
          world_frame_pos=ee_rot_converted[hid][
              sample_idx: sample_idx + future_len
          ],
      )

      # current_hand_wrist_2d = hand_kps_2D[hid][sample_idx, 0]

      future_ee_trans_2d, future_ee_inframe_mask = project_set(
          future_ee_trans_3d, cam_intrinsics, TACO_WIDTH, TACO_HEIGHT
      )

      future_wrist_trans_2d, future_wrist_inframe_mask = project_set(
          future_wrist_trans_3d, cam_intrinsics, TACO_WIDTH, TACO_HEIGHT
      )

    #   print(future_ee_trans_3d.shape)

      future_hand_kp_2d, future_hand_kp_wrist_inframe_mask = project_set(
          future_hand_kp, cam_intrinsics, TACO_WIDTH, TACO_HEIGHT
      )

      is_valid = (
            # Current timestep must be in frame
            future_ee_inframe_mask[0] == 1
          and (
              np.sum(future_ee_inframe_mask)
              > (future_ee_inframe_mask.shape[0] * 0.9)
          )
          and (
              np.sum(future_wrist_inframe_mask)
              > (future_wrist_inframe_mask.shape[0] * 0.9)
          )
      )

      # if not is_valid:
      #   continue
    #   print(task_language_label["short"])
      current_data.update({
          "seq_name": seq_name,
          "frame_count": frame_count,
          "raw_width": TACO_WIDTH,
          "raw_height": TACO_HEIGHT,

          # Language label
          "language_label_verb": task_language_label["verb"],
          "language_label_noun": task_language_label["noun"],
          "language_label_short": task_language_label["short"],
          "language_label": task_language_label["short"],

          # Current
          f"current_{hand_side}_ee_trans_3d": future_ee_trans_3d[0],
          f"current_{hand_side}_ee_trans_2d": future_ee_trans_2d[0],
          f"current_{hand_side}_hand_kp": future_hand_kp[0],
          f"current_{hand_side}_flag": future_ee_inframe_mask[0],
          # As KP
          f"current_{hand_side}_wrist_trans_3d": future_wrist_trans_3d[0],
          f"current_{hand_side}_wrist_trans_2d": future_wrist_trans_2d[0],
          f"current_{hand_side}_kp_flag": future_wrist_inframe_mask[0],

          # Future
          f"future_{hand_side}_ee_trans_3d": future_ee_trans_3d,
          f"future_{hand_side}_ee_trans_2d": future_ee_trans_2d,
          f"future_{hand_side}_wrist_3d": future_wrist_trans_3d,
          f"future_{hand_side}_wrist_2d": future_wrist_trans_2d,
          f"future_{hand_side}_hand_kp": future_hand_kp,
          f"future_{hand_side}_flag": future_ee_inframe_mask,
          f"future_{hand_side}_kp_flag": future_wrist_inframe_mask,
          # MANO hand joint rot
          f"current_{hand_side}_pose_theta": hand_component[hid][sample_idx],
          f"future_{hand_side}_pose_theta": hand_component[hid][
              sample_idx: sample_idx + future_len
          ],
          # Rotation
          f"current_{hand_side}_wrist_rot": future_ee_rot[0],
          f"future_{hand_side}_wrist_rot": future_ee_rot,
          # "current_kp_2d": hand_kps_2D[hid][sample_idx],
      })
    seq_data.append(current_data)
  # for hand_data
  return seq_data


def visualize_one_step(
    frame,
    current_cam_pose,
    current_trans_left,
    current_trans_right,
    hand_pose,
    idx,
    cam_intrinsics,
):

  # current_cam_pose = cam_pose[i]
  # inv_cam_pose = np.linalg.inv(current_cam_pose)
  inv_cam_pose = current_cam_pose
  # inv_cam_pose = current_cam_pose.T

  trans_left_cam_frame = transform_to_current_frame(
      inv_cam_pose, current_trans_left,
  )
  trans_right_cam_frame = transform_to_current_frame(
      inv_cam_pose, current_trans_right,
  )

  trans_2d_left, _ = project_set(
      trans_left_cam_frame, cam_intrinsics,
      TACO_WIDTH, TACO_HEIGHT
  )
  trans_2d_right, _ = project_set(
      trans_right_cam_frame, cam_intrinsics,
      TACO_WIDTH, TACO_HEIGHT
  )

  frame = plot_points(trans_2d_left, frame, (0, 255, 0))
  frame = plot_points(trans_2d_right, frame, (0, 0, 255))

  hand_pose_left_cam_frame = transform_to_current_frame(
      inv_cam_pose, hand_pose[0][idx],
  )

  hand_pose_right_cam_frame = transform_to_current_frame(
      inv_cam_pose, hand_pose[1][idx],
  )

  hand_pose_2d_left, _ = project_set(
      hand_pose_left_cam_frame, cam_intrinsics,
      TACO_WIDTH, TACO_HEIGHT
  )

  hand_pose_2d_right, _ = project_set(
      hand_pose_right_cam_frame, cam_intrinsics,
      TACO_WIDTH, TACO_HEIGHT
  )

  frame = plot_hand_mano(
      hand_pose_2d_left,
      frame,
      (0, 0, 255)
  )

  frame = plot_hand_mano(
      hand_pose_2d_right,
      frame,
      (0, 255, 0)
  )
  return frame


def visualize_single_sequence(
    seq_images,
    seq_hands,
    save_root: str,
    seq_name: str,
):

  task_language_label, \
      cam_pose, hand_pose, hand_trans, hand_rot, \
      hand_pose_converted, hand_trans_converted, hand_rot_converted, \
      hand_theta_converted, \
      cam_intrinsics = seq_hands

  output_dir = os.path.join(
      save_root, task_language_label["short"].replace(" ", "_")
  )
  new_directory_path = Path(output_dir)
  new_directory_path.mkdir(parents=True, exist_ok=True)
  output_path = os.path.join(
      output_dir, f"{seq_name.replace('/', '_')}.mp4"
  )

  output_width, output_height = TACO_WIDTH, TACO_HEIGHT

  print(output_path)

  # original_fps = cap.get(cv2.CAP_PROP_FPS)
  out = cv2.VideoWriter(
      output_path,
      cv2.VideoWriter_fourcc(*"mp4v"),
      15,
      (output_width * 2, output_height)
  )

  # print(original_fps)
  frame_idx = -1
  # seq_data = []

  # print(len(seq_images))
  for i in range(len(seq_images)):
    # print(hand_pose.shape)
    # print(hand_trans.shape)
    current_trans_left = hand_trans[0:1, i:i + 1, :]
    current_trans_right = hand_trans[1:2, i:i + 1, :]

    print(hand_trans_converted.shape)
    current_trans_converted_left = hand_trans_converted[0:1, i:i + 1, :]
    current_trans_converted_right = hand_trans_converted[1:2, i:i + 1, :]

    frame_converted = cv2.resize(
        seq_images[i]["rgb_obs"].copy(),
        (TACO_WIDTH, TACO_HEIGHT)
    )

    frame = cv2.resize(
        seq_images[i]["rgb_obs"].copy(),
        (TACO_WIDTH, TACO_HEIGHT)
    )

    frame = visualize_one_step(
        frame,
        cam_pose[i],
        current_trans_left,
        current_trans_right,
        hand_pose,
        i,
        cam_intrinsics
    )

    frame_converted = visualize_one_step(
        frame_converted,
        cam_pose[i],
        current_trans_converted_left,
        current_trans_converted_right,
        hand_pose_converted,
        i,
        cam_intrinsics
    )

    out.write(np.concatenate([frame, frame_converted], axis=1))
  out.release()
  return


if __name__ == "__main__":
  dataset_root = "/mnt/data3/data/TACO"
  all_seqs = get_all_seqs(dataset_root)

  # print(all_seqs)
  # print(len(all_seqs))
  for i in range(0, len(all_seqs), 100):
    print(all_seqs[i])

    idx = i
    seq_images = parse_single_seq_image(
        dataset_root,
        all_seqs[idx],
        frame_skip=2,
        image_w=384,
        image_h=384
    )
    print(len(seq_images))

    seq_hands = parse_single_seq_hand(
        dataset_root,
        all_seqs[idx],
        frame_skip=2,
        future_len=60,
        sample_skip=5,
    )
    # for key in seq_hands[0]:
    #   print(key, seq_hands[0][key].shape)

    seq_infos = obtain_single_seq_hand(
        dataset_root,
        all_seqs[idx],
        frame_skip=2,
    )

    visualize_single_sequence(
        seq_images,
        seq_infos,
        "playground/dataset_vis/taco_visualization_noinv",
        all_seqs[idx],
    )
