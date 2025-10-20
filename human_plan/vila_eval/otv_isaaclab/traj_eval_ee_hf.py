from llava.data.dataset import LazyVLAIOAIEEDataset

from transformers import HfArgumentParser, AutoTokenizer, AutoConfig, LlamaForCausalLM
from human_plan.vila_train.args import (
    VLATrainingArguments, VLAModelArguments, VLADataArguments
)
import numpy as np
from human_plan.utils.action_tokenizer import build_action_tokenizer

from llava.model.builder import load_pretrained_model

from llava.mm_utils import get_model_name_from_path

from llava import conversation as conversation_lib

from human_plan.vila_eval.utils.load_model import load_model_eval

import pickle

from tqdm import tqdm

import os
from PIL import Image
from io import BytesIO

from llava.mm_utils import (
 process_image_bytes
)

import torch

import datasets


from human_plan.preprocessing.preprocessing import preprocess_vla
from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla

import cv2

from scipy.spatial.transform import Rotation as R

from human_plan.preprocessing.preprocessing import (
  preprocess_vla,
  preprocess_vla_qa,
  preprocess_multimodal_vla,
  # preprocess_language_instruction
)
from human_plan.preprocessing.prompting_format import (
  preprocess_language_instruction,
  preprocess_language_instruction_qa,
)

from human_plan.vila_eval.utils.eval_func import (
  eval_single_sample,
  to_ndarray
)

from human_plan.utils.hand_dof import compute_hand_dof_5dim
from human_plan.utils.normalization import normalize_item

from human_plan.dataset_preprocessing.hoi4d.utils import (
  obtain_cam_intrinsics,
  parse_sequence_info,
  get_task_label
)

OTV_SIM_IMAGE_WIDTH = 1280
OTV_SIM_IMAGE_HEIGHT = 720


finger_tip_index = torch.Tensor([4, 8, 12, 16, 20]).long()
frame_count_scaler_up = 1
frame_count_scaler = 1
reverse_channel_order = True

raw_image_width = OTV_SIM_IMAGE_HEIGHT
raw_image_height = OTV_SIM_IMAGE_HEIGHT
hand_dof_dim = 15

# dof_normalization = np.load(os.path.join(
#     data_args.stats_path, "finger_stats.npy"
# ), allow_pickle=True).item()
# ee_normalization = np.load(os.path.join(
#     data_args.stats_path, "pos_data_stats.npy"
# ), allow_pickle=True).item()

def normalize_2d(ee_2d, sample):
  ee_2d_normalization = torch.Tensor([
    OTV_SIM_IMAGE_WIDTH, OTV_SIM_IMAGE_HEIGHT
  ]).reshape(1, 1, 2)
  ee_2d  = ee_2d / ee_2d_normalization
  return ee_2d

def get_current_language_label(sample):
  current_language_label = sample["language_label"]
  return current_language_label

def get_current_hand_data(sample, hand):
  if hand == "left":
    valid_mask = torch.zeros((1))
    single_ee_3d = torch.zeros((1, 3))
    single_ee_2d = torch.zeros((1, 2))
    single_ee_rot = torch.zeros((1, 4))
    single_handkp_3d = torch.zeros((1, 21, 3))
    return valid_mask, \
      single_ee_3d, \
      single_ee_2d, \
      single_ee_rot, \
      single_handkp_3d

  valid_mask = torch.tensor(
      sample["current_" + hand + "_flag"].reshape(-1, 1)
  )

  single_ee_3d = torch.tensor(
      sample["current_" + hand + "_wrist_3d"]
  ).reshape(-1, 3)

  single_ee_2d = torch.tensor(
      sample["current_" + hand + "_wrist_2d"]
  ).reshape(-1, 2)

  current_rot = sample["current_" + hand + "_wrist_rot"].reshape(-1, 3, 3)
  r = R.from_matrix(current_rot)
  current_rot = r.as_rotvec()
  single_hand_rot = torch.tensor(
    current_rot
  ).reshape(-1, 3)

  single_handkp_3d = torch.tensor(
    sample["current_" + hand + "_hand_kp"]
  ).reshape(-1, 21, 3)
  # future_right_hand_kp

  return valid_mask, \
    single_ee_3d, \
    single_ee_2d, \
    single_hand_rot, \
    single_handkp_3d

from human_plan.utils.hand_dof import (
  compute_hand_dof_5dim,
  convert_full_mano_to_pca_dof
)
from human_plan.utils.mano.model import (
  mano_right
)

mano_hand_mean = mano_right.hand_mean.detach().cpu().numpy()
mano_hand_components = mano_right.np_hand_components

def get_pca_dim(raw_rotation) -> torch.Tensor:
  return convert_full_mano_to_pca_dof(
    raw_rotation, mano_hand_mean, mano_hand_components
  )

def get_future_hand_data(
  data_args, sample, hand, future_step, future_idx
):
  valid_mask = torch.zeros((1, 1))

  max_len = sample["future_" + hand + "_mano_trans"].reshape(-1, 3).shape[0]

  target_idx = min(
      future_idx * (future_step + 1), max_len - 1
  )

  single_ee_3d_label = torch.tensor(
      sample[
          "future_" + hand + "_mano_trans"
      ].reshape(-1, 3)[target_idx]
  ).reshape(1, 3)

  single_ee_2d_label = torch.tensor(
      sample[
          "future_" + hand + "_mano_ee_2d"
      ].reshape(-1, 2)[target_idx]
  ).reshape(1, 2)

  single_handkp_3d_label = torch.tensor(
      sample[
          "future_" + hand + "_mano_kps3d"
      ].reshape(-1, 21, 3)[target_idx]
  ).reshape(1, 21, 3)

  assert data_args.use_mano
  hand_dof = torch.tensor(
    sample[
      "future_" + hand + "_mano_parameters"
    ].reshape(-1, 15)[target_idx]
  )
  hand_dof = torch.tensor(hand_dof)

  assert data_args.no_norm_ee_label

  valid_mask = torch.tensor(
    sample["future_" + hand + "_flag"].reshape(-1, 1)[target_idx]
  ).unsqueeze(-1)

  valid_mask = valid_mask.reshape(1, 1)

  future_rot = sample[
      "future_" + hand + "_mano_rot"
  ].reshape(-1, 3)[target_idx]

  # r = R.from_matrix(future_rot)
  # future_rot = r.as_rotvec()
  single_ee_rot_label = torch.tensor(
      future_rot
  ).reshape(1, 3)

  # single_handkp_3d_label = torch.tensor(
  #     sample["future_"+ hand + "_mano_kps3d"].reshape(-1, 21, 3)[target_idx]
  # ).reshape(1, 21, 3)
  
  return valid_mask, \
    single_ee_3d_label, \
    single_ee_2d_label, \
    hand_dof, \
    single_ee_rot_label, \
    single_handkp_3d_label


def process_hand_label_masks(raw_hand_label_masks):
  dof_hand_label_masks = torch.repeat_interleave(
      raw_hand_label_masks.bool().squeeze(-1),
      hand_dof_dim, dim=-1
  ).reshape(-1, 2, hand_dof_dim)

  ee_hand_label_masks = torch.repeat_interleave(
      raw_hand_label_masks.bool(),
      3, dim=-1
  ).reshape(-1, 2, 3)

  rot_hand_label_masks = torch.repeat_interleave(
      raw_hand_label_masks.bool(),
      3, dim=-1
  ).reshape(-1, 2, 3)

  ee_2d_hand_label_masks = ee_hand_label_masks[..., :2]
  return ee_hand_label_masks, ee_2d_hand_label_masks, dof_hand_label_masks, rot_hand_label_masks

def obtain_data(
  sample,
  image_mapping_dict,
  img_dataset,
  ee_normalization,
  dof_normalization,
  # data_item,
  data_args,
  action_tokenizer,
  tokenizer
):

    to_ndarray(sample)

    image_list = []
    valid_his_len = 0
    seq_name = ''.join(sample["seq_name"])
    try:
      current_image_file_index = image_mapping_dict[
        seq_name
      ][sample["frame_count"]]
    except Exception:
      print(seq_name, sample["frame_count"])

    raw_rgb_obs_sequence = []

    assert data_args.add_his_img_skip % frame_count_scaler == 0
    for i in range(data_args.add_his_obs_step):
      # Need to fxxking align with dataset
      query_idx = max(
        0,
        sample["frame_count"] - (i + 1) * data_args.add_his_img_skip * frame_count_scaler_up // frame_count_scaler
      )
      print("Query Index:", query_idx)
      if not query_idx in image_mapping_dict[seq_name]:
          continue
      c_his_image_file_index = image_mapping_dict[
          seq_name
      ][query_idx]
      image_list.append(process_image_bytes(
          img_dataset[c_his_image_file_index]["rgb_obs"],
          data_args,
          reverse_channel_order=reverse_channel_order
      ))
      raw_rgb_obs_sequence.append(img_dataset[c_his_image_file_index]["rgb_obs"])
      valid_his_len += 1

    print("Current Query Index:", sample["frame_count"])
    image_list.append(process_image_bytes(
        img_dataset[current_image_file_index]["rgb_obs"],
        data_args,
        reverse_channel_order=reverse_channel_order
    ))

    raw_rgb_obs_sequence.append(img_dataset[current_image_file_index]["rgb_obs"])
    image = torch.stack(image_list, dim=0)

    future_idx = data_args.future_index

    hands = ["left", "right"]

    current_input_masks = []

    # hand_inputs_3d = []
    # hand_inputs_2d = []
    # hand_inputs_pose = []
    # for hand in hands:
    #   valid_mask = torch.tensor(
    #       sample["current_" + hand + "_flag"].reshape(-1, 1)
    #   ).unsqueeze(-1)

    #   single_hand_trans = torch.tensor(
    #       sample["current_" + hand + "_wrist_3d"]
    #   ).reshape(-1, 3)

    #   single_hand_2d = torch.tensor(
    #       sample["current_" + hand + "_wrist_2d"]
    #   )

    #   # single_hand_pose = torch.tensor(
    #   #     sample["current_" + hand + "_wrist_2d"]
    #   # )

    #   hand_inputs_3d.append((
    #       single_hand_trans * valid_mask
    #   ).reshape(-1))

    #   hand_inputs_2d.append((
    #       single_hand_2d * valid_mask
    #   ).reshape(-1))

    #   # hand_inputs_pose.append((
    #   #   single_hand_pose * valid_mask
    #   # ))

    #   # for key in hand_info_masks:
    #   current_input_masks.append(
    #       valid_mask.reshape(1, 1)
    #   )

    # # hand_inputs = torch.concat(hand_inputs)

    # hand_inputs_3d = torch.concat(hand_inputs_3d)
    # hand_inputs_2d = torch.concat(hand_inputs_2d)

    future_idx = data_args.future_index

    ee_2d_labels = []
    ee_3d_labels = []
    ee_rot_labels = []

    dof_hand_labels = []
    handkp_3d_labels = []

    raw_hand_label_masks = []

    #[TODO] fix translation with head pose.
    for future_step in range(data_args.predict_future_step):
      for hand in hands:
        # Do not check length for now -> make the future length consistent
        future_hand_data = get_future_hand_data(
          data_args, sample, hand, future_step, future_idx
        )

        valid_mask, future_ee_3d, future_ee_2d, \
          future_hand_pose, future_ee_rot, future_handkp_3d = future_hand_data

        ee_3d_labels.append((
           future_ee_3d * valid_mask
        ).reshape(1, 3))

        ee_2d_labels.append((
            future_ee_2d * valid_mask
        ).reshape(1, 2))

        if data_args.use_mano:
          dof_hand_labels.append((
              future_hand_pose * valid_mask
          ).reshape(1, 15))
        else:
          dof_hand_labels.append((
              future_hand_pose * valid_mask
          ).reshape(1, 5))
        # for key in hand_info_masks:

        # Use RotVec
        ee_rot_labels.append((
            future_ee_rot * valid_mask
        ).reshape(1, 3))

        # Fxxk it there is something wrong with the orders fxxk it
        # handkp_3d_labels.append((
        #     future_handkp_3d * valid_mask
        # ).reshape(1, 21, 3))

        raw_hand_label_masks.append(
            valid_mask.reshape(1, -1)
        )

    # Raw Future
    ee_3d_labels = torch.concat(ee_3d_labels, dim=0).reshape(-1, 2, 3)
    ee_2d_labels = torch.concat(ee_2d_labels, dim=0).reshape(-1, 2, 2)
    ee_rot_labels = torch.concat(ee_rot_labels, dim=0).reshape(-1, 2, 3)
    print(ee_3d_labels.shape, ee_2d_labels.shape)
    # Fxxk it there is something wrong with the orders fxxk it
    # handkp_3d_labels = torch.concat(handkp_3d_labels, dim=0).reshape(-1, 2, 21, 3)

    ee_2d_labels = normalize_2d(
      ee_2d_labels, sample
    )
    ee_2d_labels = ee_2d_labels.clamp(0, 1)

    dof_hand_labels = torch.concat(
      dof_hand_labels, dim=0
    ).reshape(-1, 2, hand_dof_dim)

    # Filter the not valid data points
    ee_2d_labels = torch.where(
        torch.isfinite(ee_2d_labels),
        ee_2d_labels, torch.tensor(0.0)
    )
    ee_rot_labels = torch.where(
        torch.isfinite(ee_rot_labels),
        ee_rot_labels, torch.tensor(0.0)
    )
    ee_3d_labels = torch.where(
        torch.isfinite(ee_3d_labels),
        ee_3d_labels, torch.tensor(0.0)
    )
    dof_hand_labels = torch.where(
        torch.isfinite(dof_hand_labels),
        dof_hand_labels, torch.tensor(0.0)
    )

    # Fxxk it there is something wrong with the orders fxxk it
    # handkp_3d_labels = torch.where(
    #     torch.isfinite(handkp_3d_labels),
    #     handkp_3d_labels, torch.tensor(0.0)
    # )

    # if data_args.use_relative_label:
    #   ee_2d_labels = ee_2d_labels - ee_2d_inputs
    #   ee_3d_labels = ee_3d_labels - ee_3d_inputs

    # -1, 1
    raw_hand_label_masks = torch.concat(
        raw_hand_label_masks, dim=0
    )

    # ee_3d_label_masks, ee_2d_label_masks, \
    #   dof_hand_label_masks, ee_rot_label_masks \
    #   handkp_3d_label_masks = process_hand_label_masks(
    #   raw_hand_label_masks
    # )

    ee_3d_label_masks, ee_2d_label_masks, \
      dof_hand_label_masks, ee_rot_label_masks  = process_hand_label_masks(
      raw_hand_label_masks
    )

    if data_args.merge_hand:
      ee_2d_labels = ee_2d_labels.reshape(-1, 4)
      ee_3d_labels = ee_3d_labels.reshape(-1, 6)
      ee_rot_labels = ee_rot_labels.reshape(-1, 6)

      dof_hand_labels = dof_hand_labels.reshape(-1, 2 * hand_dof_dim)

      ee_2d_label_masks = ee_2d_label_masks.reshape(-1, 4)
      ee_3d_label_masks = ee_3d_label_masks.reshape(-1, 6)
      ee_rot_label_masks = ee_rot_label_masks.reshape(-1, 6)
      dof_hand_label_masks = dof_hand_label_masks.reshape(-1, 2 * hand_dof_dim)

    label_list = []
    mask_list = []
    if data_args.include_2d_label:
      label_list.append(ee_2d_labels)
      mask_list.append(ee_2d_label_masks)
  
    label_list += [ee_3d_labels, dof_hand_labels]
    mask_list += [ee_3d_label_masks, dof_hand_label_masks]

    if data_args.include_rot_label:
      label_list.append(ee_rot_labels)
      mask_list.append(ee_rot_label_masks)

    if data_args.include_handkp:
      raise NotImplementedError
      label_list.append(handkp_3d_labels)
      # mask_list.append(handkp_3d_label_masks)
    # for m in mask_list:
    #    print(m.shape)
    hand_labels = torch.cat(label_list, dim=-1).reshape(-1)
    hand_label_masks = torch.cat(mask_list, dim=-1).reshape(-1)

    current_language_label = get_current_language_label(sample)

    language_instruction = preprocess_language_instruction(
      current_language_label, valid_his_len, data_args
    )
    print(language_instruction)

    language_instruction = preprocess_multimodal_vla(
      language_instruction,
      data_args
    )
    print(language_instruction)

    data_dict = preprocess_vla(
      language_instruction,
      hand_labels,
      hand_label_masks,
      action_tokenizer,
      tokenizer,
      mask_input=data_args.mask_input,
      mask_ignore=data_args.mask_ignore,
      raw_action_label=data_args.raw_action_label,
      traj_action_output_dim=data_args.traj_action_output_dim,
      input_placeholder_diff_index=data_args.input_placeholder_diff_index,
      language_response=None,
      include_response=data_args.include_response,
      include_repeat_instruction=data_args.include_repeat_instruction,
      raw_language_label=current_language_label
    )

    print(data_dict["input_ids"])
    print(data_dict["labels"])
    # print("Hand label:", hand_labels)
    # print("Hand lbael masks:", hand_label_masks)
    data_dict["image"] = image
    data_dict["image"] = image
    data_dict["raw_rgb_his"] = raw_rgb_obs_sequence
    # data_dict["raw_2d"] = ee_2d_hand_labels
    # data_dict["input_2d"] = hand_inputs_2d
    data_dict["raw_width"] = sample["raw_width"]
    data_dict["raw_height"] = sample["raw_height"]
    data_dict["seq_name"] = sample["seq_name"]
    data_dict["frame_count"] = sample["frame_count"]
    # data_dict["kp_2d"] = sample["current_kp_2d"]

    data_dict["raw_image_obs"] = Image.open(
      BytesIO(img_dataset[current_image_file_index]["rgb_obs"])
    )
    data_dict["raw_image_obs"] = np.array(data_dict["raw_image_obs"])
    data_dict["raw_image_obs"] = data_dict["raw_image_obs"]
    data_dict["raw_image_obs"] = cv2.resize(
      data_dict["raw_image_obs"], 
      (sample["raw_width"], sample["raw_height"])
    )
    return data_dict

import datasets

from human_plan.utils.visualization import (
  project_points
)

def main():
  import numpy as np

  # data_path = "data/eval_epic_kitchen_filtered_raw.pkl"
  # label_path = "data/EPIC_KITCHENS_HOI_LABEL"
  # image_path = "data/EPIC_KITCHEN_HF_images"
  # image_mapping_path = "data/epic_kitchen_hf_images_mapping.pkl"
  image_mapping_path="data/otv_isaaclab_hf_v3/hf_images_mapping.pkl"
  stats_path="data/hoi4d_hf/stats"
  data_path="data/otv_isaaclab_hf_v3/HF_hand_V1_train"
  image_path="data/otv_isaaclab_hf_v3/HF_images"
  
  save_path = "playground/otv_sim_eval_mano_fix_fps_his_v3"
  parser = HfArgumentParser(
      (VLAModelArguments, VLADataArguments, VLATrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  model, tokenizer, model_args, data_args, training_args = load_model_eval(
    model_args, data_args, training_args
  )


  dataset = datasets.load_from_disk(data_path)

  img_dataset = datasets.load_from_disk(image_path)

  with open(image_mapping_path, "rb") as f:            
    image_mapping_dict = pickle.load(f)

  dof_normalization = np.load(os.path.join(
      stats_path, "finger_stats.npy"
  ), allow_pickle=True).item()
  ee_normalization = np.load(os.path.join(
      stats_path, "pos_data_stats.npy"
  ), allow_pickle=True).item()

  for hand in ["right", "full"]:
    for item in ["lower_bound", "upper_bound", "mean", "std"]:
      dof_normalization[hand][item] = torch.Tensor(dof_normalization[hand][item])
      ee_normalization[hand][item] = torch.Tensor(ee_normalization[hand][item])
  

  if not os.path.exists(save_path):
    os.mkdir(save_path)
  loss_list = []
  prediction_dist_list = []

  data_skip = 2
  total_samples = 10000
  data_idxes = list(range(len(dataset)))[:total_samples:data_skip]

  for data_idx in tqdm(data_idxes):
    
    sample = dataset[data_idx]
    seq_name = "".join(sample["seq_name"])

    seq_info = parse_sequence_info(seq_name)
    cam_intrinsics = np.array([
      [488.6662,   0.0000, 640.0000],
      [  0.0000, 488.6662, 360.0000],
      [  0.0000,   0.0000,   1.0000]
    ])


    raw_data_dict = obtain_data(
      sample,
      image_mapping_dict,
      img_dataset,
      ee_normalization,
      dof_normalization,
      data_args,
      data_args.action_tokenizer,
      tokenizer
    )
    results = eval_single_sample(
      raw_data_dict,
      tokenizer, model,
      image_width=raw_data_dict["raw_width"],
      image_height=raw_data_dict["raw_height"]
    )
    # print(unnormalized_pred)
    pred, result_img, action_labels, action_masks, loss = results
    # prediction_dist_list.append(prediction_dist)
    image_shape_normalization = np.array([[raw_data_dict["raw_width"], raw_data_dict["raw_height"]]]).reshape(1, 1, 2)
    print(pred.shape)
    unnormalized_pred = pred[:, :4].reshape(-1, 2, 2) * image_shape_normalization

    unnormalized_action_labels = action_labels[:, :4].reshape(-1, 2, 2) * image_shape_normalization

    pred_3d = pred[:, 4:10].reshape(-1, 2, 3)
    proj_2d = project_points(
      pred_3d, cam_intrinsics
    )
    proj_2d = proj_2d.reshape(-1, 2, 2)


    label_3d = action_labels[:, 4:10].reshape(-1, 2, 3)
    label_proj_2d = project_points(
      label_3d, cam_intrinsics
    )
    label_proj_2d = label_proj_2d.reshape(-1, 2, 2)

    # print("Data Idx:", data_idx, loss, unnormalized_pred.shape, action_masks.shape)
    loss_list.append(loss)
    # result_img = cv
    # result_img = cv2.imread(os.path.join(image_path, image_file_path))
    action_masks = action_masks[:, :4].reshape(-1, 2, 2)
    result_img = result_img[:, :, ::-1]
    result_img_2d = result_img.copy()
    result_img_3d = result_img.copy()
    for i in range(unnormalized_pred.shape[0]):
      print(action_masks.shape)
      for j in range(2):
        if action_masks[i, j, 0]:
          result_img_2d = cv2.circle(
            result_img_2d, 
            (int(unnormalized_pred[i, j, 0]),int(unnormalized_pred[i, j, 1])),
            5, (0, 255, 0), thickness=-1
          )
          if i < unnormalized_pred.shape[0] - 1:
            result_img_2d = cv2.line(
              result_img_2d, 
              (int(unnormalized_pred[i, j, 0]),int(unnormalized_pred[i, j, 1])),
              (int(unnormalized_pred[i + 1, j, 0]),int(unnormalized_pred[i + 1, j, 1])),
              (0, 255, 0), thickness=2
            ) 
          result_img_2d = cv2.circle(
            result_img_2d, 
            (int(unnormalized_action_labels[i, j, 0]),int(unnormalized_action_labels[i, j, 1])),
            5, (0, 0, 255), thickness=-1
          )
          if i < unnormalized_pred.shape[0] - 1:
            result_img_2d = cv2.line(
              result_img_2d, 
            (int(unnormalized_action_labels[i, j, 0]),int(unnormalized_action_labels[i, j, 1])),
            (int(unnormalized_action_labels[i + 1, j, 0]),int(unnormalized_action_labels[i + 1, j, 1])),
              (0, 0, 255), thickness=2
            )

          result_img_3d = cv2.circle(
            result_img_3d, 
            (int(proj_2d[i, j, 0]),int(proj_2d[i, j, 1])),
            5, (0, 255, 0), thickness=-1
          )
          if i < proj_2d.shape[0] - 1:
            result_img_3d = cv2.line(
              result_img_3d, 
            (int(proj_2d[i, j, 0]),int(proj_2d[i, j, 1])),
            (int(proj_2d[i + 1, j, 0]),int(proj_2d[i + 1, j, 1])),
              (0, 255, 0), thickness=2
            )

          result_img_3d = cv2.circle(
            result_img_3d, 
            (int(label_proj_2d[i, j, 0]),int(label_proj_2d[i, j, 1])),
            5, (0, 0, 255), thickness=-1
          )
          if i < proj_2d.shape[0] - 1:
            result_img_3d = cv2.line(
              result_img_3d, 
            (int(label_proj_2d[i, j, 0]),int(label_proj_2d[i, j, 1])),
            (int(label_proj_2d[i + 1, j, 0]),int(label_proj_2d[i + 1, j, 1])),
              (0, 0, 255), thickness=2
            )
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.image as mpimg
    # Example 5D vector time series data
    # time = np.linspace(0, 10, 100)  # Time variable
    time = np.arange(unnormalized_pred.shape[0])
    # vector = np.random.rand(5, 100)  # Example 5D data with 100 points for each dimension

    # Create a figure and subplots: 1 column for image, 5 rows for the 5 dimensions of the vector
    # fig, axs = plt.subplots(5, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [1, 2]})

    # fig = plt.figure(figsize=(20, 10))
    # gs = fig.add_gridspec(5, 2, width_ratios=[5, 2])

    # # Create the large image plot that spans all rows
    # ax_img = fig.add_subplot(gs[:, 0])
    # ax_img.imshow(result_img[..., ::-1])
    # ax_img.axis('off')  # Hide axis for the image

    # finger_pred = pred[:, 10:].reshape(-1, 2, 15)
    # finger_label = action_labels[:, 10:].reshape(-1, 2, 15)

    # # Adjust layout
    # plt.tight_layout()
    # Adjust layout
    # plt.tight_layout()

    # Save the plot to disk (change the filename and format as needed)
    # plt.savefig(os.path.join(save_path, f"{data_idx}_with_finger.jpeg"), dpi=300)
    # seq_name = "".join(sample["seq_name"]).replace("/", "_")
    seq_save_path = os.path.join(
      save_path,
      seq_name.replace("/", "_")#+"_"+get_task_label(seq_info)["verb"]
    )
    if not os.path.exists(seq_save_path):
      os.mkdir(seq_save_path)
    # print(os.path.join(save_path, f"{data_idx}_{data['language_label']}.jpeg"))
    cv2.imwrite(
      # os.path.join(save_path, f"{data_idx}_{raw_data_dict['language_label_short']}.jpeg"),
      os.path.join(seq_save_path, f"{sample['frame_count']}.jpeg"),
      np.concatenate([result_img_2d, result_img_3d], axis=1)
    )
    raw_rgb_his = raw_data_dict["raw_rgb_his"]
    raw_rgb_his = [np.array(Image.open(BytesIO(rgb_ob))) for rgb_ob in raw_rgb_his]
    # print(np.array(Image.open(BytesIO(raw_rgb_his[0]))).shape)
    # exit()
    cv2.imwrite(
      # os.path.join(save_path, f"{data_idx}_{raw_data_dict['language_label_short']}.jpeg"),
      os.path.join(seq_save_path, f"history_{sample['frame_count']}.jpeg"),
      np.concatenate(raw_rgb_his, axis=0)
    )

  loss_all = np.mean(loss_list)
  print("Average Loss:", loss_all)
  # print("Average Prediction Dist:", np.mean(prediction_dist_list)) 

if __name__ == "__main__":
  main()
