import argparse
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import tqdm
import sys
import shutil
# from ffprobe import FFProbe


from llava.data.dataset import LazyVLAHoloAssistDataset
# from typing import Object

from transformers import HfArgumentParser, AutoTokenizer, AutoConfig, LlamaForCausalLM
# from transformers.modeling_utils import unwrap_model
# from transformers import set_seed

# from torch.utils.data import Dataset
# from llava.train.llava_trainer import LLaVATrainer
from human_plan.vila_train.args import (
  VLATrainingArguments, VLAModelArguments, VLADataArguments
)

from human_plan.utils.action_tokenizer import build_action_tokenizer

from llava.model.builder import load_pretrained_model


from llava.data import build_datasets

from llava.data.dataset import LazyVLAHoloAssistHFAbsDataset
from human_plan.utils.action_tokenizer import build_action_tokenizer

from llava.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    vision_resolution_elevation,
    unit_test_rope_scaling,
    mprint,
)

from llava.model import *
from human_plan.utils.normalization import denormalize_item

import torch

def denormalize_image(image_tensor, mean, std):
    # Ensure the mean and std are tensors and of the right shape
  mean = torch.tensor(mean).view(-1, 1, 1)
  std = torch.tensor(std).view(-1, 1, 1)

  denormalized_image = image_tensor * std + mean
  
  return denormalized_image

def project_one_hand(hand_points, img, color, img_intrinsics):

    # Put an empty camera pose for image.
  rvec = np.array([[0.0, 0.0, 0.0]])
  tvec = np.array([0.0, 0.0, 0.0])

  points, _ = cv2.projectPoints(
      hand_points, rvec, tvec, img_intrinsics, np.array([])
  )
  print("Projected Points:", points)
  img = cv2.circle(
    img, (int(round(points[0][0][0])), int(round(points[0][0][1]))),
    radius=5, color=color,
    thickness=-1
  )
  return img

def visualize_single_entry(
  data_sample, 
  idx, save_path,
  mean, std,
  dataset
):
  
  image = denormalize_image(
     data_sample["image"][-1, :, :, :].contiguous(), mean, std
  ).cpu().permute(1,2,0).numpy()
  image = (image * 255).astype(np.uint8)
  # image = image * p
  image = cv2.resize(image[:, :, ::-1], (896, 504))
  # print(data_sample["raw_action_label"])
  # print(data_sample["raw_action_mask"])
  points = data_sample["raw_action_label"][:, :3].reshape(5, 2, 3)
  masks = data_sample["raw_action_mask"].reshape(5, 2)
  cam_intrinsics = data_sample["cam_intrinsics"].reshape(3, 3)
  print("Raw W & H:", data_sample["raw_width"], data_sample["raw_height"])
  for point, mask in zip(points, masks):
     left_mask, right_mask = mask
     left_point, right_point = point
     if left_mask:
        left_point = denormalize_item(
           left_point, dataset.ee_normalization["left"]
        )
        print("Left:", left_point)
        # print()
        image = project_one_hand(
           left_point.cpu().unsqueeze(0).numpy(), image,
           (0, 0, 255), cam_intrinsics
        )
        

     if right_mask:
        right_point = denormalize_item(
           right_point, dataset.ee_normalization["right"]
        ) 
        image = project_one_hand(
           right_point.cpu().unsqueeze(0).numpy(), image, 
           (0, 255, 0), cam_intrinsics
        )

        print("Right:", right_point)

  cv2.imwrite(os.path.join(save_path, f"image_{idx}_{data_sample['language_label_short']}.jpg"), image)


def main():
  parser = argparse.ArgumentParser()
  
  parser = HfArgumentParser(
      (VLAModelArguments, VLADataArguments, VLATrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()
  training_args.run_name = training_args.output_dir.split("/")[-1]
  local_rank = training_args.local_rank

  dataset_name="holoassist_train_abs_with_dof_filtered",
  dataset_type="holoassist_hf_abs_hand",
  image_mapping_path="data/ha_dataset/hf_images_mapping.pkl"
  stats_path="data/ha_dataset/stats"
  data_path="data/ha_dataset/HoloAssist_HF_hands_filtered_multilevelv4_train"
  image_path="data/ha_dataset/HoloAssist_HF_images"
  description="HoloAssist Full Dataset in HF",
  data_skip=1

  # tokenizer, model, image_processor, context_len = load_pretrained_model(
  #     checkpoint_path, model_name, model_base,
  #     # load_4bit=False, load_8bit=False, use_flash_attn=True
  # )
  # prepare_config_for_training(config, model_args, training_args, data_args)

    ## first time training
  resume_from_checkpoint = False
  if "mpt" in model_args.model_name_or_path:
      config = AutoConfig.from_pretrained(
          model_args.model_name_or_path, trust_remote_code=True
      )
      config.attn_config["attn_impl"] = training_args.mpt_attn_impl
      model_cls = LlavaMPTForCausalLM
  elif "mistral" in model_args.model_name_or_path.lower():
      config = LlavaMistralConfig.from_pretrained(model_args.model_name_or_path)
      config._attn_implementation = "flash_attention_2"
      model_cls = LlavaMistralForCausalLM
  elif "mixtral" in model_args.model_name_or_path.lower():
      config = LlavaMixtralConfig.from_pretrained(model_args.model_name_or_path)
      config._attn_implementation = "flash_attention_2"
      model_cls = LlavaMixtralForCausalLM
  elif "gemma" in model_args.model_name_or_path.lower():
      config = LlavaGemmaConfig.from_pretrained(model_args.model_name_or_path)
      config._attn_implementation = "flash_attention_2"
      model_cls = LlavaGemmaForCausalLM
  else:
      ## llm and default multimodal model
      model_cls = LlavaLlamaModel
      config = LlavaLlamaConfig.from_pretrained(
          model_args.model_name_or_path,
          resume=resume_from_checkpoint
      )
  if getattr(config, "resume_path", None) is not None:
      config.resume_path = model_args.model_name_or_path

    ## extra configurations
  prepare_config_for_training(config, model_args, training_args, data_args)

  model = model_cls(
      config=config,
      attn_implementation="flash_attention_2",
      model_max_length=training_args.model_max_length,
      cache_dir=training_args.cache_dir,
  )
    ## extra configurations

          # image_folder = dataset.image_path
          # dataset_cls = LazyVLAHoloAssistHFAbsDataset
          # additional_kwargs["data_skip"] = getattr(dataset, "data_skip", 1)

  tokenizer = model.tokenizer
  model_args.predict_future_step = data_args.predict_future_step
  data_args.action_tokenizer = build_action_tokenizer(
      model_args.action_tokenizer, tokenizer, model_args
  )
  vision_tower = model.get_vision_tower()
  data_args.image_processor = vision_tower.image_processor

  data_args.traj_action_output_dim = model.traj_decoder.out_dim
  model.config.invalid_token_weight = training_args.invalid_token_weight

  # data_args.meta_path = getattr(dataset, "meta_path", None)
  # data_args.depth_path = getattr(dataset, "depth_path", None)
  # data_args.label_path = getattr(dataset, "label_path", None)
  # Abs dataset
  data_args.image_mapping_path = image_mapping_path
  data_args.stats_path = stats_path

  dataset = LazyVLAHoloAssistHFAbsDataset(
    data_path=data_path,
    image_folder=image_path,
    # image_mapping_path=image_mapping_path,
    # stats_path=stats_path,
    tokenizer=tokenizer,
    training_args=training_args,
    data_args=data_args,
    data_skip=1000
  )


  mean = np.array(data_args.image_processor.image_mean)  # Typically [0.485, 0.456, 0.406]
  std = np.array(data_args.image_processor.image_std)    # Typically [0.229, 0.224, 0.225]
  
  save_path = "playground/dataset_vis/holoassist_multilevelv3"
  if not os.path.exists(save_path):
     os.mkdir(save_path)
  # print(dataset[0])
  # for k, v in dataset[0].items():
  #   print(k, v.shape)
    # print(v)
  for idx, data_sample in enumerate(dataset):
    visualize_single_entry(
      data_sample, idx, save_path,
      mean, std, dataset
    )

  # visualize_single_entry(
  #    dataset[20], 20, save_path,
  #    mean, std, dataset
  # )

  # visualize_single_entry(
  #    dataset[23], 23, save_path,
  #    mean, std, dataset
  # )
  # visualize_single_entry(
  #    dataset[100], 100, save_path,
  #    mean, std, dataset
  # )
  # image = denormalize_image(
  #    dataset[0]["image"][-1, :, :, :], mean, std
  # ).cpu().permute(1,2,0).numpy()
  # image = (image * 255).astype(np.uint8)
  # # image = image * p
  # image = cv2.resize(image, (896, 504))
  # cv2.imwrite(os.path.join(save_path, "image0.jpg"), image)

  # # print(dataset[1])
  # for k, v in dataset[1].items():
  #   print(k, v.shape)
  #   print(v)

  # image = denormalize_image(
  #    dataset[1]["image"][-1, :, :, :], mean, std
  # ).cpu().permute(1,2,0).numpy()
  # image = (image * 255).astype(np.uint8)
  # image = cv2.resize(image, (896, 504))
  # cv2.imwrite(os.path.join(save_path, "image0.jpg"), image)
  # cv2.imwrite(os.path.join(save_path, "image1.jpg"), image)

if __name__ == '__main__':
  main()
