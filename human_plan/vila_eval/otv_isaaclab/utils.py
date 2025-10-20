
import torch
from PIL import Image

import re
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)

from llava import conversation as conversation_lib
from llava.mm_utils import process_images_ndarray, tokenizer_image_token
import numpy as np
import os
from human_plan.dataset.vla_ioai import IOAI_RAW_IMAGE_HIEGHT, IOAI_RAW_IMAGE_WIDTH

def load_image(image_file):
  image = Image.open(image_file).convert('RGB')
  return image


_picked_index = torch.Tensor([1, 5, 10, 15, 20, 25]).long()

def get_hand_label(
  data_dict,
  future_idx=30,
  action_tokenizer=None
):

  hands = ["left_hand", "right_hand"]

  current_input_masks = []

  hand_inputs = []
  # hand_inputs_scaled = []
  for hand in hands:
    valid_mask = torch.tensor(
        data_dict["current_" + hand + "_pose"]["valid_state"]
    ).unsqueeze(-1) * torch.tensor(
        data_dict["current_" + hand + "_pose"]["track_state"]
    ).unsqueeze(-1)

    single_hand_trans = torch.tensor(
        data_dict["current_" + hand + "_pose"]["hand_trans_cam_frame"]
    ).reshape(-1, 26, 3) * valid_mask

    hand_inputs.append((
        single_hand_trans
    ).reshape(-1))

    # for key in hand_info_masks:
    current_input_masks.append(
        valid_mask.reshape(1, -1, 1)
    )

  hand_inputs = torch.concat(hand_inputs)
  # hand_inputs_scaled = torch.concat(hand_inputs_scaled)
  hand_input_masks = torch.concat(
      current_input_masks, dim=0
  )

  hand_labels = []
  hand_label_masks = []
  for hand in hands:
    # for key in hand_infos:
    # print(data_dict[
    #         "future_" + hand + "_pose"
    #     ]["hand_trans_cam_frame"].shape)
    single_hand_label_trans = torch.tensor(
        data_dict[
            "future_" + hand + "_pose"
        ]["hand_trans_cam_frame"][future_idx]
    ).reshape(-1, 26, 3)

    valid_mask = torch.tensor(
        data_dict["future_" + hand + "_pose"]["valid_state"][future_idx]
    ).unsqueeze(-1) * torch.tensor(
        data_dict["future_" + hand + "_pose"]["track_state"][future_idx]
    ).unsqueeze(-1)

    hand_labels.append((
        single_hand_label_trans * valid_mask
    ).reshape(-1))
    # for key in hand_info_masks:
    hand_label_masks.append(
        valid_mask.reshape(1, -1, 1)
    )

  # Raw Future
  hand_labels = torch.concat(hand_labels)
  

  # Filter the not valid data points
  hand_labels = torch.where(
      torch.isfinite(hand_labels),
      hand_labels, torch.tensor(0.0)
  )
  hand_inputs = torch.where(
      torch.isfinite(hand_inputs),
      hand_inputs, torch.tensor(0.0)
  )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

  hand_labels = torch.clamp(
      hand_labels, -1, 1
  )
  hand_inputs = torch.clamp(
      hand_inputs, -1, 1
  )

  # Use Diff as label
  hand_labels = hand_labels - hand_inputs

  hand_label_masks = torch.concat(
      hand_label_masks, dim=0
  )
  # 2, 26, 3
  # 256, 2, 26, 3
  hand_label_masks = torch.repeat_interleave(
      (hand_label_masks * hand_input_masks).bool(),
      3, dim=-1
  ).reshape(-1)

  picked_index = _picked_index.to(hand_inputs.device)
  # print(hand_inputs.shape)
  # print(hand_labels.shape)
  # print(hand_label_masks.shape)
  hand_inputs = hand_inputs.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
  hand_labels = hand_labels.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)
  hand_label_masks = hand_label_masks.reshape(2, 26, 3)[:, picked_index].reshape(2 * 6 * 3)

  # ee_labels = torch.Tensor(ee_labels)
  # ee_labels = torch.Tensor(ee_labels
  # ee_label_masks = torch.ones_like(ee_labels).bool()
  if action_tokenizer is not None:
    hand_ids = action_tokenizer(hand_labels, hand_label_masks)
    hand_ids = action_tokenizer.tokenizer(hand_ids)["input_ids"][2:]
    return hand_labels, hand_ids, hand_label_masks
  return hand_labels, hand_label_masks


def get_single_step_prediction(
    model, image_processor, tokenizer, action_tokenizer,
    image_ndarray, language_label, return_ids=False
):
    language_instruction = f"<image>\nWhat should the robot do to: {language_label} ? A: "

    # image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images_ndarray([image_ndarray], image_processor, model.config)
    # image_tensor = torch.stack([image_tensor], dim=0)
    # print()
    if type(image_tensor) is list:
      image_tensor = [
        image.to(model.device, dtype=torch.float16)
        for image in image_tensor
      ]
    else:
      image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = language_instruction
  
    if DEFAULT_IMAGE_TOKEN in inp:
        inp = inp.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        inp = inp.strip()
        if "mmtag" in conversation_lib.default_conversation.version:
            inp = inp.replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
    replace_token = DEFAULT_IMAGE_TOKEN
    if model.config.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    inp = inp.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    image = None

    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    # conv.append_message(conv.roles[1], None)
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    prompt = conv.get_prompt()
    # print(prompt)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    # Let's see do we need this
    input_ids = torch.cat(
        (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    )
    # print(input_ids)
    input_ids = torch.cat(
        (input_ids, torch.unsqueeze(torch.Tensor([29871] + [action_tokenizer.input_placeholder_token_idx] * 36).long(), dim=0).to(input_ids.device)), dim=1
    )
    with torch.inference_mode():
    #   output_ids = model.generate(
    #       input_ids,
    #       images=[image_tensor],
    #       # image_sizes=[image_size],
    #       do_sample=True,
    #       temperature=0.1,
    #       max_new_tokens=2 * 6 * 3,
    #       # streamer=streamer,
    #       min_new_tokens=2 * 6 * 3,
    #       use_cache=True,
    #       return_dict_in_generate=True, output_scores=True
    #   )
        output_ids = model.forward(
           input_ids,
           images=[image_tensor],
           attention_mask=input_ids.ne(tokenizer.pad_token_id).to(model.device)
        )
        action_results = output_ids["logits"].argmax(-1)[..., -38:-2].cpu().squeeze()
    
    # hand_labels = action_tokenizer(hand_labels)
    # continuous_action =  action_tokenizer.decode_token_ids_to_actions(output_ids["sequences"][:, 1:].cpu().squeeze())
    continuous_action =  action_tokenizer.decode_token_ids_to_actions(action_results)
    if return_ids:
    #   scores = torch.cat(output_ids["scores"], dim=0)
      # print(scores)
      # should be 6 * 32000
      return continuous_action, action_results, output_ids["logits"].argmax(-1)[..., -38:-2, :].cpu().numpy()
    return continuous_action
