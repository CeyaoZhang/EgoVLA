# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# @yunhao: deprecated comment
"""
A inference test on llava_arch.py
This test can be simply run by
python llava_arch_unit_test.py \
            --model_path path_to_model \
            --question_file path_to_question_file \
            --image_folder image_directory \
            --device "cuda:0"
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.utils import prepare_config_for_training


def build_model():
    # This test is supposed to run on a single GPU
    if torch.cuda.is_available():
        rank = 0
        torch.cuda.set_device(rank)
    model_name_or_path = "lmsys/vicuna-7b-v1.5"
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        version="v1",
        vision_tower="openai/clip-vit-large-patch14-336",
        mm_projector="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_use_im_patch_token=False,
    )
    data_args = DataArguments()
    training_args = TrainingArguments(output_dir="")
    config = LlavaLlamaConfig.from_pretrained(model_name_or_path)
    prepare_config_for_training(config, model_args, training_args, data_args)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    print("Initializing LlavaLlamaModel...")
    model = LlavaLlamaModel(config=config)
    model.vision_tower = model.vision_tower.to(device)
    model.mm_projector = model.mm_projector.to(device)
    model = model.to(device)

    return tokenizer, model, model.vision_tower.image_processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--question_file", type=str, default="tests/sample_data/llava_arch_test.json")
    parser.add_argument("--image_folder", type=str, default="tests/sample_data/llava_arch_test_images")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # model initialization
    device = args.device

    tokenizer, model, image_processor = build_model()
    vision_tower = model.get_vision_tower()
    image_size = vision_tower.config.image_size
    patch_size = vision_tower.config.patch_size
    visual_tokens_per_image = (image_size // patch_size) ** 2

    questions = json.load(open(os.path.expanduser(args.question_file)))

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        print("Checking Question: %s" % qs.split("\n")[0])
        if "image" in line:
            image_file = line["image"]
            print("Image file: %s" % image_file)
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            cur_prompt = "<image>" + "\n" + cur_prompt
        else:
            images = None

        input_ids = (
            tokenizer_image_token(cur_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.int64)
        position_ids = torch.arange(input_ids.shape[-1], device=device)

        (
            input_ids_after,
            position_ids_after,
            attention_mask_after,
            _,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images)

        if images is None:
            assert (
                position_ids_after - position_ids
            ).abs().sum() == 0, "positions_ids should not be changed, without images"
            assert (
                attention_mask_after - attention_mask
            ).abs().sum() == 0, "attention_mask should not be changed, without images"
            assert (input_ids_after - input_ids).abs().sum() == 0, "input_ids should not be changed without images"
            assert inputs_embeds is None, "inputs_embeds should be None without images"
        else:
            assert position_ids_after.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
            ), "positions_ids should not be changed, without images"
            assert attention_mask_after.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
            ), "attention_mask should not be changed, without images"
            assert input_ids_after is None, "input_ids should not be changed without images"
            assert inputs_embeds.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
                4096,
            ), "inputs_embeds should have shape (batch size, num_tokens, hidden_dim)"

        print("Checking passed.")
