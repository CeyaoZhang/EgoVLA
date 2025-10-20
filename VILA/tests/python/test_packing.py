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

import unittest

import torch
from parameterized import parameterized

from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.utils import prepare_config_for_training

MODELS = ["lmsys/vicuna-7b-v1.5", "Qwen/Qwen2-7B"]


class TestInputPacking(unittest.TestCase):
    @parameterized.expand(MODELS)
    def test_loss_close(self, model_name_or_path: str) -> None:
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            version="v1",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector="mlp2x_gelu",
            mm_vision_select_layer=-2,
            mm_use_im_patch_token=False,
        )
        data_args = DataArguments()
        training_args = TrainingArguments(output_dir="", bf16=True)

        config = LlavaLlamaConfig.from_pretrained(model_name_or_path)
        prepare_config_for_training(config, model_args, training_args, data_args)

        model = LlavaLlamaModel(config=config, attn_implementation="flash_attention_2")
        model = model.cuda()
        model.train()

        data = torch.load("tests/sample_data/test_packing.pth")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32:
                    v = v.to(torch.bfloat16 if training_args.bf16 else torch.float16)
                v = v.cuda()
            data[k] = v

        losses = {}
        for packing in [False, True]:
            outputs = model(**data, packing=packing, use_cache=False)
            losses[packing] = outputs.loss.item()
        torch.testing.assert_close(losses[True], losses[False], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
