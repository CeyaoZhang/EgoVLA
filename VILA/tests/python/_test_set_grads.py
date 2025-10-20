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

import copy
import unittest

import torch
from transformers import AutoTokenizer

from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.train.args import ModelArguments, TrainingArguments
from llava.train.utils import prepare_config_for_training

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"


class TestSetGrads(unittest.TestCase):
    def setUp(self):
        # This test is supposed to run on a single GPU
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)
        torch.set_default_dtype(torch.bfloat16)

        self.model_args = ModelArguments(
            model_name_or_path="lmsys/vicuna-7b-v1.5",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector="mlp2x_gelu",
        )

        self.training_args = TrainingArguments(
            tune_language_model=False, tune_vision_tower=False, tune_mm_projector=False, output_dir=None, bf16=True
        )

    def single_forward_backward(self):
        print("Preprocessing inputs...")
        data = copy.deepcopy(self.data)
        data["input_ids"] = data["input_ids"].to(device)
        data["images"] = data["images"].to(torch.bfloat16).to(device)
        data["attention_mask"] = data["attention_mask"].to(device)
        data["labels"] = data["labels"].to(device)
        data["position_ids"] = None
        data["past_key_values"] = None

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(**data)

        print("Packing inputs...")
        (
            _,
            new_position_ids,
            new_attention_mask,
            _,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        ) = self.model.repack_multimodal_data(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

        print("Running models...")

        # forward results with input packing
        outputs = self.model.llm.forward(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=None,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        loss = outputs.loss
        loss.backward()

    def build_vila_model(self):
        # first time training
        config = LlavaLlamaConfig.from_pretrained(self.model_args.model_name_or_path)
        model_cls = LlavaLlamaModel

        print("Initializing data...")
        data = torch.load("tests/sample_data/test_packing.pth")
        self.data = data
        print("Initializing model...")
        prepare_config_for_training(config, self.model_args, self.training_args)
        self.model = model_cls(config=config).to(device)
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        # necessary for model forward
        self.model.llm.pad_token_id = self.tokenizer.pad_token_id
        self.model.get_llm().requires_grad_(self.training_args.tune_language_model)
        self.model.get_vision_tower().requires_grad_(self.training_args.tune_vision_tower)
        self.model.get_mm_projector().requires_grad_(self.training_args.tune_mm_projector)
        # save the loaded weights
        self.loaded_weights_dict = {
            name: copy.deepcopy(param.detach().cpu().data) for name, param in self.model.named_parameters()
        }

    def verify_grads_state(self, tune_language_model, tune_vision_tower, tune_mm_projector):
        print("Checking gradients state...")

        for name, param in self.model.named_parameters():
            if "vision_tower" in name:
                if "post_layernorm" in name:
                    continue
                if tune_vision_tower:
                    # some position embeddings are not trained
                    self.assertEqual(param.grad is not None and (param.grad != 0).any(), True)
                else:
                    self.assertEqual(param.grad is not None, False)

            elif "mm_projector" in name:
                if tune_mm_projector:
                    self.assertEqual(param.grad is not None and (param.grad != 0).all(), True)
                else:
                    self.assertEqual(param.grad is not None, False)
            else:
                if tune_language_model:
                    # some tokens are not trained
                    self.assertEqual(param.grad is not None and (param.grad != 0).any(), True)
                else:
                    self.assertEqual(param.grad is not None, False)

    def verify_weights_state(self, tune_language_model, tune_vision_tower, tune_mm_projector):
        print("Checking weights state...")

        for name in self.updated_weights_dict.keys():
            if "vision_tower" in name:
                if "post_layernorm" in name:
                    continue
                self.assertEqual(
                    (self.updated_weights_dict[name] != self.loaded_weights_dict[name]).any(), tune_vision_tower
                )
            elif "mm_projector" in name:
                self.assertEqual(
                    (self.updated_weights_dict[name] != self.loaded_weights_dict[name]).any(), tune_mm_projector
                )
            else:
                if self.updated_weights_dict[name].numel() > 1:
                    self.assertEqual(
                        (self.updated_weights_dict[name] != self.loaded_weights_dict[name]).any(), tune_language_model
                    )

    def test_tune_projector_and_language_model(self):
        print("Testing tune projector and language model ...")
        self.training_args.tune_mm_projector = True
        self.training_args.tune_language_model = True
        self.build_vila_model()
        self.single_forward_backward()

        self.verify_grads_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )
        self.updated_weights_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param = param - 100 * param.grad
            self.updated_weights_dict[name] = copy.deepcopy(param.detach().cpu().data)
        self.verify_weights_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    def test_tune_projector_and_vision_tower(self):
        print("Testing tune projector and vision tower ...")
        self.training_args.tune_mm_projector = True
        self.training_args.tune_vision_tower = True
        self.build_vila_model()
        self.single_forward_backward()

        self.verify_grads_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )
        self.updated_weights_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param = param - 100 * param.grad
            self.updated_weights_dict[name] = copy.deepcopy(param.detach().cpu().data)
        self.verify_weights_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    def test_tune_projector(self):
        print("Testing tune projector ...")
        self.training_args.tune_mm_projector = True
        self.build_vila_model()
        self.single_forward_backward()

        self.verify_grads_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )
        self.updated_weights_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param = param - 100 * param.grad
            self.updated_weights_dict[name] = copy.deepcopy(param.detach().cpu().data)
        self.verify_weights_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    def test_tune_vision_tower(self):
        print("Testing tune vision tower ...")
        self.training_args.tune_vision_tower = True
        self.build_vila_model()
        self.single_forward_backward()

        self.verify_grads_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )
        self.updated_weights_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param = param - 100 * param.grad
            self.updated_weights_dict[name] = copy.deepcopy(param.detach().cpu().data)
        self.verify_weights_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    def test_tune_language_model(self):
        print("Testing tune language model ...")
        self.training_args.tune_language_model = True
        self.build_vila_model()
        self.single_forward_backward()

        self.verify_grads_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )
        self.updated_weights_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param = param - 100 * param.grad
            self.updated_weights_dict[name] = copy.deepcopy(param.detach().cpu().data)
        self.verify_weights_state(
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )


if __name__ == "__main__":
    unittest.main()
