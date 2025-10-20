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

import os
import random
import shutil
import unittest
from collections import OrderedDict

import shortuuid
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig

from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.train.args import ModelArguments, TrainingArguments
from llava.train.utils import get_checkpoint_path, prepare_config_for_training

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"


class TestModelInitialization(unittest.TestCase):
    def setUp(self):
        # This test is supposed to run on a single GPU
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)
        random.seed(42)

        self.model_args = ModelArguments(
            model_name_or_path="lmsys/vicuna-7b-v1.5",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector="mlp2x_gelu",
        )

        self.training_args = TrainingArguments(output_dir="")

        self.random_path = "/tmp/test-loading-" + shortuuid.uuid()
        print(f"Using temporary directory: {self.random_path} ...")
        if os.path.exists(self.random_path):
            shutil.rmtree(self.random_path)

    def fast_normal_initialized(self, model, std=0.02, num_reinitialize=20):
        num_reinitialized = 0
        modules = [module for module in model.modules()]
        random.shuffle(modules)
        for module in tqdm(modules, total=len(modules)):
            num_reinitialized += 1
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                # if module.bias is not None:
                #     module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
            else:
                num_reinitialized -= 1
            if num_reinitialized == num_reinitialize:
                break

    def build_vila_model(self, resume_path=""):
        resume_path = get_checkpoint_path(resume_path)
        if resume_path:
            config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
            config.resume_path = resume_path
            model_cls = eval(config.architectures[0])
        else:
            ## first time training
            config = LlavaLlamaConfig.from_pretrained(self.model_args.model_name_or_path)
            model_cls = LlavaLlamaModel

        prepare_config_for_training(config, self.model_args, self.training_args)
        model = model_cls(config=config)
        return model

    def test_first_build(self):
        """
        Build model from scratch.
        """
        model = self.build_vila_model(resume_path=self.random_path)
        ## load pretrained model
        from transformers import CLIPVisionModel, LlamaForCausalLM

        language_model = LlamaForCausalLM.from_pretrained(
            self.model_args.model_name_or_path, torch_dtype=eval(model.config.model_dtype)
        )
        vision_tower = CLIPVisionModel.from_pretrained(
            self.model_args.vision_tower, torch_dtype=eval(model.config.model_dtype)
        )

        first_loading_params = {param_name: param for param_name, param in model.named_parameters()}
        pretrained_params = {
            "vision_tower.vision_tower." + param_name: param for param_name, param in vision_tower.named_parameters()
        }
        pretrained_params.update(
            {"llm." + param_name: param for param_name, param in language_model.named_parameters()}
        )

        for k, v in pretrained_params.items():
            if k in first_loading_params.keys():
                self.assertAlmostEqual(torch.equal(v.data, first_loading_params[k].data), True)

    def test_resume(self):
        """
        Resume the whole model from checkpoints.
        """

        model = self.build_vila_model()
        self.fast_normal_initialized(model)
        print("Saving random initialized moodel ...")

        state_dict = model.state_dict()
        if model.get_llm():
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            model.llm.generation_config.do_sample = True
            model.llm.save_pretrained(os.path.join(self.random_path, "llm"), state_dict=llm_state_dict)
            model.config.llm_cfg = model.llm.config

        if model.get_vision_tower():
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            model.vision_tower.vision_tower.save_pretrained(
                os.path.join(self.random_path, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            model.vision_tower.image_processor.save_pretrained(os.path.join(self.random_path, "vision_tower"))
            model.config.vision_tower_cfg = model.vision_tower.config

        if model.get_mm_projector():
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            model.mm_projector.save_pretrained(
                os.path.join(self.random_path, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            model.config.mm_projector_cfg = model.mm_projector.config
        # update and save top-level config
        model.config.architectures = [model.__class__.__name__]
        model.config.save_pretrained(self.random_path)
        # load again
        loaded_model = self.build_vila_model(self.random_path)
        saved_state_dict = {param_name: param for param_name, param in model.named_parameters()}
        loaded_state_dict = {param_name: param for param_name, param in loaded_model.named_parameters()}
        for k, v in saved_state_dict.items():
            self.assertEqual(torch.equal(v.data, loaded_state_dict[k].data), True)
        shutil.rmtree(self.random_path)


if __name__ == "__main__":
    unittest.main()
