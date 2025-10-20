# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import unittest

from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel

from llava.model.language_model.llava_llama import LlavaLlamaConfig


def check_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 10e5


hf_repo = "Efficient-Large-Model/CI-new-format-llama7b-siglip"


class TestModelLoadingAndSaving(unittest.TestCase):
    def test_load_from_config(self):
        global hf_repo
        cpath = hf_hub_download(repo_id=hf_repo, filename="config.json")

        # testing loading from config
        # TODO(ligeng): why LlavaLlamaConfig(config_path=cpath) is different btw LlavaLlamaConfig.from_pretrained(cpath)
        LlavaLlamaConfig.from_pretrained(cpath)
        LlavaLlamaConfig.from_pretrained(hf_repo)
        config = AutoConfig.from_pretrained(hf_repo)
        model = AutoModel.from_config(config)
        check_params(model)

    def test_from_config(self):
        # Model from /home/yunhaof/workspace/scripts/ckpts/vila/debug/reproduce/scratch_stable_test1/stage3
        # fpath = "Efficient-Large-Model/CI-format-7b-v2"
        global hf_repo
        config = AutoConfig.from_pretrained(hf_repo)
        model = AutoModel.from_config(config)
        check_params(model)

    def test_from_pretrained(self):
        # fpath = "Efficient-Large-Model/CI-format-7b-v2"
        global hf_repo
        model = AutoModel.from_pretrained(hf_repo)
        check_params(model)

    def test_save_and_reload(self):
        # fpath = "Efficient-Large-Model/CI-format-7b-v2"
        global hf_repo
        model = AutoModel.from_pretrained(hf_repo)
        shutil.rmtree("checkpoints/tmp", ignore_errors=True)
        model.save_pretrained("checkpoints/tmp")
        model = AutoModel.from_pretrained("checkpoints/tmp")
        check_params(model)
        shutil.rmtree("checkpoints/tmp", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
