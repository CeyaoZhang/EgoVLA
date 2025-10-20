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

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
from llava.data.dataset import preprocess_multimodal
from llava.train.args import ModelArguments

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"


class TestPreprocessMultimodal(unittest.TestCase):
    def setUp(self):
        # This test is supposed to run on a single GPU
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)
        torch.set_default_dtype(torch.bfloat16)
        self.data_args = ModelArguments()
        self.data_args.is_multimodal = True

    def test_single_image_data_with_placeholder(self):
        print("Testing single image conversation with <image> placeholder ...")
        sources = [
            [
                {"from": "human", "value": "Random nonsense question <image>?"},
                {"from": "gpt", "value": "Random nonsense answer..."},
            ]
        ]
        sources = preprocess_multimodal(sources, self.data_args)
        self.assertEqual(sources[0][0]["value"], f"Random nonsense question {DEFAULT_IMAGE_TOKEN}\n?")

    def test_single_image_data_without_placeholder(self):
        print("Testing single image conversation without <image> placeholder ...")
        sources = [
            [
                {"from": "human", "value": "Random nonsense question?"},
                {"from": "gpt", "value": "Random nonsense answer..."},
            ]
        ]
        sources = preprocess_multimodal(sources, self.data_args)
        self.assertEqual(sources[0][0]["value"], f"{DEFAULT_IMAGE_TOKEN}\nRandom nonsense question?")

    def test_single_image_data_with_placeholder_with_im_start_end(self):
        print("Testing single image conversation with <image> placeholder and mm_use_im_start_end=True ...")
        sources = [
            [
                {"from": "human", "value": "Random nonsense question <image>?"},
                {"from": "gpt", "value": "Random nonsense answer..."},
            ]
        ]
        self.data_args.mm_use_im_start_end = True
        sources = preprocess_multimodal(sources, self.data_args)
        self.assertEqual(
            sources[0][0]["value"],
            f"Random nonsense question {DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_IM_END_TOKEN}\n?",
        )

    def test_single_image_data_without_placeholder_with_im_start_end(self):
        print("Testing single image conversation without <image> placeholder and mm_use_im_start_end=True ...")
        sources = [
            [
                {"from": "human", "value": "Random nonsense question?"},
                {"from": "gpt", "value": "Random nonsense answer..."},
            ]
        ]
        self.data_args.mm_use_im_start_end = True
        sources = preprocess_multimodal(sources, self.data_args)
        self.assertEqual(
            sources[0][0]["value"],
            f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}\n{DEFAULT_IM_END_TOKEN}\nRandom nonsense question?",
        )

    def test_multiple_image_data_with_placeholder(self):
        print("Testing multi-images conversation with <image> placeholder ...")
        sources = [
            [
                {"from": "human", "value": "Random nonsense <image> question <image>?"},
                {"from": "gpt", "value": "Random nonsense answer..."},
            ]
        ]
        sources = preprocess_multimodal(sources, self.data_args)
        self.assertEqual(
            sources[0][0]["value"],
            f"Random nonsense {DEFAULT_IMAGE_TOKEN}\nquestion {DEFAULT_IMAGE_TOKEN}\n?",
        )


if __name__ == "__main__":
    unittest.main()
