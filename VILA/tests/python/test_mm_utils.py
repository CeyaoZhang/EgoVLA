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
from unittest import mock

import torch
from parameterized import parameterized
from PIL import Image

from llava import mm_utils

CLOSEST_ASPECT_RATIO_TEST_CASES = [
    ((4 / 3), 640, 480, 384, (4, 3)),
    ((16 / 9), 1920, 1080, 384, (4, 2)),
    ((3 / 4), 800, 1000, 384, (3, 4)),
    ((1.5), 1200, 800, 384, (3, 2)),
    ((1.2), 1200, 1000, 384, (4, 3)),
    ((1.8), 1900, 1000, 384, (4, 2)),
]

DYNAMIC_PREPROCESS_TEST_CASES = [
    (760, 350, 384, 12, 3),
    (1000, 750, 384, 6, 7),
    (1000, 750, 384, 12, 13),
    (1200, 800, 384, 12, 7),
    (1200, 900, 384, 12, 13),
]


class TestProcessImage(unittest.TestCase):
    def setUp(self):
        self.mock_data_args = mock.MagicMock()
        self.mock_data_args.image_processor = mock.MagicMock()
        self.mock_data_args.image_aspect_ratio = "dynamic"
        self.mock_data_args.image_processor.crop_size = {"height": 384, "width": 384}
        self.mock_data_args.image_processor.preprocess = lambda *args, **kwargs: {"pixel_values": torch.zeros(1, 1, 1)}

    @parameterized.expand(CLOSEST_ASPECT_RATIO_TEST_CASES)
    def test_find_closest_aspect_ratio(self, aspect_ratio, width, height, image_size, expected_ratio):
        # Test cases with various aspect ratios, target ratios, widths, heights, and image sizes
        min_num = 1
        max_num = 12
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        result = mm_utils.find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
        self.assertEqual(result, expected_ratio, f"Expected: {expected_ratio}, Got: {result}")

    @parameterized.expand(DYNAMIC_PREPROCESS_TEST_CASES)
    def test_dynamic_preprocess(self, height, width, image_size, max_num, num_tiles):
        image = Image.new("RGB", (height, width))
        processed_images = mm_utils.dynamic_preprocess(
            image, min_num=1, max_num=max_num, image_size=image_size, use_thumbnail=True
        )
        self.assertEqual(len(processed_images), num_tiles)
        for img in processed_images:
            self.assertEqual(img.size, (image_size, image_size))

    def test_dynamic_process_images_and_prompt(self):
        images = [Image.new("RGB", (760, 384))]
        prompt = "<image> test"

        mm_utils.dynamic_process_images_and_prompt(images, prompt, self.mock_data_args, image_folder=None)

    def test_process_image(self):
        mock_image = Image.new("RGB", (760, 384))
        mock_image_folder = None

        # Call the function with mocked data
        mm_utils.process_image(mock_image, self.mock_data_args, mock_image_folder, enable_dynamic_res=True)

    def test_process_images(self):
        mock_image = [Image.new("RGB", (760, 384)), Image.new("RGB", (1000, 2000))]

        # Call the function with mocked data
        mm_utils.process_images(
            mock_image, self.mock_data_args.image_processor, self.mock_data_args, enable_dynamic_res=True
        )


if __name__ == "__main__":
    unittest.main()
