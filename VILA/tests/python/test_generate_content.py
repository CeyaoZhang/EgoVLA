import unittest
from typing import Any

from parameterized import parameterized
from PIL import Image

import llava

TEST_PROMPTS = [
    ["What is the color of the sky?"],
    [[llava.Image("demo_images/vila-logo.jpg"), "please describe the image"]],
    [[Image.open("demo_images/vila-logo.jpg"), "please describe the image"]],
]


class TestGenerateContent(unittest.TestCase):
    def setUp(self):
        self.model = llava.load("Efficient-Large-Model/VILA1.5-3b")

    @parameterized.expand(TEST_PROMPTS)
    def test_generate_content(self, prompt: Any) -> None:
        self.model.generate_content(prompt)


if __name__ == "__main__":
    unittest.main()
