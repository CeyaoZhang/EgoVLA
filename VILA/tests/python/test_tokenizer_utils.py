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
from typing import Any, Dict, List, Tuple

from parameterized import parameterized
from transformers import AutoTokenizer, PreTrainedTokenizer

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.utils.tokenizer import infer_stop_tokens, preprocess_conversation

TEST_CONVERSATIONS = [
    [
        {
            "from": "human",
            "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase.",
        },
        {"from": "gpt", "value": "*"},
        {"from": "human", "value": "What is the title of this book?"},
        {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"},
        {"from": "human", "value": "What is the genre of this book?"},
        {"from": "gpt", "value": "Travel"},
        {"from": "human", "value": "Is this book related to Travel?"},
        {"from": "gpt", "value": "Yes"},
        {"from": "human", "value": "Is this book related to History?"},
        {"from": "gpt", "value": "No"},
    ],
    [
        {"from": "human", "value": "<image>\nWhat do you see happening in this image?"},
        {
            "from": "gpt",
            "value": "In the image, a person's hand is seen holding a pink highlighter pen, which is being used to highlight text in a notebook. The notebook is open, revealing a page filled with handwritten text. The text appears to be in a foreign language, suggesting that the person might be studying or learning a new language. The background of the image is blurred, drawing focus to the hand and the notebook. The overall scene suggests a learning or study environment.",
        },
    ],
    [
        {
            "from": "human",
            "value": "Who is the author of this book?\nAnswer the question using a single word or phrase.",
        },
        {"from": "gpt", "value": "*"},
        {"from": "human", "value": "What is the title of this book?"},
        {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"},
        {"from": "human", "value": "What is the genre of this book?"},
        {"from": "gpt", "value": "Travel"},
        {"from": "human", "value": "Is this book related to Travel?"},
        {"from": "gpt", "value": "Yes"},
        {"from": "human", "value": "Is this book related to History?"},
        {"from": "gpt", "value": "No"},
    ],
]


def _build_test_tokenizers() -> List[Tuple[PreTrainedTokenizer, Dict[str, Any]]]:
    tokenizers = []

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B")
    tokenizers.append((tokenizer, {"conv_mode": "vicuna_v1", "stop_tokens": ["</s>"]}))

    tokenizer = AutoTokenizer.from_pretrained("Efficient-Large-Model/Meta-Llama-3-8B-Instruct")
    tokenizers.append((tokenizer, {"conv_mode": "llama_3", "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"]}))

    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    tokenizers.append((tokenizer, {"conv_mode": "vicuna_v1", "stop_tokens": ["</s>"]}))

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Yi-34B", use_fast=False)
    tokenizers.append((tokenizer, {"conv_mode": "hermes-2", "stop_tokens": ["<|im_end|>"]}))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    tokenizers.append((tokenizer, {"stop_tokens": ["<|im_end|>", "<|endoftext|>"]}))

    return tokenizers


TEST_TOKENIZERS = _build_test_tokenizers()


class TestTokenizerUtils(unittest.TestCase):
    @parameterized.expand(TEST_TOKENIZERS)
    def test_preprocess_conversation(self, tokenizer: PreTrainedTokenizer, info: Dict[str, Any]) -> None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[info.get("conv_mode", "auto")]
        for conversation in TEST_CONVERSATIONS:
            labels = preprocess_conversation(conversation, tokenizer)["labels"]

            # Decode the responses from the label
            outputs = []
            for k, label in enumerate(labels):
                if label == IGNORE_INDEX:
                    continue
                if k == 0 or labels[k - 1] == IGNORE_INDEX:
                    start = k
                if k == len(labels) - 1 or labels[k + 1] == IGNORE_INDEX:
                    end = k
                    outputs.append(tokenizer.decode(labels[start:end]).lstrip())

            # Get the target responses
            targets = []
            for source in conversation:
                if source["from"] == "gpt":
                    targets.append(source["value"])

            self.assertListEqual(outputs, targets)

    @parameterized.expand(TEST_TOKENIZERS)
    def test_infer_stop_tokens(self, tokenizer: PreTrainedTokenizer, info: Dict[str, Any]) -> None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[info.get("conv_mode", "auto")]
        outputs = infer_stop_tokens(tokenizer)
        targets = info["stop_tokens"]
        self.assertSetEqual(set(outputs), set(targets))


if __name__ == "__main__":
    unittest.main()
