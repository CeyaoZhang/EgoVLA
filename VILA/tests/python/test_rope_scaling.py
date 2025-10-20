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
import sys
import unittest


def patch_unit_test_rope_scaling(model, config, training_args):

    import math

    input_max_len = training_args.model_max_length
    model_max_position_embeddings = config.max_position_embeddings
    ideal_scaling_factor = float(math.ceil(input_max_len / model_max_position_embeddings))
    rope_scaling_info = config.rope_scaling
    assert rope_scaling_info["factor"] == ideal_scaling_factor
    print("rope_scaling factor is correct")
    print(f"scaling_factor: {ideal_scaling_factor}")
    print(f"scaling_type: {rope_scaling_info['type']}")
    # rotary_emb = model.model.layers[0].self_attn.rotary_emb
    # model_max_position_embeddings_in_practice = rotary_emb.max_position_embeddings
    # print(f"model_max_position_embeddings_in_practice: {model_max_position_embeddings_in_practice}")
    # assert model_max_position_embeddings_in_practice == input_max_len
    # print("max_position_embeddings within the model is correct")

    print("unit_test_rope_scaling passed")

    return True


class TestRopeScaling(unittest.TestCase):
    def setUp(self):

        from dataclasses import dataclass, field
        from typing import Optional

        from llava.train.args import ModelArguments as UnpatchedModelArguments
        from llava.train.args import TrainingArguments as UnpatchedTrainingArguments

        user_name = os.getenv("USER")
        print(f"User name: {user_name}")

        @dataclass
        class PatchedTrainingArguments(UnpatchedTrainingArguments):
            output_dir: str = field(
                default=f"{user_name}/cache/vila_rope_scaling_test_output",
                metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
            )
            overwrite_output_dir: bool = field(
                default=True,
                metadata={
                    "help": (
                        "Overwrite the content of the output directory. "
                        "Use this to continue training if output_dir points to a checkpoint directory."
                    )
                },
            )
            model_max_length: int = field(
                default=8192,
                metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
            )

        @dataclass
        class PatchedModelArguments(UnpatchedModelArguments):
            model_name_or_path: Optional[str] = field(default="/home/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5")
            vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
            mm_vision_select_layer: Optional[int] = field(default=-2)
            mm_projector: Optional[str] = field(default="mlp2x_gelu")

        self.PatchedTrainingArguments = PatchedTrainingArguments
        self.PatchedModelArguments = PatchedModelArguments

    def test_rope_scaling(self):

        with (
            unittest.mock.patch("llava.train.args.TrainingArguments", new=self.PatchedTrainingArguments),
            unittest.mock.patch("llava.train.args.ModelArguments", new=self.PatchedModelArguments),
            unittest.mock.patch("llava.train.utils.unit_test_rope_scaling", new=patch_unit_test_rope_scaling),
        ):
            sys.argv = sys.argv[:1]
            from llava.train.train import train

            train()


if __name__ == "__main__":
    unittest.main()
