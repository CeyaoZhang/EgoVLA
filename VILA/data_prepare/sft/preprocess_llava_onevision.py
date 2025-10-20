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

import json
import os
import pickle
from multiprocessing import Pool

import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm


def general_conversation_preprocessor(item, dataset_name, id):
    # process the conversation item to llava format.
    conversations = []
    ret_item = dict(id=id)
    if item["image"] is not None:
        img = item["image"]
        img_idx = 0
        save_path_to_append = os.path.join("images", dataset_name, f"{id}_{img_idx}.png")
        img_path = os.path.join(save_path, save_path_to_append)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(img_path)
        ret_item["image"] = img_path
    ret_item["conversations"] = item["conversations"]
    return ret_item


def process_dataset(args):
    dataset_name, dataset_path, metadata_path, save_path = args
    if os.path.exists(os.path.join(metadata_path, dataset_name + "_train.jsonl")):
        return
    print("Processing", dataset_name, "...")
    loaded = load_dataset(dataset_path, dataset_name)["train"]
    dataset = list(loaded)
    cnt = 0
    cur_llava_format_dataset = []
    for item in tqdm(dataset):
        new_item = general_conversation_preprocessor(item, dataset_name, cnt)
        if cnt == 0:
            print(new_item)
        cnt += 1
        cur_llava_format_dataset.append(new_item)

    with open(os.path.join(metadata_path, dataset_name + "_train.jsonl"), "w") as f:
        for item in cur_llava_format_dataset:
            json.dump(item, f)
            f.write("\n")


def main(
    dataset_path="/nobackup/datasets/LLaVA-OneVision-Data/",
    save_path="/raid/kentang/datasets/LLaVA-OneVision-Data-processed/",
):
    # download M3IT to the dataset_path directory
    metadata_path = os.path.join(save_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)

    # These two have already been processed
    skipped_datasets = ["ureader_kg", "ureader_qa"]

    _dataset_names = sorted(os.listdir(dataset_path))
    dataset_names = []
    for name in _dataset_names:
        if name.startswith("."):
            continue
        if name in skipped_datasets:
            continue
        if os.path.isdir(os.path.join(dataset_path, name)):
            dataset_names.append(name)
            os.makedirs(os.path.join(save_path, "images", name), exist_ok=True)
    print(dataset_names, len(dataset_names))

    # sequential version
    # for dataset_name in dataset_names:
    #     process_dataset((dataset_name, dataset_path, metadata_path, save_path))
    # parallel version
    with Pool(processes=min(16, len(dataset_names))) as pool:
        # Prepare the arguments for the process_dataset function
        args = [(dataset_name, dataset_path, metadata_path, save_path) for dataset_name in dataset_names]

        # Map the process_dataset function to the arguments
        for _ in tqdm(pool.imap_unordered(process_dataset, args), total=len(args), desc="Processing datasets"):
            pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)
