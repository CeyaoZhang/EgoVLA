
# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import os
import datasets
import pickle

# Example usage
dataset_path = "data/ha_dataset/HoloAssist_HF_images_v2"

mapping_save_path = "data/ha_dataset/hf_images_v2_mapping.pkl"

loaded_dataset = datasets.load_from_disk(dataset_path)

num_data_samples = len(loaded_dataset)
# loaded_dataset = load_from_disk(dataset_file_root)

print(loaded_dataset)

import torch

from PIL import Image
from io import BytesIO

import numpy as np
# print(loaded_dataset[::100])
import tqdm

data_mapping_dict = {}

for i in tqdm.tqdm(range(num_data_samples)):
  data_sample = loaded_dataset[i]
  seq_name = data_sample["seq_name"]
  frame_count = data_sample["frame_count"]
  if seq_name not in data_mapping_dict:
    data_mapping_dict[seq_name] = {}
  data_mapping_dict[seq_name][frame_count] = i


with open(mapping_save_path, "wb") as f:
  pickle.dump(data_mapping_dict, f)