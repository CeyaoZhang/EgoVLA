from PIL import Image
import io
import os
import pandas as pd
import json
import glob
import copy
import tqdm
import cv2
from datasets import Dataset, concatenate_datasets, load_from_disk
import natsort



from human_plan.dataset_preprocessing.holoassist.utils import (
  parse_single_seq_image,
  convert_annotation
)

from human_plan.dataset_preprocessing.utils.hf_dataset import (
  create_data,
  save_data_to_parquet
)

def get_all_seqs(dataset_root):
  all_seq_names = os.listdir(dataset_root)
  return all_seq_names

def hot3d_write_parquet(
  all_seqs,
  dataset_root,
  frame_skip,
  image_w,
  image_h,
  write_frequency=20,
  save_path=None,
  num_chunks=None,
  chunk_id=None
):
  seq_idx = 0
  seq_data_list = []
  save_idx = 0
  for seq_name in tqdm.tqdm(all_seqs, desc=f"{chunk_id} out of {num_chunks}"):
    seq_data = parse_single_seq_image(
      seq_name=seq_name,
      dataset_root=dataset_root,
      frame_skip=frame_skip,
      image_w=image_w,
      image_h=image_h
    )
    print(dataset_root, seq_name)
    seq_data_list.extend(seq_data)

    seq_idx += 1
    if seq_idx % write_frequency == 0:
      save_data_to_parquet(
        seq_data_list,
        save_idx,
        save_path,
        chunk_id,
        dataset_prefix="HoloAssist_Image",
        skip_keys=["rgb_obs", "language_label", "frame_count", "seq_name"]
      )
      save_idx += 1
      seq_data_list = []
  
  if len(seq_data_list) > 0:
    save_data_to_parquet(
      seq_data_list,
      save_idx,
      save_path,
      chunk_id,
      dataset_prefix="HoloAssist_Image",
      skip_keys=["rgb_obs", "language_label", "frame_count", "seq_name"]
    )

import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == "__main__":
  import argparse  
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--num_chunks", type=int, default=16,
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--chunk_id", type=int, default=0,
    help="Number of parallel workers"
  )

  # parser.add_argument(
  #   "--eval", type=bool, default=False,
  #   help="Number of parallel workers"
  # )

  args = parser.parse_args()

  # set_label = "val" if args.eval else "train"
  # Example usage
  dataset_root = "data/ha_dataset"
  save_path = f"data/ha_dataset/image_parquets"
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  annotation_file = "data-annotation-trainval-v1_1.json"

  
  # if args.eval:
  eval_data_list_file = "val-v1_2.txt"
  # else:
  train_data_list_file = "train-v1_2.txt"

  annotation_path = os.path.join(
      dataset_root,
      annotation_file
  )

  import json
  with open(annotation_path, "r") as f:
    annotations = json.load(f)
  annotations = convert_annotation(annotations=annotations)


  all_seqs = get_all_seqs(os.path.join(dataset_root, "HoloAssist"))
# Open the file and read the lines
  with open(os.path.join(
    dataset_root, eval_data_list_file
  ), 'r') as file:
    eval_data_list = file.readlines()
  eval_data_list = [seq.strip() for seq in eval_data_list]
  
  with open(os.path.join(
    dataset_root, train_data_list_file
  ), 'r') as file:
    train_data_list = file.readlines()
  train_data_list = [seq.strip() for seq in train_data_list]
  
  all_seqs = [seq for seq in all_seqs if seq in eval_data_list or seq in train_data_list]
  all_seqs = get_chunk(all_seqs, args.num_chunks, args.chunk_id)

  holoassist_write_parquet(
    all_seqs=all_seqs,
    dataset_root=os.path.join(dataset_root, "HoloAssist"),
    frame_skip=5,
    image_w=384,
    image_h=384,
    write_frequency=5,
    save_path=save_path,
    num_chunks=args.num_chunks,
    chunk_id=args.chunk_id
  )
