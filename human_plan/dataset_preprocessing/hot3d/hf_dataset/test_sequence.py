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

from pathlib import Path

from human_plan.dataset_preprocessing.hot3d.utils import (
  parse_single_seq_image_aria,
  get_hot3d_data_provider,
)

from human_plan.dataset_preprocessing.utils.hf_dataset import (
  create_data,
  save_data_to_parquet
)

from human_plan.dataset_preprocessing.utils.funcs import (
  get_chunk
)

from human_plan.dataset_preprocessing.hot3d.utils import (
  get_all_seqs,
  get_aria_seqs
)

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
    data_provider = get_hot3d_data_provider(
      dataset_root, seq_name
    )
    seq_data = parse_single_seq_image_aria(
      data_provider,
      seq_name=seq_name,
      frame_skip=frame_skip,
      image_w=image_w,
      image_h=image_h
    )
    print(dataset_root, seq_name)
    seq_data_list.extend(seq_data)
    print(len(seq_data))


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
  parser.add_argument(
    "--frame_skip", type=int, default=6,
    help="Frame Skip"
  )

  args = parser.parse_args()

  # Example usage
  dataset_root = "data/hot3d"
  save_path = f"data/hot3d_hf/image_parquets"

  new_directory_path = Path(save_path)
  new_directory_path.mkdir(parents=True, exist_ok=True)

  all_aria_seqs = get_aria_seqs(dataset_root)
  # print("P0002_65085bfc" in all_aria_seqs)
  # exit()
  for i in range(args.num_chunks):
    current_seqs = get_chunk(all_aria_seqs, args.num_chunks, i)
    print(i, "P0002_65085bfc" in current_seqs, 'P0008_85497a0e' in current_seqs, 'P0012_c06d939b' in current_seqs)
    print(current_seqs)

  all_aria_seqs = get_chunk(all_aria_seqs, args.num_chunks, 10)
  print(all_aria_seqs)
  # all_aria_seqs = ["P0002_65085bfc"]
  # hot3d_write_parquet(
  #   all_seqs=all_aria_seqs,
  #   dataset_root=dataset_root,
  #   frame_skip=args.frame_skip,
  #   image_w=384,
  #   image_h=384,
  #   write_frequency=5,
  #   save_path=save_path,
  #   num_chunks=args.num_chunks,
  #   chunk_id=args.chunk_id
  # )
