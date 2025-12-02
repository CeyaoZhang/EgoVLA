# from PIL import Image
# import io
# import os
# import pandas as pd
# import json
# import glob
# import copy
# import cv2
# from datasets import Dataset, concatenate_datasets, load_from_disk
# import natsort

import tqdm
from pathlib import Path

from human_plan.dataset_preprocessing.utils.funcs import get_chunk
from human_plan.dataset_preprocessing.utils.hf_dataset import save_data_to_parquet
from human_plan.dataset_preprocessing.otv_isaaclab.utils import get_all_seqs, parse_single_seq_image



def otv_isaaclab_write_parquet(
  all_seqs,
  dataset_root,
  frame_skip,
  image_w,
  image_h,
  write_frequency=20,
  save_path=None,
  num_chunks=None,
  chunk_id=None,
  clip_starting=20
):
  seq_idx = 0
  seq_data_list = []
  save_idx = 0

  for task, seq_name in tqdm.tqdm(all_seqs, desc=f"{chunk_id} out of {num_chunks}"):
    
    seq_data = parse_single_seq_image(
      dataset_root,
      task_name=task,
      seq_name=seq_name,
      # frame_skip=frame_skip,
      image_w=image_w,
      image_h=image_h,
      clip_starting=clip_starting
    )
    print(dataset_root, task, seq_name, len(seq_data))
    seq_data_list.extend(seq_data)

    seq_idx += 1
    if seq_idx % write_frequency == 0:
      save_data_to_parquet(
        seq_data_list,
        save_idx,
        save_path,
        chunk_id,
        dataset_prefix="OTV_ISAACLAB_Image",
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
      dataset_prefix="OTV_ISAACLAB_Image",
      skip_keys=["rgb_obs", "language_label", "frame_count", "seq_name"]
    )

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
    "--frame_skip", type=int, default=1,
    help="Frame Skip"
  )
  parser.add_argument(
    "--clip_starting", type=int, default=20,
    help="clip starting"
  )
  parser.add_argument(
    "--dataset_root", type=str, default="",
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--save_path", type=str, default="",
    help="Number of parallel workers"
  )
  args = parser.parse_args()

  # Example usage
  dataset_root = args.dataset_root
  save_path = f"{args.save_path}/image_parquets"

  new_directory_path = Path(save_path)
  new_directory_path.mkdir(parents=True, exist_ok=True)

  all_seqs = get_all_seqs(dataset_root)
  print(len(all_seqs))
  # print(all_seqs)
  # exit()
  all_seqs = get_chunk(all_seqs, args.num_chunks, args.chunk_id)

  print(len(all_seqs))

  otv_isaaclab_write_parquet(
    all_seqs=all_seqs,
    dataset_root=dataset_root,
    frame_skip=args.frame_skip,
    image_w=384,
    image_h=384,
    write_frequency=10,
    save_path=save_path,
    num_chunks=args.num_chunks,
    chunk_id=args.chunk_id,
    clip_starting=args.clip_starting
  )
