# from PIL import Image
# import io
# import os
# import pandas as pd
# import json
# import glob
# import copy
import tqdm
# import cv2
# from datasets import Dataset, concatenate_datasets, load_from_disk
# import natsort

from pathlib import Path

from human_plan.dataset_preprocessing.hot3d.utils import (
  parse_single_seq_image_aria,
  parse_single_seq_image_quest3,
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
  get_aria_seqs,
  get_quest_seqs
)

def hot3d_write_parquet(
  all_aria_seqs,
  all_quest_seqs,
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

  is_aria_flags = [True] * len(all_aria_seqs) + [False] * len(all_quest_seqs)
  all_seqs = all_aria_seqs + all_quest_seqs
  # is_aria_flags = [False] * len(all_quest_seqs)
  # all_seqs = all_quest_seqs
  # print(len(is_aria_flags), len(all_seqs))
  # exit()
  for seq_name, is_aria in tqdm.tqdm(
    zip(all_seqs, is_aria_flags),
    desc=f"{chunk_id} out of {num_chunks}"
  ):
    try:
      data_provider = get_hot3d_data_provider(dataset_root, seq_name) # 获取数据提供者

      if is_aria:
        seq_data = parse_single_seq_image_aria(
          data_provider,
          seq_name=seq_name,
          frame_skip=frame_skip,
          image_w=image_w,
          image_h=image_h
        )
      else:
        seq_data = parse_single_seq_image_quest3(
          data_provider,
          seq_name=seq_name,
          frame_skip=frame_skip,
          image_w=image_w,
          image_h=image_h
        )
      print(dataset_root, seq_name, is_aria)
      seq_data_list.extend(seq_data)
    except Exception as e:
      print(e)

    seq_idx += 1
    if seq_idx % write_frequency == 0:
      save_data_to_parquet(
        seq_data_list,
        save_idx,
        save_path,
        chunk_id,
        dataset_prefix="HOT3D_Image",
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
      dataset_prefix="HOT3D_Image",
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
  print(len(all_aria_seqs))
  all_aria_seqs = get_chunk(all_aria_seqs, args.num_chunks, args.chunk_id)

  all_quest_seqs = get_quest_seqs(dataset_root)
  print(len(all_quest_seqs))
  all_quest_seqs = get_chunk(all_quest_seqs, args.num_chunks, args.chunk_id)

  # exit()

  hot3d_write_parquet(
    all_aria_seqs=all_aria_seqs,
    all_quest_seqs=all_quest_seqs,
    dataset_root=dataset_root,
    frame_skip=args.frame_skip,
    image_w=384,
    image_h=384,
    write_frequency=2,
    save_path=save_path,
    num_chunks=args.num_chunks,
    chunk_id=args.chunk_id
  )
