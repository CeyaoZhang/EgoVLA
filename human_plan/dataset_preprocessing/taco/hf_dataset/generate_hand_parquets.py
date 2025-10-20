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

import zipfile

from human_plan.dataset_preprocessing.taco.utils import (
    parse_single_seq_hand,
)

from human_plan.dataset_preprocessing.utils.hf_dataset import save_data_to_parquet

from human_plan.dataset_preprocessing.utils.funcs import get_chunk, split_train_eval

from human_plan.dataset_preprocessing.taco.utils import (
    get_all_seqs,
)


def taco_write_parquet(
    all_seqs,
    dataset_root,
    frame_skip,
    sample_skip,
    future_len,
    write_frequency=20,
    save_path=None,
    num_chunks=None,
    chunk_id=None,
    image_mapping_dict=None
):
    seq_idx = 0
    seq_data_list = []
    save_idx = 0

    for seq_name in tqdm.tqdm(all_seqs, desc=f"{chunk_id} out of {num_chunks}"):
        seq_data = parse_single_seq_hand(
            dataset_root=dataset_root,
            seq_name=seq_name,
            frame_skip=frame_skip,
            future_len=future_len,
            sample_skip=sample_skip,
            image_mapping_dict=image_mapping_dict
        )
        print(dataset_root, seq_name, len(seq_data))
        seq_data_list.extend(seq_data)

        seq_idx += 1
        if seq_idx % write_frequency == 0:
            save_data_to_parquet(
                seq_data_list,
                save_idx,
                save_path,
                chunk_id,
                dataset_prefix="taco_hand",
                skip_keys=[
                    "rgb_obs",
                    "language_label",
                    "language_label_short",
                    "language_label_verb",
                    "language_label_noun",
                    "frame_count",
                    "raw_width",
                    "raw_height",
                    "seq_name",
                ],
            )
            save_idx += 1
            seq_data_list = []

    if len(seq_data_list) > 0:
        save_data_to_parquet(
            seq_data_list,
            save_idx,
            save_path,
            chunk_id,
            dataset_prefix="taco_hand",
            skip_keys=[
                "rgb_obs",
                "language_label",
                "language_label_short",
                "language_label_verb",
                "language_label_noun",
                "frame_count",
                "raw_width",
                "raw_height",
                "seq_name",
            ],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_chunks", type=int, default=16, help="Number of parallel workers"
    )
    parser.add_argument(
        "--chunk_id", type=int, default=0, help="Number of parallel workers"
    )
    parser.add_argument(
        "--eval", type=bool, default=False, help="Number of parallel workers"
    )
    parser.add_argument(
        "--additional_tag", type=str, default="", help="Number of parallel workers"
    )
    parser.add_argument("--frame_skip", type=int, default=6, help="Frame Skip")
    parser.add_argument("--sample_skip", type=int, default=5, help="Frame Skip")
    parser.add_argument("--future_len", type=int, default=40, help="Future Len")
    args = parser.parse_args()
    set_label = "val" if args.eval else "train"

    # Example usage
    ## TODO please replace the following paths with your local paths

    # Example usage
    dataset_root = f"data/TACO"

    save_path = f"data/TACO_HF/hand_{args.additional_tag}_{set_label}_parquets"

    new_directory_path = Path(save_path)
    new_directory_path.mkdir(parents=True, exist_ok=True)
    all_seqs = get_all_seqs(dataset_root)
    train_seqs, eval_seqs = split_train_eval(all_seqs)
    input_seqs = eval_seqs if args.eval else train_seqs
    input_seqs = get_chunk(input_seqs, args.num_chunks, args.chunk_id)

    print(len(all_seqs))

    image_mapping_path = f"data/TACO_HF/hf_images_mapping.pkl"
    import pickle
    with open(image_mapping_path, "rb") as f:
      image_mapping_dict = pickle.load(f)

    taco_write_parquet(
        all_seqs=input_seqs,
        dataset_root=dataset_root,
        frame_skip=args.frame_skip,
        sample_skip=args.sample_skip,
        future_len=args.future_len,
        write_frequency=2,
        save_path=save_path,
        num_chunks=args.num_chunks,
        chunk_id=args.chunk_id,
        image_mapping_dict=image_mapping_dict
    )
