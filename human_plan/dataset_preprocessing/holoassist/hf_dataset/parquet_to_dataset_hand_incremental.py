import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
from tqdm import tqdm
import os

from human_plan.dataset_preprocessing.utils.hf_dataset import parquet_to_dataset_generator

def save_partial_dataset(partial_dataset, output_dir, chunk_idx):
    # Save each chunk to disk
    partial_output_path = os.path.join(output_dir, f"dataset_chunk_{chunk_idx}")
    partial_dataset.save_to_disk(partial_output_path)
    print(f"Saved chunk {chunk_idx} to disk at {partial_output_path}")
    return partial_output_path

if __name__ == "__main__":
    import argparse  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", type=bool, default=False,
        help="Evaluation mode"
    )
    parser.add_argument(
        "--additional_tag", type=str, default="",
        help="Additional tag for dataset"
    )

    args = parser.parse_args()

    set_label = "val" if args.eval else "train"

    # Example usage
    dataset_file_root = f"data/ha_dataset/HoloAssist_HF_hands_filtered{args.additional_tag}_{set_label}_incremental"
    parquet_pattern = f"data/ha_dataset/hand_filtered{args.additional_tag}_{set_label}_parquets/*.parquet"

    parquet_file_paths = glob.glob(parquet_pattern)

    # # Process and concatenate in chunks
    import math
    split_size = 5
    chunk_size = math.ceil(len(parquet_file_paths) / split_size)  # You can adjust this based on memory availability
    chunked_files = [
      parquet_file_paths[i:i + chunk_size] for i in range(0, len(parquet_file_paths), chunk_size)
    ]

    all_chunk_paths = []
    for chunk_idx, chunk in tqdm(enumerate(chunked_files), desc="Incremental Datasets"):
        if chunk_idx < 3:
            continue
        print(chunk_idx)
        print(chunk_idx)
        print(chunk_idx)
        print(chunk_idx)
        parquet_datasets = list(parquet_to_dataset_generator(chunk))
        merged_chunk_dataset = concatenate_datasets(parquet_datasets)

        # Save each chunk to disk
        chunk_output_path = save_partial_dataset(merged_chunk_dataset, dataset_file_root, chunk_idx)
        all_chunk_paths.append(chunk_output_path)

    # Optionally: Load all the saved chunks and merge them back into a final dataset

    all_chunk_paths = []
    final_datasets = []
    for chunk_idx, chunk in tqdm(enumerate(chunked_files), desc="Incremental Datasets"):
        # Save each chunk to disk
        chunk_output_path = partial_output_path = os.path.join(dataset_file_root, f"dataset_chunk_{chunk_idx}")
        all_chunk_paths.append(chunk_output_path)
        final_datasets.append(load_from_disk(chunk_output_path))
    print("Loaded separate datasets")
    # final_datasets = [load_from_disk(path) for path in all_chunk_paths]
    final_merged_dataset = concatenate_datasets(final_datasets)
    print(final_merged_dataset)
