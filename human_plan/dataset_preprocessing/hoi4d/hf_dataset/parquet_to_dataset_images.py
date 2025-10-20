
# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import tqdm

from human_plan.dataset_preprocessing.utils.hf_dataset import parquet_to_dataset_generator

# Example usage
dataset_file_root = f"data/hoi4d_hf/HF_images_30hz"

parquet_pattern = f"data/hoi4d_hf/image_parquets_30hz/*.parquet"

# same for MagicBrush
parquet_file_paths = glob.glob(parquet_pattern)
parquet_datasets = list(parquet_to_dataset_generator(parquet_file_paths))
merged_dataset = concatenate_datasets(parquet_datasets)
print(merged_dataset)

# load
# MagicBruth_HF_path = 'Seedx_Multiturn_HF'
merged_dataset.save_to_disk(dataset_file_root)

loaded_dataset = load_from_disk(dataset_file_root)
print(merged_dataset)
# MagicBruth_HF_path = load_from_disk(MagicBruth_HF_path)