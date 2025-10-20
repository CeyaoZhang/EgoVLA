
# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import tqdm

# Define a generator function that loads Parquet files one by one and converts them into a dataset
def parquet_to_dataset_generator(file_paths):
    index = 0
    for file_path in tqdm.tqdm(file_paths):
        print('Number:', index)
        df = pd.read_parquet(file_path)
        dataset = Dataset.from_pandas(df)
        index = index + 1
        yield dataset

set_label = "train"

# Example usage
dataset_file_root = f"data/ha_dataset/HoloAssist_HF_images_v2"

parquet_pattern = f"data/ha_dataset/image_v2_parquets/*.parquet"

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