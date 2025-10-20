
# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import tqdm

from human_plan.dataset_preprocessing.utils.hf_dataset import parquet_to_dataset_generator


if __name__ == "__main__":
  import argparse  
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--eval", type=bool, default=False,
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--additional_tag", type=str, default="",
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--save_path", type=str, default="",
    help="Save Path"
  )
  args = parser.parse_args()

  set_label = "val" if args.eval else "train"

  # Example usage
  dataset_file_root = f"{args.save_path}/HF_hand_{args.additional_tag}_{set_label}"
  parquet_pattern = f"{args.save_path}/hand_{args.additional_tag}_{set_label}_parquets/*.parquet"

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