
# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import os
import datasets
import pickle


if __name__ == "__main__":

  import argparse  
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--save_path", type=str, default="",
    help="Number of parallel workers"
  )
  args = parser.parse_args()
  # Example usage
  dataset_path = f"{args.save_path}/HF_images"
  mapping_save_path = f"{args.save_path}/hf_images_mapping.pkl"

  loaded_dataset = datasets.load_from_disk(dataset_path)

  num_data_samples = len(loaded_dataset)

  print(loaded_dataset)

  import numpy as np
  import tqdm

  # Function to process a subset of the dataset
  def process_subset(start_idx, end_idx):
      subset_mapping = {}
      for i in tqdm.tqdm(range(start_idx, end_idx)):
          data_sample = loaded_dataset[i]
          seq_name = data_sample["seq_name"]
          frame_count = data_sample["frame_count"]
          if seq_name not in subset_mapping:
              subset_mapping[seq_name] = {}
          subset_mapping[seq_name][frame_count] = i
      return subset_mapping

  # Define number of workers and batch size
  num_workers = 32  # Adjust based on your CPU count
  import math
  batch_size = math.ceil(num_data_samples / num_workers)

  # Use ProcessPoolExecutor for parallel processing

  from concurrent.futures import ProcessPoolExecutor, as_completed

  data_mapping_dict = {}
  with ProcessPoolExecutor(max_workers=num_workers) as executor:
      futures = []
      for i in range(num_workers):
          start_idx = i * batch_size
          end_idx = min((i + 1) * batch_size, num_data_samples)
          futures.append(executor.submit(process_subset, start_idx, end_idx))

      # Collect results
      for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
          subset_mapping = future.result()
          for seq_name, frames in subset_mapping.items():
              if seq_name not in data_mapping_dict:
                  data_mapping_dict[seq_name] = {}
              data_mapping_dict[seq_name].update(frames)

  # Save the mapping dictionary to a file
  with open(mapping_save_path, "wb") as f:
      pickle.dump(data_mapping_dict, f)