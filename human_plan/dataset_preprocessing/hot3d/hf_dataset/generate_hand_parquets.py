import tqdm

from pathlib import Path

from human_plan.dataset_preprocessing.hot3d.utils import (
  # parse_single_seq_image_aria,
  parse_single_seq_hand,
  get_hot3d_data_provider,
)

from human_plan.dataset_preprocessing.utils.hf_dataset import (
  create_data,
  save_data_to_parquet
)

from human_plan.dataset_preprocessing.utils.funcs import (
  get_chunk,
  split_train_eval
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

  is_aria_flags = [True] * len(all_aria_seqs) + [False] * len(all_quest_seqs)
  all_seqs = all_aria_seqs + all_quest_seqs

  for seq_name, is_aria in tqdm.tqdm(
    zip(all_seqs, is_aria_flags), 
    desc=f"{chunk_id} out of {num_chunks}"
  ):
    try:
      data_provider = get_hot3d_data_provider(
        dataset_root, seq_name
      )
      seq_data = parse_single_seq_hand(
        data_provider,
        seq_name=seq_name,
        is_aria=is_aria,
        frame_skip=frame_skip,
        sample_skip=sample_skip,
        future_len=future_len,
        his_len=future_len,
        image_mapping_dict=image_mapping_dict
      )
      print(dataset_root, seq_name)
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
        dataset_prefix="HOT3D_hand",
        skip_keys=["rgb_obs", "language_label", "raw_height", "raw_width", "frame_count", "seq_name"]
      )
      save_idx += 1
      seq_data_list = []

  if len(seq_data_list) > 0:
    save_data_to_parquet(
      seq_data_list,
      save_idx,
      save_path,
      chunk_id,
      dataset_prefix="HOT3D_hand",
      skip_keys=["rgb_obs", "language_label", "raw_height", "raw_width", "frame_count", "seq_name"]
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
    "--eval", type=bool, default=False,
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--additional_tag", type=str, default="",
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--frame_skip", type=int, default=6,
    help="Frame Skip"
  )
  parser.add_argument(
    "--sample_skip", type=int, default=5,
    help="Frame Skip"
  )
  parser.add_argument(
    "--future_len", type=int, default=40,
    help="Future Len"
  )

  args = parser.parse_args()
  set_label = "val" if args.eval else "train"

  # Example usage
  dataset_root = "data/hot3d"
  save_path = f"data/hot3d_hf/hand_{args.additional_tag}_{set_label}_parquets"

  import pickle
  image_mapping_path = "data/hot3d_hf/hf_images_mapping.pkl"

  with open(image_mapping_path, "rb") as f:            
    image_mapping_dict = pickle.load(f)
    
  new_directory_path = Path(save_path)
  new_directory_path.mkdir(parents=True, exist_ok=True)

  all_aria_seqs = get_aria_seqs(dataset_root)
  all_quest_seqs = get_quest_seqs(dataset_root)

  # all_seqs = all_aria_seqs + all_quest_seqs

  # print(len(all_seqs))
  
  train_aria_seqs, eval_aria_seqs = split_train_eval(all_aria_seqs)
  train_quest_seqs, eval_quest_seqs = split_train_eval(all_quest_seqs)

  aria_seqs = eval_aria_seqs if args.eval else train_aria_seqs
  quest_seqs = eval_quest_seqs if args.eval else train_quest_seqs
  aria_seqs = get_chunk(aria_seqs, args.num_chunks, args.chunk_id)
  quest_seqs = get_chunk(quest_seqs, args.num_chunks, args.chunk_id)
  print(len(all_aria_seqs))
  print(len(all_quest_seqs))
  # exit()
  hot3d_write_parquet(
    all_aria_seqs=aria_seqs,
    all_quest_seqs=quest_seqs,
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
