
def get_args():
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
  parser.add_argument(
    "--clip_starting", type=int, default=20,
    help="clip starting"
  )
  parser.add_argument(
    "--seq_skip", type=int, default=1,
    help="Future Len"
  )
  parser.add_argument(
    "--room_idx", type=int, default=1,
    help="Room IDX"
  )
  parser.add_argument(
    "--table_idx", type=int, default=1,
    help="Table IDX"
  )
  parser.add_argument(
    "--dataset_root", type=str, default="",
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--save_path", type=str, default="",
    help="Number of parallel workers"
  )
  parser.add_argument(
    "--filter_by_success", type=bool, default=False,
    help="Number of parallel workers"
  )
  args = parser.parse_args()
  return args