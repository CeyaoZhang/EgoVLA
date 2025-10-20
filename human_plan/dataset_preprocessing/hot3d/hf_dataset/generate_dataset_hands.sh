
additional_tag=30Hz

python human_plan/dataset_preprocessing/hot3d/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag

python human_plan/dataset_preprocessing/hot3d/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag --eval True
