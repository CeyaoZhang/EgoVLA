conda activate hot3d
which python

additional_tag=_skip1_mano_optim


python human_plan/dataset_preprocessing/holoassist/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag

python human_plan/dataset_preprocessing/holoassist/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag --eval True
