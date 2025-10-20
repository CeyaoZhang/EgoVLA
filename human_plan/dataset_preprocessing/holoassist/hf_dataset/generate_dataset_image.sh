CHUNKS=32

for IDX in {0..31}; do
    python human_plan/dataset_preprocessing/holoassist/hf_dataset/holoassist_to_parquet_images.py \
        --num_chunk $CHUNKS \
        --chunk_id $IDX --frame_skip 1 &
done
wait

python human_plan/dataset_preprocessing/holoassist/hf_dataset/parquet_to_dataset.py
python human_plan/dataset_preprocessing/holoassist/hf_dataset/generate_images_mapping.py