which python

CHUNKS=32
for IDX in {0..15}; do
    python human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_img_parquets.py \
        --num_chunk $CHUNKS \
        --chunk_id $IDX --frame_skip 1 &
done
wait

for IDX in {16..31}; do
    python human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_img_parquets.py \
        --num_chunk $CHUNKS \
        --chunk_id $IDX --frame_skip 1 &
done
wait

python human_plan/dataset_preprocessing/hoi4d/hf_dataset/parquet_to_dataset_images.py

python human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_images_mapping.py