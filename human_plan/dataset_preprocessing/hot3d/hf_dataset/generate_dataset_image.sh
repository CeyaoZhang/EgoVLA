
CHUNKS=16
for IDX in {8..15}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_img_parquets.py \
        --num_chunk $CHUNKS \
        --chunk_id $IDX --frame_skip 1 &
done
wait


python human_plan/dataset_preprocessing/hot3d/hf_dataset/parquet_to_dataset_images.py

python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_images_mapping.py