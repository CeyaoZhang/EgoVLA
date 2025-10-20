additional_tag=30Hz

CHUNKS=32

# CHUNKS=32
for IDX in {0..31}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/taco/hf_dataset/generate_hand_parquets.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag  \
        --chunk_id $IDX --frame_skip 1 --future_len 65 
done

wait

# #### Eval
CHUNKS=32
for IDX in {0..31}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/taco/hf_dataset/generate_hand_parquets.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag \
        --chunk_id $IDX --frame_skip 1 --future_len 65 --eval True  
done

wait

python human_plan/dataset_preprocessing/taco/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag

python human_plan/dataset_preprocessing/taco/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag --eval True
