
additional_tag=30Hz


CHUNKS=16
for IDX in {8..15}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_hand_parquets.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag  \
        --chunk_id $IDX --frame_skip 1 --sample_skip 5 --future_len 65 
done

wait

CHUNKS=16
for IDX in {8..15}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_hand_parquets.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag \
        --chunk_id $IDX --frame_skip 1 --sample_skip 5 --future_len 65 --eval True 
done

wait