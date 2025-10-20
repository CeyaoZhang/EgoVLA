conda activate hot3d

additional_tag=_skip1_mano_optim

CHUNKS=64
for IDX in {32..63}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 4)) python human_plan/dataset_preprocessing/holoassist/hf_dataset/holoassist_to_parquet_hands_filter.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag \
        --frame_skip 1 --sample_skip 5 \
        --future_len 61 \
        --chunk_id $IDX &
done

wait

CHUNKS=64
for IDX in {32..63}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 4)) python human_plan/dataset_preprocessing/holoassist/hf_dataset/holoassist_to_parquet_hands_filter.py \
        --num_chunk $CHUNKS  --additional_tag $additional_tag \
        --frame_skip 1 --sample_skip 5 \
        --future_len 61 \
        --chunk_id $IDX --eval True &
done

wait
