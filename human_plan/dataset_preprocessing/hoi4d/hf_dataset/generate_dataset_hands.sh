additional_tag=V2

CHUNKS=16
for IDX in {0..15}; do
   python human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_hand_parquets.py \
       --num_chunk $CHUNKS  --additional_tag $additional_tag  \
       --chunk_id $IDX --frame_skip 1 --sample_skip 5 --future_len 61
done

wait

CHUNKS=16
for IDX in {0..15}; do
   python human_plan/dataset_preprocessing/hoi4d/hf_dataset/generate_hand_parquets.py \
       --num_chunk $CHUNKS  --additional_tag $additional_tag \
       --chunk_id $IDX --frame_skip 1 --sample_skip 5 --future_len 61 --eval True
done

wait

python human_plan/dataset_preprocessing/hoi4d/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag

python human_plan/dataset_preprocessing/hoi4d/hf_dataset/parquet_to_dataset_hand.py --additional_tag $additional_tag --eval True
