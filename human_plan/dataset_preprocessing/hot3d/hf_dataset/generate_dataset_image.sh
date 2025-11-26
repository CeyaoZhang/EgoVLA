#!/bin/bash
# 生成图像 Parquet 文件
CHUNKS=16
for IDX in {0..15}; do
    CUDA_VISIBLE_DEVICES=$(($IDX % 8)) python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_img_parquets.py \
        --num_chunk $CHUNKS \
        --chunk_id $IDX --frame_skip 1 & # & 表示后台运行，不等待完成
done
wait  # 等待所有后台进程完成

# 合并 Parquet → HuggingFace Dataset, 合并 parquet（I/O 密集，不需要 GPU）
python human_plan/dataset_preprocessing/hot3d/hf_dataset/parquet_to_dataset_images.py 

# 生成映射（内存操作，很快）
python human_plan/dataset_preprocessing/hot3d/hf_dataset/generate_images_mapping.py 