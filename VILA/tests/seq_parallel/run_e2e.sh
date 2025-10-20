#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA
# cd ~/workdir/VILA

export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDNN_DETERMINISTIC=1

# Check if required arguments are provided
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <gpu_count> <dataset_name> <mode> <global_batch_size> <project_name> <max_steps> <mock_type> <sp_degree>"
    echo "Mode should be either 'dp' or 'sp'"
    exit 1
fi
if [ "$SLURM_JOB_NUM_NODES" -ne 1 ]; then
    echo "Error: This script must be run on a single node (n_node = 1). Current n_node: $SLURM_JOB_NUM_NODES"
    exit 1
fi

gpu_count=$1
dataset_name=$2
mode=$3
global_batch_size=$4
project_name=$5
max_steps=$6
mock_type=$7
sp_degree=$8

if [[ "$mode" != "dp" && "$mode" != "sp" ]]; then
    echo "Error: mode must be either 'dp' or 'sp'"
    exit 1
fi


if [ "$mock_type" = "disable" ]; then
    if [ "$mode" = "dp" ]; then
        script_name="llava/train/train_mem.py"
        bs=$(($global_batch_size / $gpu_count))
        sp_size=-1
    elif [ "$mode" = "sp" ]; then
        script_name="llava/train/train_hybrid.py"
        bs=$(($global_batch_size / $gpu_count * $sp_degree))
        sp_size=$sp_degree
    fi
else
    if [ "$mode" = "dp" ]; then
        script_name="tests/seq_parallel/attn_mock/train_mem_mock_attn.py"
        bs=$(($global_batch_size / $gpu_count))
        sp_size=-1
    elif [ "$mode" = "sp" ]; then
        script_name="tests/seq_parallel/attn_mock/train_hybrid_mock_attn.py"
        bs=$(($global_batch_size / $gpu_count * $sp_degree))
        sp_size=$sp_degree
    fi
fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

# --model_name_or_path meta-llama/Meta-Llama-3-8B \
# --version llama_3 \
# ai2d_train_12k+chartqa_train_18k+shot2story_shotonly

torchrun --nnodes=$n_node --nproc_per_node=$gpu_count --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    $script_name \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_mixture $dataset_name \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$project_name \
    --max_steps $max_steps \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 270 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --num_video_frames 8 \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --debug_e2e True \
    --seq_parallel_size $sp_size
