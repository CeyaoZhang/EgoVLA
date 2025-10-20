#!/bin/bash
set -e

# TIME="20240816205949"
# TIME=$(date +%Y-%m-%d_%H-%M)
MODEL=${1:-"vicuna-7b-v1.5"}

# TIME=$(date +%Y-%m) # for simplicity, we only keep monthly checkpoints
# _TIME=$(date +%Y%m%d-%H%M)
_TIME=$(date +%Y-%m)
_COMMIT_HASH=$(git rev-parse --short=12 HEAD)

TIME=${TIME:-$_TIME}
COMMIT_HASH=${COMMIT_HASH:-$_COMMIT_HASH}

ROOTDIR=$HOME/workspace/checkpoints-regression
BASEDIR=$ROOTDIR/$TIME-$COMMIT_HASH

echo $(hostname)

if [ "$MODEL" = "vicuna-7b-v1.5" ]; then
    # a quick fix to avoid error during first training
    huggingface-cli download lmsys/vicuna-7b-v1.5
    CONV_MODE="v1"
elif [ "$MODEL" = "qwen2-1.5b-instruct" ]; then
    huggingface-cli download Qwen/Qwen2-1.5B-Instruct
    CONV_MODE="auto"
elif [ "$MODEL" = "llama3-8b-s2" ]; then
    huggingface-cli download Efficient-Large-Model/Meta-Llama-3-8B-Instruct
    CONV_MODE="llama_3"
else
    echo "Invalid model input"
    exit 1
fi

mkdir -p $BASEDIR
STAGE1=$BASEDIR/$MODEL-align
STAGE2=$BASEDIR/$MODEL-pretrain
STAGE3=$BASEDIR/$MODEL-sft

# Handle interrupt signal and cleanup all slurm (training) jobs
exitfn () {
    trap SIGINT  # Restore signal handling for SIGINT.
    for stage in align pretrain sft; do
        jname=$VILA_SLURM_ACCOUNT:regression/$MODEL-$stage
        echo 'Aarghh Interuption! Canceling jobs' $jname
        scancel -n $jname
    done
    exit -1
}
trap "exitfn" INT

if [ ! -f $BASEDIR/$MODEL-align/model/config.json ]; then
    vila-run \
        --mode regression \
        --job-name $MODEL-align \
        --nodes 8 \
        --output-dir $BASEDIR/$MODEL-align \
        scripts/regression/$MODEL/align.sh
fi

if [ ! -f $BASEDIR/$MODEL-pretrain/model/config.json ]; then
    vila-run \
        --mode regression \
        --job-name $MODEL-pretrain \
        --nodes 8 \
        --output-dir $BASEDIR/$MODEL-pretrain \
        scripts/regression/$MODEL/pretrain.sh \
            $BASEDIR/$MODEL-align/model
fi

if [ ! -f $BASEDIR/$MODEL-sft/model/config.json ]; then
    vila-run \
        --mode regression \
        --job-name $MODEL-sft \
        --nodes 8 \
        --output-dir $BASEDIR/$MODEL-sft \
        scripts/regression/$MODEL/sft.sh \
            $BASEDIR/$MODEL-pretrain/model
fi

echo "Training finished, now launch evaluation"

# TODO(zhijian, ligeng): pin this to pyproject.toml.
pip install lmms-eval==0.2.1

# add interactive partition to accelerate
export VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
export WANDB_PROJECT="vila-eval"
export WANDB_ENTITY="efficient-large-model"

vila-eval \
    --model-path $BASEDIR/$MODEL-sft/model \
    --model-name $MODEL-sft \
    --conv-mode $CONV_MODE \
    --tags-include regression \
    --output-dir $BASEDIR/$MODEL-sft-eval \
    --report-to wandb

# Default backup checkpoints https://huggingface.co/Efficient-Large-Model/checkpoints-regression/tree/main
# NOTE(ligeng): better to upload separately to avoid potential conflicts
for stage in align pretrain sft sft-eval; do
    vila-upload \
        $BASEDIR/$MODEL-$stage \
        --root-dir $ROOTDIR \
        --repo-id Efficient-Large-Model/checkpoints-regression
done
