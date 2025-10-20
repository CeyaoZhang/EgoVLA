#!/bin/bash

# Configuration
MOCK_TYPE="disable" # choices=['disable', 'shortcut', 'ignore_mask']
SCRIPT_PATH="tests/seq_parallel/run_e2e.sh"
SP_DEGREE=4
GPU_COUNT=8
DATASET_NAME="shot2story_shotonly" # ai2d_train_12k+chartqa_train_18k+shot2story_shotonly
GLOBAL_BATCH_SIZE=64
PROJECT_ID=$(uuidgen | cut -c1-4)
PROJECT_NAME="loss_test_id${PROJECT_ID}"
MAX_STEPS=1

generate_training_script() {
    local mode=$1
    local run_name="${mode}_${PROJECT_NAME}"
    echo "MOCK_TYPE=${MOCK_TYPE} bash ${SCRIPT_PATH} ${GPU_COUNT} ${DATASET_NAME} ${mode} ${GLOBAL_BATCH_SIZE} ${run_name} ${MAX_STEPS} ${MOCK_TYPE} ${SP_DEGREE}"
}

check_log_files() {
    echo "Checking log files..."
    python tests/seq_parallel/test_check_loss_alignment.py "$PROJECT_NAME"

    local status=$?
    if [ $status -ne 0 ]; then
        echo "SP Log file check failed!"
        exit 1  # Trigger failure in the script if the log check fails
    fi
    return $status
}


test_loss_alignment() {
    dp_command=$(generate_training_script "dp")
    sp_command=$(generate_training_script "sp")

    echo "Executing SP command: $sp_command"
    eval "$sp_command"
    echo "Executing DP command: $dp_command"
    eval "$dp_command"

    # Wait and check for the log files every 10 seconds
    for i in {1..60}; do
        if check_log_files; then
            return
        fi
        sleep 10
    done

    echo "Timeout waiting for log files"
}

clean_checkpoints() {
    echo "Cleaning up checkpoints..."
    rm -rf "./checkpoints/dp_${PROJECT_NAME}"
    rm -rf "./checkpoints/sp_${PROJECT_NAME}"
    echo "Cleanup complete."
}

# Run the test
test_loss_alignment

clean_checkpoints
