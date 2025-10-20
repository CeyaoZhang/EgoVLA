#/bin/bash

# TODO(ligeng): replace with the model with finalized 7B model.
#   VILA1.5-3b -> vicuna_v1
#   VILA1.5-13b -> vicuna_v1
#   Llama-3-VILA1.5-8B -> llama_3
#   VILA1.5-40B -> hermes-2

# infernece test with a single image
python llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --query "Please describe the image" \
    --image-file inference_test/test_data/caption_meat.jpeg

if [ $? != 0 ];
then
    exit -1
fi

# infernece test with a single video
python llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --query "Please describe the video" \
    --video-file https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4

if [ $? != 0 ];
then
    exit -1
fi
