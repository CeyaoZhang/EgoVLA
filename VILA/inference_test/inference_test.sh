source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA

# NOTE(ligeng): all testing scripts should use relative path.
python ~/workspace/VILA/inference_test/inference_test.py \
    --model-name $1 \
    --test_json_path ~/workspace/VILA/inference_test/inference_test.json \
    --test_image_path ~/workspace/VILA/inference_test/test_data \
    --conv-mode vicuna_v1
