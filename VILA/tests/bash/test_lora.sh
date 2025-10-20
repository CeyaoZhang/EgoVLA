export WANDB_DISABLED=true
CKPT1=checkpoints/CIs-lora-test

rm -rfv $CKPT1 &
wait

# bash scripts/finetuning/tests/nosp_sqa_lora.sh --vt ft --llm ft --output_dir $CKPT1
# rm -rfv $CKPT1/config.json &
# rm -rfv $CKPT1/llm &
# rm -rfv $CKPT1/trainer_state.json &
# rm -rfv $CKPT1/vision_tower &
# wait
# bash scripts/finetuning/tests/nosp_sqa_lora.sh --vt ft --llm ft --output_dir $CKPT1
# rm -rfv $CKPT1 &
# wait

bash scripts/finetuning/tests/nosp_sqa_lora.sh --vt ft --llm lora --output_dir $CKPT1
rm -rfv $CKPT1/config.json &
rm -rfv $CKPT1/llm &
rm -rfv $CKPT1/trainer_state.json &
rm -rfv $CKPT1/vision_tower &
wait
bash scripts/finetuning/tests/nosp_sqa_lora.sh --vt ft --llm lora --output_dir $CKPT1
