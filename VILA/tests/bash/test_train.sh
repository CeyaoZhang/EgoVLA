export WANDB_DISABLED=true
CKPT1=checkpoints/CIs-stage1
CKPT2=checkpoints/CIs-stage2
CKPT3=checkpoints/CIs-stage3

rm -rfv $CKPT1 &
rm -rfv $CKPT2 &
rm -rfv $CKPT3 &
wait

bash scripts/v1_5/tests/1_mm_align.sh $CKPT1
bash scripts/v1_5/tests/2_pretrain.sh $CKPT1 $CKPT2
bash scripts/v1_5/tests/3_sft_captioner.sh $CKPT2 $CKPT3
