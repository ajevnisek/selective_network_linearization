if $FREEZE_ALPHAS_AND_WEIGHTS
then
  python3 snl_finetune_freeze_alphas_and_weights.py $DATASET $ARCH $OUTDIR $INITIAL_CHECKPOINT \
    --block_type LearnableAlphaAndBetaNoSigmoidWithGamma --beta_epochs $BETA_EPOCHS --lr_beta $LR_BETA --relu_budget $RELU_BUDGET \
    --noise_init_for_betas $NOISE_INIT_FOR_BETAS --lr_snl $LR_SNL --epochs $SNL_EPOCHS  \
    --finetune_epochs $FINETUNE_EPOCHS --lr_finetune $LR_FINETUNE \
    --freeze_alphas_and_weights --ones_init_weights;
else
  python3 snl_finetune_freeze_alphas_and_weights.py $DATASET $ARCH $OUTDIR $INITIAL_CHECKPOINT \
    --block_type LearnableAlphaAndBetaNoSigmoidWithGamma --beta_epochs $BETA_EPOCHS --lr_beta $LR_BETA --relu_budget $RELU_BUDGET \
    --noise_init_for_betas $NOISE_INIT_FOR_BETAS --lr_snl $LR_SNL --epochs $SNL_EPOCHS --finetune_epochs $FINETUNE_EPOCHS \
    --ones_init_weights;
fi

python3 test_freeze_alpha_network.py $DATASET $ARCH $OUTDIR \
  $OUTDIR/snl_best_checkpoint_resnet18_in_cifar100_$RELU_BUDGET.pth.tar --block_type LearnableAlphaAndBetaNoSigmoidWithGamma