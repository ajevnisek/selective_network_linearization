echo $SIGMA_TYPE
python3 snl_relu_autoencoder_multiple_layers_one_by_one.py $DATASET $ARCH $OUTDIR $INITIAL_CHECKPOINT \
    --block_type LearnableAlpha  --layer_name $LAYER_NAME --sigma_type $SIGMA_TYPE \
    --hidden_dim $HIDDEN_DIM  --learning-rate $LEARNING_RATE \
    --beta_epochs $BETA_EPOCHS --lr_beta $LR_BETA --relu_budget $RELU_BUDGET \
    --lr_snl $LR_SNL --epochs $SNL_EPOCHS  \
    --finetune_epochs $FINETUNE_EPOCHS --lr_finetune $LR_FINETUNE ;

#python3 test_freeze_alpha_network.py $DATASET $ARCH $OUTDIR \
#  $OUTDIR/snl_best_checkpoint_resnet18_in_cifar100_$RELU_BUDGET.pth.tar --block_type LearnableAlphaAndBetaNoSigmoid