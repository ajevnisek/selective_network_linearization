export ARCH=resnet18_in
export DATASET=cifar100
export LAYER_INDEX=1
export BLOCK_INDEX=1
export SUBBLOCK_RELU_INDEX=2
export HIDDEN_DIM=10
export OUTDIR=/storage/jevnisek/relu_autoencoder_one_channel/relu_autoencoder_one_channel_local/smooth-drelu-layer-$LAYER_INDEX-block-$BLOCK_INDEX-subblock-$SUBBLOCK_RELU_INDEX-hiddendim-$HIDDEN_DIM
export INITIAL_CHECKPOINT=//storage/jevnisek/snl/pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
export BETA_EPOCHS=100
export LR_BETA=0.1
export RELU_BUDGET=15000
export FREEZE_ALPHAS_AND_WEIGHTS=true
export SNL_EPOCHS=1
export LR_SNL=1e-7
export FINETUNE_EPOCHS=1
export LR_FINETUNE=1e-7

export SIGMA_TYPE='smooth-drelu'
