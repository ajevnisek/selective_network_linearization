export ARCH=resnet18_in
export DATASET=cifar100
export OUTDIR=/local_code/relu_autoencoder_one_channel/relu_autoencoder_one_channel_local/
export INITIAL_CHECKPOINT=/local_code/checkpoints/resnet18_cifar100.pth
export BETA_EPOCHS=100
export LR_BETA=0.1
export RELU_BUDGET=15000
export FREEZE_ALPHAS_AND_WEIGHTS=true
export SNL_EPOCHS=1
export LR_SNL=1e-7
export FINETUNE_EPOCHS=1
export LR_FINETUNE=1e-7
export LAYER_INDEX=1
export BLOCK_INDEX=1
export SUBBLOCK_RELU_INDEX=2