export ARCH=resnet18_in
export DATASET=cifar100
export OUTDIR=/local_code/alternating_alphas_and_betas/freeze_only_alphas_little_noise_local/
export INITIAL_CHECKPOINT=/local_code/checkpoints/snl-15k.pth.tar
export BETA_EPOCHS=100
export LR_BETA=0.001
export NOISE_INIT_FOR_BETAS=0.01
export RELU_BUDGET=15000
export FREEZE_ALPHAS_AND_WEIGHTS=true
export SNL_EPOCHS=1
export LR_SNL=0.001
export FINETUNE_EPOCHS=1
export LR_FINETUNE=0.001