export ARCH=resnet18_in
export DATASET=cifar100
#export OUTDIR=/storage/jevnisek/snl/alternating_alphas_and_betas/freeze_alphas_and_weights_ones_init/
export OUTDIR=/local_code/alternating_alphas_and_betas/freeze_alphas_and_weights_ones_init/
#export INITIAL_CHECKPOINT=/storage/jevnisek/snl/outputs/cifar100/15000/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar
export INITIAL_CHECKPOINT=/local_code/checkpoints/snl-15k.pth.tar
export BETA_EPOCHS=120
export LR_BETA=0.01
export NOISE_INIT_FOR_BETAS=0.01
export RELU_BUDGET=15000
export FREEZE_ALPHAS_AND_WEIGHTS=true
export SNL_EPOCHS=1
export LR_SNL=0.001
export FINETUNE_EPOCHS=1
export LR_FINETUNE=0.001