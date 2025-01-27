export ARCH=resnet18_in
export DATASET=cifar100
export HIDDEN_DIM=10
export OUTDIR=/storage/jevnisek/snl/relu_autoencoder_multiple_layers_one_by_one/relu_autoencoder_multiple_layers_one_by_one/smooth-drelu-layer-$LAYER_INDEX-block-$BLOCK_INDEX-subblock-$SUBBLOCK_RELU_INDEX-hiddendim-$HIDDEN_DIM
export INITIAL_CHECKPOINT=/storage/jevnisek/snl/pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
export BETA_EPOCHS=100
export LR_BETA=0.1
export RELU_BUDGET=15000
export FREEZE_ALPHAS_AND_WEIGHTS=true
export SNL_EPOCHS=1
export LR_SNL=1e-7
export FINETUNE_EPOCHS=1
export LR_FINETUNE=1e-7
export LAYER_NAME='layer1[0].alpha1 layer1[0].alpha2 layer1[1].alpha1 layer1[1].alpha2 layer2[0].alpha1 layer2[0].alpha2 layer2[1].alpha1 layer2[1].alpha2 layer3[0].alpha1 layer3[0].alpha2 layer3[1].alpha1 layer3[1].alpha2 layer4[0].alpha1 layer4[0].alpha2 layer4[1].alpha1 layer4[1].alpha2'
export SIGMA_TYPE="smooth-drelu"
export LEARNING_RATE=0.1
