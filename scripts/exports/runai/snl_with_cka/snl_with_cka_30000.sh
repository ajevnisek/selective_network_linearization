export DATASET=cifar100
export RELU_BUDGET=30000
export SAVEDIR=/storage/jevnisek/snl_with_cka/pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
export OUTDIR=/storage/jevnisek/snl_with_cka/outputs/snl_with_cka/cifar100/${RELU_BUDGET}/resnet18_in/
export ARCH=resnet18_in
export FINETUNE_EPOCH=100
export EPOCHS=2000
export LOGNAME=snl_with_cka_resnet18_in_unstructured_$RELU_BUDGET.txt
export LR=1e-3
export THRESHOLD=1e-2
export ALPHA=1e-5
export ALPHA_CKA=0.5
export BATCH=128

