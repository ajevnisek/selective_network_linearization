export DATASET=cifar100-new-split
export RELU_BUDGET=9000
export SAVEDIR=/local_code/pretrained_models/cifar100-new-split/resnet18_in/best_checkpoint.pth.tar
export OUTDIR=/local_code/docker_outputs/cifar100-new-split/9000/resnet18_in/
export ARCH=resnet18_in
export FINETUNE_EPOCH=100
export EPOCHS=2000
export LOGNAME=resnet18_in_unstructured_$RELU_BUDGET.txt
export LR=1e-3
export THRESHOLD=1e-2
export ALPHA=1e-5
export BATCH=128

