export DATASET=cifar100
export RELU_BUDGET=9000
export MODELDIR=/local_code/snl_output/cifar100/30000/snl_best_checkpoint_resnet18_in_cifar100_30000.pth.tar
export SAVEDIR=/storage/jevnisek/snl/outputs/snl-abc-squash/original_cifar100/$RELU_BUDGET_starts_from_30k/resnet18_in/
export OUTDIR=/storage/jevnisek/snl/outputs/snl-abc-squash/original_cifar100/$RELU_BUDGET_starts_from_30k/resnet18_in/
export ARCH=resnet18_in
export FINETUNE_EPOCH=100
export EPOCHS=2000
export LOGNAME=resnet18_in_unstructured_$RELU_BUDGET.txt
export LR=1e-3
export THRESHOLD=1e-2
export ALPHA=1e-5
export BATCH=128

