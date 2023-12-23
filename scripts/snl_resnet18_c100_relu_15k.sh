DATASET=cifar100
ARCH=resnet18_in
SAVEDIR=./snl_output2/cifar100/$ARCH/
MODELDIR=./checkpoints/resnet18_cifar100.pth
RELU_BUDGET=15000
FINETUNE_EPOCH=100
EPOCHS=2000
LOGNAME=resnet18_in_unstructured_$RELU_BUDGET_betas.txt
SAVEDIR=./snl_output2/cifar100/$RELU_BUDGET/$ARCH/
LR=1e-3
THRESHOLD=1e-2
ALPHA=1e-5
BATCH=128

python3 snl_finetune_unstructured.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH 

