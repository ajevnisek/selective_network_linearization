#DATASET=cifar100
ARCH=resnet18_in
#RELU_BUDGET=100000
FINETUNE_EPOCH=100
EPOCHS=2000
#MODELDIR=./pretrained_models/cifar100/resnet18_in/best_checkpoint.pth.tar
LOGNAME=resnet18_in_unstructured_with_betas_$RELU_BUDGET.txt
#SAVEDIR=./snl_output/cifar100/LearnableAlphaAndBeta/$RELU_BUDGET/$ARCH/
LR=1e-3
THRESHOLD=1e-2
ALPHA=1e-5
BATCH=128
BLOCK_TYPE=LearnableAlphaAndBeta
python3 snl_finetune_unstructured_additional_parameters.py "$DATASET" "$ARCH" "$SAVEDIR" "$MODELDIR" --relu_budget $RELU_BUDGET --alpha $ALPHA --lr $LR --threshold $THRESHOLD --batch $BATCH --logname "$LOGNAME" --finetune_epochs $FINETUNE_EPOCH --block_type "$BLOCK_TYPE"

