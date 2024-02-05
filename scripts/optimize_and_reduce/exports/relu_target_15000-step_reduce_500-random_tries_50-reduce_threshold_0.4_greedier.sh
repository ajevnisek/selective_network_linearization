
SNL_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100_30000.pth
TEACHER_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100.pth


REDUCE_THRESHOLD=0.4
STEP_REDUCE=500
RANDOM_TRIES=50
RELU_TARGET=15000


OUTPUT_DIR=/storage/vladi/snl/optimize_and_reduce/outputs/RELU_TARGET_$RELU_TARGET-REDUCE_THRESHOLD_$REDUCE_THRESHOLD-RANDOM_TRIES_$RANDOM_TRIES-STEP_REDUCE_$STEP_REDUCE