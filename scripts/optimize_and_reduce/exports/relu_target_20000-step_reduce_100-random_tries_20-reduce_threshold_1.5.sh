
SNL_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100_30000.pth
TEACHER_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100.pth


REDUCE_THRESHOLD=1.5
STEP_REDUCE=100
RANDOM_TRIES=20
RELU_TARGET=20000


OUTPUT_DIR=/storage/vladi/snl/optimize_and_reduce/outputs/RELU_TARGET_$RELU_TARGET-REDUCE_THRESHOLD_$REDUCE_THRESHOLD-RANDOM_TRIES_$RANDOM_TRIES-STEP_REDUCE_$STEP_REDUCE
