    # args.add_argument("snl_model_checkpoint_path", type=str, help='path to the checkpoint of the SNL model')
    # args.add_argument("teacher_model_checkpoint_path", type=str, help='path to the checkpoint of the teacher model')
    # args.add_argument("output_dir", type=str, help='the directory to which the final checkpoints will be saved')
    # args.add_argument("-r", "--reduce_threshold", type=float, default=0.05, help='the train set percentage drop threshold for reducing alphas')
    # args.add_argument("-l", "--step_reduce", type=int, default=100, help='the number of alphas to prune at a time')
    # args.add_argument("-e", "--random_tries", type=int, default=100, help='the maximum number of random prune attempts per cycle')
    # args.add_argument("-p", "--relu_target", type=int, default=15000, help='the target ReLU budget to arrive at')
    # args.add_argument("-a", "--arch", type=str, default="resnet18_in", help='the SNL-based architecture to use')
    # args.add_argument("-d", "--dataset", type=str, default="cifar100", help='the dataset to use')
    # args.add_argument("-b", "--batch_size", type=int, default=128, help='batch size for data loaders')
    # args.add_argument("-w", "--workers", type=int, default=4, help='number of workers to use for data loaders')
    # args.add_argument("-s", "--stride", type=int, default=1, help='the relevant stride for the chosen architecture')
    # args.add_argument("-t", "--threshold", type=float, default=0.01, help='the threshold for alphas')
    # args.add_argument("-k", "--block_type", type=str, default="LearnableAlpha", help='the block type to use in the SNL routine')
    # args.add_argument("-n", "--num_of_neighbors", type=int, default=4, help='where applicable, the number of neighbors for a ReLU pruning routine')


SNL_MODEL_CHECKPOINT_PATH=checkpoints/resnet18_cifar100_30000.pth
TEACHER_MODEL_CHECKPOINT_PATH=checkpoints/resnet18_cifar100.pth
OUTPUT_DIR=checkpoints/docker_sandbox/optimize_and_reduce

REDUCE_THRESHOLD=0.05
STEP_REDUCE=100
RANDOM_TRIES=1
RELU_TARGET=15000


