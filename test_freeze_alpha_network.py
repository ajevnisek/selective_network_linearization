# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 
"""
python3 test_freeze_alpha_network.py cifar100 resnet18_in snl_alpha_freeze_output snl_alpha_freeze_output/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar --block_type LearnableAlphaAndBetaNoSigmoid --beta_epochs 20 --relu_budget 15000
"""
import os
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import sys

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-5, type=float,
                    help='Lasso coefficient')
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--budegt_type', default='absolute', type=str, choices=['absolute', 'relative'])
parser.add_argument('--relu_budget', default=50000, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
parser.add_argument('--block_type', type=str, default='LearnableAlpha')
parser.add_argument('--num_of_neighbors', type=int, default=4)
parser.add_argument('--beta_epochs', type=int, default=5)
args = parser.parse_args()


if args.budegt_type == 'relative' and args.relu_budget > 1:
    print(f'Warning: relative budget type is used, but the relu budget is {args.relu_budget} > 1.')
    sys.exit(1)


def relu_counting(net, args):
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alphas' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    logfilename = os.path.join(args.outdir, args.logname)

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Dataset: {:}".format(args.dataset))
    log(logfilename, "Architecture: {:}".format(args.arch))
    log(logfilename, "ReLU Budget: {:}".format(args.relu_budget))
    log(logfilename, "Finetune Epochs: {:}".format(args.finetune_epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "Block Type: {:}".format(args.block_type))
    log(logfilename, "Num of Neighbours: {:}".format(args.num_of_neighbors if args.block_type != 'LearnableAlpha' else 0))

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)


    # Loading the base_classifier
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    """
    WARNING: I added strict=False here to handle the case that the base model does not contain parameters which we want
    to selectively learn.
    """
    base_classifier.load_state_dict(checkpoint['state_dict'], strict=False)

    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    print(original_acc)
    # if args.block_type in ['LearnableAlphaAndBetaNewAlgorithm', 'LearnableAlphaBetaGamma']:
    #     # new_lr = checkpoint['optimizer']['param_groups'][0]['lr']
    #     # log(logfilename, f"Changing learning rate from {args.lr} to {new_lr}")
    #     # args.lr = new_lr
    #     for layer in [f'layer{i}' for i in range(1, 4 + 1)]:
    #         for block_index in [0, 1]:
    #             for alpha_index in [1, 2]:
    #                 base_classifier.get_submodule(layer)[block_index].get_submodule(
    #                     f'alpha{alpha_index}').set_default_params_beta_and_gamma()

    base_classifier.eval()

    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    log(logfilename, "Model Test Accuracy: {:.5}".format(original_acc))
    relu_count = relu_counting(base_classifier, args)

    log(logfilename, "ReLU Count: {}".format(relu_count))
    import matplotlib.pyplot as plt


    root = args.outdir
    os.makedirs(root, exist_ok=True)
    for layer in [f"layer{i}" for i in range(1, 1+4)]:
        for idx in [0, 1]:
            for alpha in [1, 2]:
                os.makedirs(os.path.join(root, f'{layer}[{idx}].alpha{alpha}'), exist_ok=True)
                # betas shape: C x (H*W) x (H*W)
                # betas = base_classifier.layer1[0].alpha1.beta.cpu().detach()
                betas = base_classifier.get_submodule(layer)[idx].get_submodule(f'alpha{alpha}').beta.cpu().detach()
                # alphas shape: 1 x C x H x W
                # alphas = base_classifier.layer1[0].alpha1.alphas.cpu().detach()
                alphas = base_classifier.get_submodule(layer)[idx].get_submodule(f'alpha{alpha}').alphas.cpu().detach()
                # exec(f'alphas = base_classifier.{layer}[{idx}].alpha{alpha}.alphas.cpu().detach()')
                _, C, H, W = alphas.shape
                for channel in range(C):
                    if alphas[0][channel].sum() != 0:
                        plt.close('all')
                        plt.suptitle(f'ReLU layer: {layer}[{idx}].alpha{alpha}, channel: {channel:03d}')
                        plt.subplot(1, 3, 1)
                        plt.title('alphas')
                        plt.imshow(alphas[0][channel])
                        plt.colorbar()
                        plt.subplot(1, 3, 2)
                        plt.title('betas')
                        plt.imshow(betas[channel])
                        plt.colorbar()
                        plt.subplot(1, 3, 3)
                        plt.title('betas contribution')
                        plt.imshow(betas[channel].sum(axis=1).reshape(H, W))
                        plt.colorbar()
                        plt.tight_layout()
                        fig = plt.gcf()
                        fig.set_size_inches((12, 6))
                        plt.savefig(os.path.join(root, f'{layer}[{idx}].alpha{alpha}', f'{channel:03d}.png'))


if __name__ == "__main__":
    main()
