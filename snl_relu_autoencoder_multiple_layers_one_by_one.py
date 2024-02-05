# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 

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
from archs_unstructured.cifar_resnet import ReLUAutoEncoder


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=2000, type=int, help='snl epochs')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_beta', default=0.1, type=float,
                    help='initial learning rate for betas optimization', dest='lr_beta')
parser.add_argument('--lr_snl', default=1e-3, type=float,
                    help='initial learning rate for snl optimization', dest='lr_snl')
parser.add_argument('--lr_finetune', default=1e-3, type=float,
                    help='initial learning rate for final finetuning', dest='lr_finetune')
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
parser.add_argument('--layer_name', type=str, nargs='+', default='layer1[1].alpha2', choices=[
    f"layer{layer_index}[{block_index}].alpha{relu_index}"
    for layer_index in range(1, 1+4)
    for block_index in range(2)
    for relu_index in range(1, 1 + 2)
])
parser.add_argument('--hidden_dim', type=int, default=10)
parser.add_argument('--sigma_type', type=str, default='drelu', choices=['relu', 'drelu', 'smooth-drelu'])
# hidden_dim
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
    for item in vars(args).items():
        log(logfilename,f"{item[0].capitalize()}, {item[1]}")
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
    # out_channels = base_classifier.get_submodule('layer1')[1].alpha2.alphas.shape[1]
    checkpoint = torch.load(args.savedir, map_location=device)
    """
    WARNING: I added strict=False here to handle the case that the base model does not contain parameters which we want
    to selectively learn.
    """
    base_classifier.load_state_dict(checkpoint['state_dict'], strict=False)

    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    print(original_acc)
    base_classifier.eval()

    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))
    relu_count = relu_counting(base_classifier, args)

    log(logfilename, "Original ReLU Count: {}".format(relu_count))

    # Creating a fresh copy of network not affecting the original network.
    net = copy.deepcopy(base_classifier)
    net = net.to(device)
    for layername_idx, layername in enumerate(args.layer_name):
        log(logfilename, f"[Iteration# {layername_idx}] Replacing {layername}...")
        layer_index = int(layername.split('layer')[-1].split('[')[0])
        block_index = int(layername.split('[')[-1].split(']')[0])
        subblock_relu_index = int(layername.split('alpha')[-1])
        print(layername, layer_index, block_index, subblock_relu_index)
        exec(
            f'out_channels =net.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas.shape[1]')
        exec(
            f'feature_size =net.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index}.alphas.shape[-1]')
        exec(
            f'net.get_submodule("layer{layer_index}")[{block_index}].alpha{subblock_relu_index} = ReLUAutoEncoder(out_channels, feature_size, hidden_dim=args.hidden_dim, sigma_type=args.sigma_type).to(device)')

        relu_count = relu_counting(net, args)

        log(logfilename, "After AutoEncoder ReLU Count: {}".format(relu_count))

        # Line 12: Finetuing the network
        finetune_epoch = args.finetune_epochs

        optimizer = SGD(net.parameters(),
                        lr=args.lr_beta, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().to(device)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.beta_epochs // 2, 3 * args.beta_epochs // 4], last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

        print("Finetuning the model")
        log(logfilename, "Finetuning the model")

        '''
        This section optimizes the beta parameters.
        '''
        # Alpha is the masking parameters initialized to 1. Enabling the grad.
        for name, param in net.named_parameters():
            if 'alphas' in name:
                param.requires_grad = False
        best_top1 = 0
        for epoch in range(args.beta_epochs):
            train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch, device)
            # train_loss, train_top1, train_top5 = train(train_loader, net, criterion, optimizer, epoch, device)
            log(logfilename,
                'Epoch [{epoch:03d}]]\t'
                'Train Loss  ({loss:.4f})\t'
                'Train Acc@1 ({top1:.3f})\t'
                'Train Acc@5 ({top5:.3f})'.format(epoch=epoch,
                    loss=train_loss, top1=train_top1, top5=train_top5))
            test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
            log(logfilename,
                'Epoch [{epoch:03d}]]\t'
                'Test Loss  ({loss:.4f})\t'
                'Test Acc@1 ({top1:.3f})\t'
                'Test Acc@5 ({top5:.3f})'.format(epoch=epoch,
                                                     loss=test_loss, top1=test_top1, top5=test_top5))
            scheduler.step()

            if best_top1 < test_top1:
                best_top1 = test_top1
                is_best = True
            else:
                is_best = False

            if is_best:
                print(f'saving checkpoint to :{args.outdir}, acc: {best_top1}')
                torch.save({
                        'arch': args.arch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, f'iter_{layername_idx}_replacing_{layername}_betas_network_{args.arch}_{args.dataset}.pth.tar'))

        print("Final best Prec@1 = {}%".format(best_top1))
        log(logfilename, "After Beta optimization, Final best Prec@1 = {}%".format(best_top1))
        log(logfilename, "After Beta optimization, ReLU Count: {}".format(relu_count))


if __name__ == "__main__":
    main()
