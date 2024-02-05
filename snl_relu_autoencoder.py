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
parser.add_argument('--layer_index', type=int, default=5)
parser.add_argument('--channel_index', type=int, default=5)
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
    out_channels = base_classifier.get_submodule('layer1')[1].alpha2.alphas.shape[1]
    feature_size = base_classifier.get_submodule('layer1')[1].alpha2.alphas.shape[-1]
    from archs_unstructured.cifar_resnet import LearnableAlphaAndBetaNoSigmoidWithGammaDiffFromEye
    base_classifier.get_submodule('layer1')[1].alpha2 = LearnableAlphaAndBetaNoSigmoidWithGammaDiffFromEye(out_channels, feature_size).to(device)

    # LearnableAlphaAndBetaNoSigmoid
    checkpoint = torch.load(args.savedir, map_location=device)
    """
    WARNING: I added strict=False here to handle the case that the base model does not contain parameters which we want
    to selectively learn.
    """
    base_classifier.load_state_dict(checkpoint['state_dict'], strict=False)

    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    print(original_acc)
    if args.block_type in ['LearnableAlphaAndBetaNewAlgorithm', 'LearnableAlphaBetaGamma']:
        # new_lr = checkpoint['optimizer']['param_groups'][0]['lr']
        # log(logfilename, f"Changing learning rate from {args.lr} to {new_lr}")
        # args.lr = new_lr
        for layer in [f'layer{i}' for i in range(1, 4 + 1)]:
            for block_index in [0, 1]:
                for alpha_index in [1, 2]:
                    base_classifier.get_submodule(layer)[block_index].get_submodule(
                        f'alpha{alpha_index}').set_default_params_beta_and_gamma()

    base_classifier.eval()

    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))
    # import ipdb; ipdb.set_trace()
    # from collections import OrderedDict
    # channel_id_to_num_activated_relus = OrderedDict({})
    # layer_index = 0
    # for name, param in base_classifier.named_parameters():
    #     if 'alphas' in name:
    #         channel_id_to_num_activated_relus[f"layer#{layer_index:02d}"] = OrderedDict({})
    #         C, H, W = param.squeeze(0).shape
    #         channel_id_to_num_activated_relus[f"layer#{layer_index:02d}"]['NUM_CAHNNELS'] = f"{C}"
    #         channel_id_to_num_activated_relus[f"layer#{layer_index:02d}"]['CHANNEL_SHAPE'] = f"{H}x{W}"
    #         for channel_index, channel in enumerate(param.squeeze(0)):
    #             num_activated_relus = (channel > 0).float().sum()
    #             if num_activated_relus > 0:
    #                 channel_id_to_num_activated_relus[f"layer#{layer_index:02d}"][f"channel#{channel_index:03d}"] = int(num_activated_relus)
    #                 print(f"(layer_index, channel): ({layer_index}, {channel_index}) -> {int(num_activated_relus)}")
    #         layer_index += 1
    # print(channel_id_to_num_activated_relus)
    # import ipdb;
    # ipdb.set_trace()
    # import json
    # with open('snl-15k-channel-id-to-num-activated-relus.json', 'w') as f:
    #     json.dump(channel_id_to_num_activated_relus, f, indent=2)
    # Creating a fresh copy of network not affecting the original network.
    net = copy.deepcopy(base_classifier)
    net = net.to(device)
    layer_index = 0



    relu_count = relu_counting(net, args)

    log(logfilename, "Original ReLU Count: {}".format(relu_count))

    # Line 12: Finetuing the network
    finetune_epoch = args.finetune_epochs

    # optimizer = SGD(net.parameters(),
    #                 lr=args.lr_beta, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = SGD([
                {'params': [p for (name, p) in net.named_parameters() if name != 'layer1.1.alpha2.beta']},
                {'params': net.layer1[1].alpha2.beta, 'lr': args.lr_beta}
    ],
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
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
    for name, param in net.named_parameters():
        if 'beta' in name:
            param.requires_grad = True
            print(name)
    # for name, param in net.named_parameters():
    #     if 'beta' not in name:
    #         param.requires_grad = False
    best_top1 = 0
    for epoch in range(args.beta_epochs):
        train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch, device)
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(args.outdir,  f'betas_viz/{epoch}/alphas'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir,  f'betas_viz/{epoch}/betas'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, f'betas_viz/{epoch}/betas_images'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir,  f'betas_viz/{epoch}/betas_contrib'), exist_ok=True)
        for idx, channel in enumerate(net.layer1[1].alpha2.beta):
            import matplotlib.pyplot as plt
            if (channel.cpu()).sum().item() != 0:
                plt.close('all')
                plt.title(f'alphas\nchannel# {idx:02d}')
                plt.imshow(net.layer1[1].alpha2.alphas[0, idx].detach().cpu())
                plt.colorbar()
                plt.savefig(os.path.join(args.outdir, f'betas_viz/{epoch}/alphas/{idx}.png'))
                plt.close('all')
                plt.title(f'beta difference from eye matrix\nchannel# {idx:02d}')
                plt.imshow((channel.cpu().detach() - torch.eye(channel.shape[0])).abs())
                plt.colorbar()
                plt.savefig(os.path.join(args.outdir, f'betas_viz/{epoch}/betas/{idx}.png'))
                plt.close('all')
                plt.title(f'betas contrib to other relus\nchannel# {idx:02d}')
                plt.imshow(channel.cpu().sum(axis=0).reshape(32, 32).detach() - torch.diagonal(channel.cpu()).reshape(32,  32).detach())
                plt.colorbar()
                plt.savefig(os.path.join(args.outdir, f'betas_viz/{epoch}/betas_contrib/{idx}.png'))
                os.makedirs(os.path.join(args.outdir, f'betas_viz/{epoch}/betas_images/channel{idx}/'), exist_ok=True)
            for prototype_idx in range(channel.shape[0]):
                if (channel.cpu().detach()[prototype_idx].reshape(32, 32)).sum() != 0:
                    plt.close('all')
                    is_it_an_snl_prototype =  net.layer1[1].alpha2.alphas[0, idx, prototype_idx // 32, prototype_idx % 32]
                    plt.title(f'beta for prototype index: {prototype_idx:04d}\nchannel# {idx:02d}')
                    plt.imshow(channel.cpu().detach()[prototype_idx].reshape(32, 32))
                    plt.colorbar()
                    plt.savefig(os.path.join(args.outdir, f'betas_viz/{epoch}/betas_images/channel{idx}/{prototype_idx}.png'))

        test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
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
            }, os.path.join(args.outdir, f'betas_network_{args.arch}_{args.dataset}_{args.relu_budget}.pth.tar'))

    print("Final best Prec@1 = {}%".format(best_top1))
    log(logfilename, "After Beta optimization, Final best Prec@1 = {}%".format(best_top1))
    log(logfilename, "After Beta optimization, ReLU Count: {}".format(relu_count))


    # '''
    # This section optimizes the alpha parameters using snl.
    # '''
    # # Alpha is the masking parameters initialized to 1. Enabling the grad.
    # for name, param in net.named_parameters():
    #     if 'alpha' in name:
    #         param.requires_grad = True
    #
    # for name, param in net.named_parameters():
    #     if 'beta' in name:
    #         param.requires_grad = False
    #
    #
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = Adam(net.parameters(), lr=args.lr_snl)
    #
    # # counting number of ReLU.
    # total = relu_counting(net, args)
    # if args.budegt_type == 'relative':
    #     args.relu_budget = int(total * args.relu_budget)
    #
    # # Corresponds to Line 4-9
    # lowest_relu_count, relu_count = total, total
    # for epoch in range(args.epochs):
    #
    #     # Simultaneous tarining of w and alpha with KD loss.
    #     train_loss = mask_train_kd_unstructured(train_loader, net, base_classifier, criterion, optimizer,
    #                             epoch, device, alpha=args.alpha, display=False)
    #     acc = model_inference(net, test_loader, device, display=False)
    #
    #     # counting ReLU in the neural network by using threshold.
    #     relu_count = relu_counting(net, args)
    #     log(logfilename, 'Epochs: {}\t'
    #           'Test Acc: {}\t'
    #           'Relu Count: {}\t'
    #           'Alpha: {:.6f}\t'.format(
    #               epoch, acc, relu_count, args.alpha
    #           )
    #           )
    #
    #     if relu_count < lowest_relu_count:
    #         lowest_relu_count = relu_count
    #
    #     elif relu_count >= lowest_relu_count and epoch >= 5:
    #         args.alpha *= 1.1
    #
    #     if relu_count <= args.relu_budget:
    #         print("Current epochs breaking loop at {:}".format(epoch))
    #         break
    #
    # log(logfilename, "After SNL Algorithm, the current ReLU Count: {}, rel. count:{}".format(relu_count, relu_count/total))
    # log(logfilename, "Saving Model before fine-tuning to checkpoint...")
    # torch.save({
    #     'arch': args.arch,
    #     'state_dict': net.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }, os.path.join(args.outdir, f'snl_before_finetuning_checkpoint_{args.arch}_{args.dataset}_{args.relu_budget}.pth.tar'))
    # # Line 11: Threshold and freeze alpha
    # for name, param in net.named_parameters():
    #     if 'beta' in name:
    #         param.requires_grad = False
    #     if 'alpha' in name:
    #         boolean_list = param.data > args.threshold
    #         param.data = boolean_list.float()
    #         param.requires_grad = False
    #
    #
    # # Line 12: Finetuing the network
    # finetune_epoch = args.finetune_epochs
    #
    # optimizer = SGD(net.parameters(), lr=args.lr_finetune, momentum=args.momentum, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss().to(device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)
    #
    # print("Finetuning the model")
    # log(logfilename, "Finetuning the model")
    #
    # best_top1 = 0
    # for epoch in range(finetune_epoch):
    #     train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch, device)
    #     test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
    #     scheduler.step()
    #
    #     if best_top1 < test_top1:
    #         best_top1 = test_top1
    #         is_best = True
    #     else:
    #         is_best = False
    #
    #     if is_best:
    #         torch.save({
    #                 'arch': args.arch,
    #                 'state_dict': net.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #         }, os.path.join(args.outdir, f'snl_best_checkpoint_{args.arch}_{args.dataset}_{args.relu_budget}.pth.tar'))
    #
    # print("Final best Prec@1 = {}%".format(best_top1))
    # log(logfilename, "Final best Prec@1 = {}%".format(best_top1))
    #

if __name__ == "__main__":
    main()
