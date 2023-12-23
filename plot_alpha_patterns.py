from architectures_unstructured import ARCHITECTURES, get_architecture
import matplotlib.pyplot as plt
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


class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


device = torch.device("cuda")
my_args = MyNamespace(alpha=1e-05, arch='resnet18_in', batch=128, block_type='LearnableAlpha',
                      budegt_type='absolute', dataset='cifar100', epochs=2000, finetune_epochs=100,
                      gamma=0.1, gpu=0, logname='resnet18_in_unstructured_.txt', lr=0.001, lr_step_size=30,
                      momentum=0.9, num_of_neighbors=4, outdir='./activations_cache/cifar100/original/resnet18_in/',
                      print_freq=100, relu_budget=15000,
                      savedir='./checkpoints/resnet18_cifar100.pth', stride=1, threshold=0.01,
                      weight_decay=0.0005, workers=4)

train_dataset = get_dataset(my_args.dataset, 'train')
test_dataset = get_dataset(my_args.dataset, 'test')
pin_memory = (my_args.dataset == "imagenet")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=my_args.batch,
                          num_workers=my_args.workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=my_args.batch,
                         num_workers=my_args.workers, pin_memory=pin_memory)

# Loading the base_classifier
base_classifier = get_architecture(my_args.arch, my_args.dataset, device, my_args)
checkpoint = torch.load(my_args.savedir, map_location=device)
root = 'snl_output/cifar100/15000/resnet18_in'
checkpoint = torch.load('snl_output/cifar100/15000/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar', map_location=device)
base_classifier.load_state_dict(checkpoint['state_dict'])
base_classifier.eval()

original_acc = model_inference(base_classifier, test_loader,
                               device, display=True)
print("Original Model Test Accuracy: {:.5}".format(original_acc))
alphas_patterns = os.path.join(root, 'alphas_patterns')

from math import sqrt, ceil
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    for block in [0, 1]:
        for relu_idx in [1, 2]:
            plt.close('all')
            exec(f'N_squared = base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[0].shape[0]')
            N = ceil(sqrt(N_squared))
            print(N)
            for i in range(N):
                for j in range(N):
                    plt.subplot(N, N, N*i + j + 1)
                    plt.xticks([]);
                    plt.yticks([])
                    # plt.title(f'channel #{N * i + j:02d}')
                    try:
                        exec(f'what_to_show = (base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[0][N * i + j] > my_args.threshold).float().cpu()')
                        plt.imshow(what_to_show)
                    except:
                        continue
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches((10, 10))
            plt.suptitle(f'{layer_name}_block{block}_relu{relu_idx}')
            plt.savefig(os.path.join(alphas_patterns, f'{layer_name}_block{block}_relu{relu_idx}.png'))
            plt.close('all')

for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    for block in [0, 1]:
        for relu_idx in [1, 2]:
            plt.close('all')
            exec(f'N_squared = base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[0].shape[0]')
            N = ceil(sqrt(N_squared))
            channels_to_show = []
            for i in range(N):
                for j in range(N):
                    try:
                        exec(
                            f'what_to_show = (base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[0][N * i + j] > my_args.threshold).float().cpu()')
                        if what_to_show.sum() > 0:
                            channels_to_show.append(N * i + j)
                    except:
                        continue
            N_squared = len(channels_to_show)
            N = ceil(sqrt(N_squared))
            for i in range(N):
                for j in range(N):
                    plt.subplot(N, N, N * i + j + 1)
                    plt.xticks([])
                    plt.yticks([])
                    try:
                        channel = channels_to_show[N * i + j]
                        exec(
                            f'what_to_show = (base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[0][channel] > my_args.threshold).float().cpu()')
                        plt.imshow(what_to_show)
                        plt.title(f'channel #{channel:02d}')
                        plt.colorbar()
                    except:
                        continue
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches((10, 10))
            plt.suptitle(f'{layer_name}_block{block}_relu{relu_idx} only set channels')
            plt.savefig(os.path.join(alphas_patterns, f'{layer_name}_block{block}_relu{relu_idx}_only_set_channels.png'))
# plt.show()
