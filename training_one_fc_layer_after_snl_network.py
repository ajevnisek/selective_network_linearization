import os

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
                      momentum=0.9, num_of_neighbors=4,
                      outdir='./training_one_fc_layer_after_snl/cifar100/original/resnet18_in/15000/',
                      print_freq=100, relu_budget=15000,
                      savedir='./checkpoints/resnet18_cifar100.pth', stride=1, threshold=0.01,
                      weight_decay=0.0005, workers=4)

os.makedirs(my_args.outdir, exist_ok=True)

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
# checkpoint = torch.load('snl_output/cifar100/15000/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar', map_location=device)
checkpoint = torch.load('checkpoints/resnet18_cifar100_30000.pth', map_location=device)
base_classifier.load_state_dict(checkpoint['state_dict'])
base_classifier.eval()

original_acc = model_inference(base_classifier, test_loader,
                               device, display=True)
print("Original Model Test Accuracy: {:.5}".format(original_acc))
import matplotlib.pyplot as plt
from collections import OrderedDict
layer_to_relu_budget = OrderedDict({})
for (name, param) in base_classifier.named_parameters():
    if 'alphas' in name:
        curr_budget = (param.data > my_args.threshold).float().sum().item()
        print(name, curr_budget)
        layer_to_relu_budget[name] = curr_budget

normalized_cumsum = torch.tensor(list(layer_to_relu_budget.values())).cumsum(0) / sum(layer_to_relu_budget.values()) * 100.0
plt.close('all')
plt.title('ReLU budget CumSum Normalized')
plt.plot(normalized_cumsum, '-x', linewidth=3, label='ReLU budget cumsum')
plt.plot(range(len(normalized_cumsum)), [50.0] * len(normalized_cumsum), '--k', linewidth=3, label='50%')
plt.legend()
plt.xticks(range(len(normalized_cumsum)), [k.split('.alphas')[0] for k in layer_to_relu_budget.keys()], rotation='vertical')
plt.ylabel('cumulative percentage of SNL prototypes')
plt.grid(True)
plt.tight_layout()
plt.savefig('output/relu_budget_cumulative_normalized.png')
new_classifier = get_architecture('resnet18_in_with_exit_at_layer3_relu_0', my_args.dataset, device, my_args)
new_classifier.load_state_dict(checkpoint['state_dict'], strict=False)


net = copy.deepcopy(new_classifier)
net = net.to(device)

# Line 11: Threshold and freeze alpha
for name, param in net.named_parameters():
    if 'linear' not in name:
        boolean_list = param.data > my_args.threshold
        param.data = boolean_list.float()
        param.requires_grad = False
logfilename = os.path.join(my_args.outdir, my_args.logname)

# Line 12: Finetuing the network
finetune_epoch = my_args.finetune_epochs
# finetune_epoch = 2

optimizer = Adam(net.parameters(), lr=1e-3,
                 # momentum=my_args.momentum, weight_decay=my_args.weight_decay
                 )
criterion = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

print("Finetuning the model")
log(logfilename, "Finetuning the model")

best_top1 = 0
for epoch in range(finetune_epoch):
    train_loss, train_top1, train_top5 = train_kd(train_loader, net, base_classifier, optimizer, criterion, epoch,
                                                  device)
    test_loss, test_top1, test_top5 = test(test_loader, net, criterion, device, 100, display=True)
    scheduler.step()

    if best_top1 < test_top1:
        best_top1 = test_top1
        is_best = True
    else:
        is_best = False

    if is_best:
        torch.save({
            'arch': my_args.arch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(my_args.outdir, f'snl_best_checkpoint_{my_args.arch}_{my_args.dataset}_{my_args.relu_budget}.pth.tar'))

print("Final best Prec@1 = {}%".format(best_top1))
log(logfilename, "Final best Prec@1 = {}%".format(best_top1))
