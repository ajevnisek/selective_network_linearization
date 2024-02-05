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
                      outdir='./retraining_after_alpha_thresholding/cifar100/original/resnet18_in/',
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

print("Replace alphas with {0, 1}")
accs = [original_acc]
name = ['snl-optimization']
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    for block in [0, 1]:
        for relu_idx in [1, 2]:
            print('-' * 20)
            print(f"Replacing: {layer_name}, block#{block} and relu{relu_idx}: ")
            exec(f'mean_ = base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas > 0].mean().item()')
            exec(
                f'std_ = base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas > 0].std().item()')
            print(f"Mean: {mean_:.3f}, Std: {std_:.3f}")
            exec(f'base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas.requires_grad = False')
            exec(f'shape_ = base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas > 0].shape')
            exec(f'base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas[base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas > 0] = nn.Parameter(torch.ones(shape_),  requires_grad=False).cuda()')
            original_acc = model_inference(base_classifier, test_loader,
                                           device, display=True)
            print("Original Model Test Accuracy: {:.5}".format(original_acc))
            accs.append(original_acc)
            name.append(f'{layer_name}[{block}].alpha{relu_idx}')
plt.close('all')
plt.title('accuracy vs layer names')
plt.plot(accs, '-x', linewidth=3)
plt.ylabel('accuracy [%]')
plt.xticks(range(0, len(name)), name, rotation='vertical')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(root, 'visualizations', 'accuracy_vs_layer_names_before_second_finetuning.png'))


net = copy.deepcopy(base_classifier)
net = net.to(device)

# Line 11: Threshold and freeze alpha
for name, param in net.named_parameters():
    if 'alpha' in name:
        boolean_list = param.data > my_args.threshold
        param.data = boolean_list.float()
        param.requires_grad = False
logfilename = os.path.join(my_args.outdir, my_args.logname)

# Line 12: Finetuing the network
finetune_epoch = my_args.finetune_epochs
# finetune_epoch = 2

optimizer = SGD(net.parameters(), lr=1e-3, momentum=my_args.momentum, weight_decay=my_args.weight_decay)
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
