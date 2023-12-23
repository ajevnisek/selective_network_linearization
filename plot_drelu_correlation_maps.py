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
checkpoint = torch.load('./retraining_after_alpha_thresholding/cifar100/original/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar', map_location=device)
base_classifier.load_state_dict(checkpoint['state_dict'])
base_classifier.eval()

original_acc = model_inference(base_classifier, test_loader,
                               device, display=True)
print("Original Model Test Accuracy: {:.5}".format(original_acc))


activation_out = {}
def getActivationOutput(name):
    def hook(model, input, output):
        if name not in activation_out:
            activation_out[name] = [output.detach().cpu()]
        else:
            activation_out[name].append(output.detach().cpu())
    return hook


activation_in = {}
def getActivationInput(name):
    def hook(model, input, output):
        if name not in activation_in:
            activation_in[name] = [input[0].detach().cpu()]
        else:
            activation_in[name].append(input[0].detach().cpu())
    return hook

activation_in = {}
activation_out = {}
hook_out = base_classifier.layer1[0].alpha1.register_forward_hook(getActivationOutput('layer1_block0_relu1'))
hook_in = base_classifier.layer1[0].alpha1.register_forward_hook(getActivationInput('layer1_block0_relu1'))

original_acc = model_inference(base_classifier, test_loader,
                               device, display=True)
print("Original Model Test Accuracy: {:.5}".format(original_acc))
print(f"activated channels: {np.unique(np.where((base_classifier.layer1[0].alpha1.alphas > my_args.threshold).cpu())[1])}")

layer1_block0_relu1_out = torch.cat(activation_out['layer1_block0_relu1'], 0)
layer1_block0_relu1_out_channel_28 = layer1_block0_relu1_out[..., 28, :, :]
print((layer1_block0_relu1_out_channel_28[..., 0, 10] < 0).any())
hook_out.remove()

layer1_block0_relu1_in = torch.cat(activation_in['layer1_block0_relu1'], 0)
layer1_block0_relu1_in_channel_28 = layer1_block0_relu1_in[..., 28, :, :]
print((layer1_block0_relu1_in_channel_28[..., 0, 10] < 0).any())
hook_in.remove()
from sklearn.preprocessing import normalize
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    for block in [0, 1]:
        for relu_idx in [1, 2]:
            activation_in = {}
            activation_out = {}
            exec(f'hook_out = base_classifier.{layer_name}[{block}].alpha{relu_idx}.register_forward_hook(getActivationOutput("{layer_name}_block{block}_relu{relu_idx}"))')
            exec(
                f'hook_in = base_classifier.{layer_name}[{block}].alpha{relu_idx}.register_forward_hook(getActivationInput("{layer_name}_block{block}_relu{relu_idx}"))')
            original_acc = model_inference(base_classifier, test_loader,
                                           device, display=True)
            exec(f'indices = np.where((base_classifier.{layer_name}[{block}].alpha{relu_idx}.alphas > my_args.threshold).cpu())')
            curr_layer_in = torch.cat(activation_in[f"{layer_name}_block{block}_relu{relu_idx}"], 0)
            drelu = curr_layer_in[..., indices[1], indices[2], indices[3]]
            drelu = 2 * drelu - 1
            cosine_similarity = normalize(drelu.T) @ normalize(drelu.T).T
            plt.close('all')
            plt.title(f'cosine similarity (higher = better)\n{layer_name}[{block}].alpha{relu_idx}')
            plt.xlabel('prototype in layer')
            plt.ylabel('prototype in layer')
            plt.imshow(cosine_similarity)
            plt.colorbar()
            prototypes_indices = [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in
                                  zip(indices[1], indices[2], indices[3])]
            plt.xticks(range(len(prototypes_indices)), prototypes_indices, rotation='vertical', fontsize='small')
            plt.yticks(range(len(prototypes_indices)), prototypes_indices, rotation='horizontal', fontsize='small')
            fig = plt.gcf()
            fig.set_size_inches((15, 15))
            plt.tight_layout()
            root = 'retraining_after_alpha_thresholding/cifar100/original/resnet18_in/visualizations'
            plt.savefig(os.path.join(root, f'cosine_similarity_between_drelus_{layer_name}[{block}].alpha{relu_idx}.png'))
            plt.savefig(os.path.join(root, f'cosine_similarity_between_drelus_{layer_name}[{block}].alpha{relu_idx}.pdf'))
            hook_in.remove()
            hook_out.remove()


