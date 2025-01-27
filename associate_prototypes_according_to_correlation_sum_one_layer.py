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
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def mul(iterable):
    result = 1
    if len(iterable) == 0:
        return result
    return iterable[0] * mul(iterable[1:])


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
snl_prototypes_indices = np.where((base_classifier.layer1[0].alpha1.alphas > my_args.threshold).cpu())[1:]
prototypes_drelu = layer1_block0_relu1_in[..., snl_prototypes_indices[-3], snl_prototypes_indices[-2], snl_prototypes_indices[-1]]
prototypes_drelu = 2 * prototypes_drelu - 1

root = 'induce_prototypes/cifar100/resnet18_in/fsp_like_alg/visualizations/'
layername = 'layer1_block0_relu1'

cosine_similarity = normalize(prototypes_drelu.T) @ normalize(prototypes_drelu.T).T
import matplotlib.pyplot as plt
plt.close('all')
plt.title('correlation sum vs prototype location')
plt.stem(cosine_similarity.sum(axis=1))
plt.xticks(range(len(snl_prototypes_indices[0])),
           [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
               snl_prototypes_indices[0],
               snl_prototypes_indices[1],
               snl_prototypes_indices[2])], rotation='vertical', fontsize='small')
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches((12, 5))
plt.tight_layout()
plt.ylabel('correlation sum')
os.makedirs(os.path.join(root, layername,), exist_ok=True)
plt.savefig(os.path.join(root, layername, 'correlation_sum_stem_plot_iter0.png'))

original_cosine_similarity = cosine_similarity.copy()
from collections import OrderedDict
CORRELATION_THRESHOLD = 0.9
inducer_to_inducee = OrderedDict({})

iternum = 0
while True:
    correlation_sum = np.nansum(cosine_similarity, axis=1)
    # avoid choosing inducers and inducess:
    for inducer in inducer_to_inducee:
        correlation_sum[inducer] = -np.inf
        for inducee in inducer_to_inducee[inducer]:
            correlation_sum[inducee] = -np.inf
    curr_max = correlation_sum.argmax()
    if correlation_sum[curr_max] == -np.inf:
        break
    inducees = np.where(cosine_similarity[curr_max] > CORRELATION_THRESHOLD)[0].tolist()
    inducer_to_inducee[curr_max] = inducees
    for item in [curr_max] + inducees:
        cosine_similarity[item, ...] = np.nan
        cosine_similarity[..., item] = np.nan


    plt.close('all')
    plt.title('correlation sum vs prototype location')
    plt.stem(correlation_sum)
    plt.xticks(range(len(snl_prototypes_indices[0])),
               [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                   snl_prototypes_indices[0],
                   snl_prototypes_indices[1],
                   snl_prototypes_indices[2])], rotation='vertical', fontsize='small')
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches((12, 5))
    plt.tight_layout()
    plt.ylabel('correlation sum')
    os.makedirs(os.path.join(root, layername, 'correlation_sum', ), exist_ok=True)
    plt.savefig(os.path.join(root, layername,'correlation_sum', f'iteration#{iternum:03d}.png'))

    plt.close('all')
    plt.title('correlation matrix')
    plt.imshow(cosine_similarity)
    plt.xticks(range(len(snl_prototypes_indices[0])),
               [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                   snl_prototypes_indices[0],
                   snl_prototypes_indices[1],
                   snl_prototypes_indices[2])], rotation='vertical', fontsize='small')
    plt.yticks(range(len(snl_prototypes_indices[0])),
               [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                   snl_prototypes_indices[0],
                   snl_prototypes_indices[1],
                   snl_prototypes_indices[2])], fontsize='small')
    plt.grid(True)
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches((15, 15))
    plt.tight_layout()
    plt.ylabel('correlation matrix')
    os.makedirs(os.path.join(root, layername, 'correlation_mat',), exist_ok=True)
    plt.savefig(os.path.join(root, layername, 'correlation_mat', f'iteration#{iternum:03d}.png'))
    iternum += 1


print(f"Reduced the number of prototypes from {len(snl_prototypes_indices[0])} to {len(inducer_to_inducee)}")

inducer_to_inducee_in_channel_row_and_col_terminology = {
    (snl_prototypes_indices[0][inducer], snl_prototypes_indices[1][inducer], snl_prototypes_indices[2][inducer]): [
        (snl_prototypes_indices[0][inducee], snl_prototypes_indices[1][inducee], snl_prototypes_indices[2][inducee])
        for inducee in inducer_to_inducee[inducer]

]
    for inducer in inducer_to_inducee
}

# channels, rows, columns
original_feature_size = layer1_block0_relu1_in.shape[-3], layer1_block0_relu1_in.shape[-2], layer1_block0_relu1_in.shape[-1]
sparse_matrix_size = (mul(original_feature_size), mul(original_feature_size))
indices_for_sparse_matrix = [[], []]
for inducer in inducer_to_inducee:
    inducer_location_in_feature_map = np.ravel_multi_index(
        (snl_prototypes_indices[0][inducer], snl_prototypes_indices[1][inducer], snl_prototypes_indices[2][inducer]),
        original_feature_size)
    inducees_location_in_feature_map = [np.ravel_multi_index((snl_prototypes_indices[0][inducee], snl_prototypes_indices[1][inducee], snl_prototypes_indices[2][inducee]), original_feature_size) for inducee in inducer_to_inducee[inducer]]
    for column in inducees_location_in_feature_map:
        indices_for_sparse_matrix[0].append(inducer_location_in_feature_map)
        indices_for_sparse_matrix[1].append(column)

sparse_matrix = torch.sparse_coo_tensor(indices=indices_for_sparse_matrix,
                                        values=[1.0] * len(indices_for_sparse_matrix[0]),
                                        size=sparse_matrix_size)

inducers_mask = torch.zeros((original_feature_size))
for inducer in inducer_to_inducee:
    inducers_mask[snl_prototypes_indices[0][inducer], snl_prototypes_indices[1][inducer], snl_prototypes_indices[2][inducer]] = 1.0


class InducedLearnableAlpha(torch.nn.Module):
    def __init__(self, alphas, inducers_mask, choosing_matrix):
        super(InducedLearnableAlpha, self).__init__()
        self.alphas = alphas
        self.inducers_mask = inducers_mask.unsqueeze(0)
        self.choosing_matrix = choosing_matrix

    def forward(self, x):
        drelu_x = (x > 0).float()
        inducers_drelu = self.inducers_mask.expand_as(x) * drelu_x
        induced_drelu = (self.choosing_matrix @ inducers_drelu.flatten(1).T).T
        induced_relu = induced_drelu.reshape(x.shape) * x
        return induced_relu * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x


alphas = base_classifier.layer1[0].alpha1.alphas.cpu().detach()
base_classifier.layer1[0].alpha1 = InducedLearnableAlpha(alphas.cuda(), inducers_mask.cuda(), sparse_matrix.cuda())
original_acc = model_inference(base_classifier, test_loader,
                               device, display=True)
print("Original Model Test Accuracy: {:.5}".format(original_acc))
