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
import matplotlib.pyplot as plt
from collections import OrderedDict
CORRELATION_THRESHOLD = 0.9
CREATE_PLOT_ARTIFACTS = True

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("teacher_model_checkpoint_path", type=str, help='path to the checkpoint of the teacher model')
    args.add_argument("")

    return args.parse_args()


def mul_iterable(iterable):
    result = 1
    if len(iterable) == 0:
        return result
    return iterable[0] * mul(iterable[1:])


class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
        return induced_relu * self.alphas.expand_as(x) + (1 - self.alphas.expand_as(x)) * x


device = torch.device("cuda")
"""
IMPORTANT: EDIT ALL SCRIPT PARAMETERS HERE, CURRENTLY THE SCRIPT DOES NOT ACCEPT COMMAND LINE ARGUMENTS.
"""
my_args = MyNamespace(alpha=1e-05, arch='resnet18_in', batch=128, block_type='LearnableAlpha',
                      budegt_type='absolute', dataset='cifar100-new-split', epochs=2000, finetune_epochs=100,
                      gamma=0.1, gpu=0, logname='resnet18_in_unstructured_.txt', lr=0.001, lr_step_size=30,
                      momentum=0.9, num_of_neighbors=4,
                      outdir='./retraining_after_alpha_thresholding/cifar100/original/resnet18_in/',
                      print_freq=100, relu_budget=15000,
                      savedir='./checkpoints/resnet18_cifar100.pth', stride=1, threshold=0.01,
                      weight_decay=0.0005, workers=4,
                      teacher_model_checkpoint_path="./checkpoints/resnet18_cifar100_new_split.pth",
                      snl_model_checkpoint_path="./checkpoints/resnet18_cifar100_snl_finetune_final.pth",
                      experiment_root="./associate_prototypes/cifar100/resnet18_in",
                      visualizations_directory="fsp_like_alg/visualizations/",
                      checkpoints_directory="checkpoints/")


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


def get_input_activations_for_layer(model, data_loader, activation_in, layer_name, block, relu_idx):
    layername = f'{layer_name}_block{block}_relu{relu_idx}'
    hook_in = model.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        getActivationInput(layername))

    # The return value here does not matter, this inference run is just for caching purposes.
    model_inference(model, data_loader,
                                   device, display=True)
    curr_block_in = torch.cat(activation_in[layername], 0)
    activation_in = {}
    hook_in.remove()
    return curr_block_in


def associate_inducers_with_inducees(cosine_similarity, layername, snl_prototypes_indices):
	layer_directory_path = os.path.join(my_args.experiment_root, my_args.visualizations_directory, layername)

    inducer_to_inducee = OrderedDict({})
    # handle the edge case of 'lone' inducers. That is, prototypes that are low correlated to all other prototypes
    for potential_inducer, current_similarity in enumerate(cosine_similarity):
        current_similarity_list = current_similarity.tolist()
        current_similarity_list[potential_inducer] = 0
        if max(current_similarity_list) < CORRELATION_THRESHOLD:
            inducer_to_inducee[potential_inducer] = [potential_inducer]
            cosine_similarity[potential_inducer, ...] = np.nan
            cosine_similarity[..., potential_inducer] = np.nan
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

        if iternum < 60:
            plt.close('all')
            plt.title(f'correlation sum vs prototype location\n{layername}')
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
            os.makedirs(os.path.join(layer_directory_path, layername, 'correlation_sum', ), exist_ok=True)
            plt.savefig(os.path.join(layer_directory_path, layername, 'correlation_sum', f'iteration#{iternum:03d}.png'))

            plt.close('all')
            plt.title(f'correlation matrix\n{layername}')
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
            # plt.grid(True)
            plt.colorbar()
            fig = plt.gcf()
            fig.set_size_inches((15, 15))
            plt.tight_layout()
            plt.ylabel('correlation matrix')
            os.makedirs(os.path.join(layer_directory_path, layername, 'correlation_mat', ), exist_ok=True)
            plt.savefig(os.path.join(layer_directory_path, layername, 'correlation_mat', f'iteration#{iternum:03d}.png'))
        iternum += 1
    return inducer_to_inducee

def get_teacher_model():
    base_classifier = get_architecture(my_args.arch, my_args.dataset, device, my_args)
    checkpoint = torch.load(my_args.savedir, map_location=device)
    checkpoint = torch.load(
        my_args.teacher_model_checkpoint_path,
        map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    return base_classifier

"""
This function performs a finetune KD loop for the student model. It does not return the student model itself because its weights are updated in-place.
"""
def finetune_student_model(student_model, teacher_model, train_loader, test_loader, outdir, epochs=20, momentum=0.9, weight_decay=0.0005, arch='resnet18_in',
                           dataset='cifar100-new-split', relu_budget=15000):
    finetune_epoch = epochs
    # finetune_epoch = 2

    optimizer = SGD(student_model.parameters(), lr=1e-3, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

    print("Finetuning the model")

    best_top1 = 0
    for epoch in range(finetune_epoch):
        train_loss, train_top1, train_top5 = train_kd(train_loader, student_model, teacher_model, optimizer, criterion, epoch,
                                                      device)
        test_loss, test_top1, test_top5 = test(test_loader, student_model, criterion, device, 100, display=True)
        scheduler.step()

        if best_top1 < test_top1:
            best_top1 = test_top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                'arch': arch,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(outdir,
                            f'snl_best_checkpoint_{arch}_{dataset}_{relu_budget}.pth.tar'))

    print("Final best Prec@1 = {}%".format(best_top1))

def create_experiment_directories():
	os.makedirs(os.path.join(my_args.experiment_root, my_args.visualizations_directory), exist_ok=True)
	os.makedirs(os.path.join(my_args.experiment_root, my_args.checkpoints_directory), exist_ok=True)

def get_data_loaders(dataset=my_args.dataset, batch_size=my_args.batch, workers=my_args.workers):
    train_dataset = get_dataset(dataset, 'train')
    test_dataset = get_dataset(dataset, 'test')
    val_dataset = get_dataset(dataset, 'val')

    pin_memory = (dataset == "imagenet")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

def load_snl_model(dataset=my_args.dataset):
    snl_model = get_architecture(my_args.arch, dataset, device, my_args)
    checkpoint = torch.load(my_args.savedir, map_location=device)
    checkpoint = torch.load(my_args.snl_model_checkpoint_path, map_location=device)
    snl_model.load_state_dict(checkpoint['state_dict'])

    return snl_model

def create_sparse_matrix(snl_prototypes_indices, inducer_to_inducee, matrix_size):
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

    return sparse_matrix

def create_plot_artifacts(cosine_similarity, layer_name, channels, rows, cols):

    layer_directory_path = os.path.join(my_args.experiment_root, my_args.visualizations_directory, layer_name)
    os.makedirs(layer_directory_path, exist_ok=True)

    plt.close('all')
    plt.title(f'correlation sum vs prototype location\n{layer_name}')
    plt.stem(cosine_similarity.sum(axis=1))
    plt.xticks(range(len(channels)),
               [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                   channels,
                   rows,
                   cols)], rotation='vertical', fontsize='small')
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches((12, 5))
    plt.tight_layout()
    plt.ylabel('correlation sum')
    plt.savefig(os.path.join(layer_directory_path, 'correlation_sum_stem_plot_iter0.png'))
    plt.close('all')
    plt.title(f'correlation matrix\n{layer_name}')
    plt.imshow(cosine_similarity)
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches((15, 15))
    plt.tight_layout()
    plt.ylabel('correlation matrix')
    plt.savefig(os.path.join(layer_directory_path, 'correlation_mat.png'))

def process_layer(layer, block, relu_idx, snl_model, train_loader, activations_cache_loader, test_loader):
	checkpoints_directory = os.path.join(my_args.experiment_root, my_args.checkpoints_directory)

    layer_name = f'{layer}_block{block}_relu{relu_idx}'
                
    # cache activations
    curr_block_in = get_input_activations_for_layer(snl_model, activations_cache_loader, activation_in, layer, block, relu_idx)

    # print stats
    nof_activated_channels = np.unique(
            np.where((snl_model.get_submodule(layer_name)[block].get_submodule(
            f'alpha{relu_idx}').alphas > my_args.threshold).cpu())[1])
    print(f"activated channels: {nof_activated_channels}")

    """
    As np.where returns a tuple of arrays for the given condition,
    each array contains the index of the specific dimension, we need to skip the batch dimension and start taking all indices starting
    from channel down to the end.
    """
    snl_prototypes_indices = np.where((snl_model.get_submodule(layer_name)[block].get_submodule(
            f'alpha{relu_idx}').alphas > my_args.threshold).cpu())[1:]

    # Access the responses based on the indices and map them from {0, 1} to {-1, 1}
    prototypes_drelu = curr_block_in[..., snl_prototypes_indices[-3], snl_prototypes_indices[-2], snl_prototypes_indices[-1]]
    prototypes_drelu = 2 * prototypes_drelu - 1

    # we measure distance as the cosime similarity between two epochs.
    # Two DReLUs are considered similar if their dot product is high:
    cosine_similarity = normalize(prototypes_drelu.T) @ normalize(prototypes_drelu.T).T
    if CREATE_PLOT_ARTIFACTS:
        create_plot_artifacts(cosine_similarity, layer_name, snl_prototypes_indices[0], snl_prototypes_indices[1], snl_prototypes_indices[2])

    print(f"layer {layer_name} has {cosine_similarity.shape[0]} SNL prototypes")

    inducer_to_inducee = associate_inducers_with_inducees(cosine_similarity, layer_name, snl_prototypes_indices)

    print(f"Reduced the number of prototypes from {len(snl_prototypes_indices[0])} to {len(inducer_to_inducee)}")

    original_feature_size = curr_block_in.shape[-3], curr_block_in.shape[-2], curr_block_in.shape[-1] # C x W x H
    sparse_matrix_size = (mul_iterable(original_feature_size), mul_iterable(original_feature_size)) # (C x W x H) x (C x W x H)

    sparse_matrix = create_sparse_matrix(snl_prototypes_indices, inducer_to_inducee, sparse_matrix_size)

    inducers_mask = torch.zeros((original_feature_size))
    for inducer in inducer_to_inducee:
        inducers_mask[snl_prototypes_indices[0][inducer], snl_prototypes_indices[1][inducer], snl_prototypes_indices[2][inducer]] = 1.0

    # Replace the alphas component with the InducedReLU block with the choosing matrix embedded within it.
    alphas = snl_model.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}'
                                                                      ).alphas.cpu().detach()
    setattr(snl_model.get_submodule(layer_name)[block], f'alpha{relu_idx}',
            InducedLearnableAlpha(alphas.cuda(), inducers_mask.cuda(), sparse_matrix.cuda()))

    original_acc = model_inference(snl_model, test_loader,
                                   device, display=True)
    print("Pre-finetune model test accuracy after replacing layer {}: {:.5}".format(layer_name, original_acc))

    current_layer_checkpoints_dir = os.path.join(checkpoints_directory, layer_name)
    os.makedirs(current_layer_checkpoints_dir, exist_ok=True)

    finetune_student_model(snl_model, get_teacher_model(), train_loader,
                           test_loader, current_layer_checkpoints_dir, epochs=20, momentum=0.9,
                           weight_decay=0.0005, arch='resnet18_in')


def main():

	create_experiment_directories()

    train_loader, val_loader, test_loader = get_data_loaders()

    activations_cache_loader = val_loader

    if val_loader is None:
        activations_cache_loader = test_loader

    snl_model = load_snl_model()
    snl_model.eval()

    original_acc = model_inference(snl_model, test_loader,
                                   device, display=True)
    print("SNL Model original test accuracy: {:.5}".format(original_acc))

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                process_layer(layer, block, relu_idx, snl_model, train_loader, activations_cache_loader, test_loader)


if __name__ == '__main__':
    main(args)
