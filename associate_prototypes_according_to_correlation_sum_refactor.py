import os
import argparse
import pickle

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
CREATE_PLOT_ARTIFACTS = True



def mul(iterable):
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


def get_input_activations_for_layer(base_classifier, test_loader, activation_in, layer_name, block, relu_idx):
    layername = f'{layer_name}_block{block}_relu{relu_idx}'
    hook_in = base_classifier.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        getActivationInput(layername))
    original_acc = model_inference(base_classifier, test_loader,
                                   device, display=True)
    print("Original Model Test Accuracy: {:.5}".format(original_acc))
    curr_block_in = torch.cat(activation_in[layername], 0)
    activation_in = {}
    hook_in.remove()
    return curr_block_in


def associate_inducers_with_inducees(cosine_similarity, correlation_threhold, root, layername, snl_prototypes_indices):

    inducer_to_inducee = OrderedDict({})
    # handle the edge case of 'lone' inducers. That is, prototypes that are low correlated to all other prototypes
    for potential_inducer, current_similarity in enumerate(cosine_similarity):
        current_similarity_list = current_similarity.tolist()
        current_similarity_list[potential_inducer] = 0
        if max(current_similarity_list) < correlation_threhold:
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
        inducees = np.where(cosine_similarity[curr_max] > correlation_threhold)[0].tolist()
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
            os.makedirs(os.path.join(root, layername, 'correlation_sum', ), exist_ok=True)
            plt.savefig(os.path.join(root, layername, 'correlation_sum', f'iteration#{iternum:03d}.png'))

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
            os.makedirs(os.path.join(root, layername, 'correlation_mat', ), exist_ok=True)
            plt.savefig(os.path.join(root, layername, 'correlation_mat', f'iteration#{iternum:03d}.png'))
        iternum += 1
    return inducer_to_inducee

def get_teacher_model(args):
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    root = 'snl_output/cifar100/15000/resnet18_in'
    checkpoint = torch.load(
        'snl_output/cifar100/15000/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar',
        map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    return base_classifier

def finetune_student_model(student_model, teacher_model, train_loader, test_loader, outdir, epochs=20, momentum=0.9, weight_decay=0.0005, arch='resnet18_in',
                           dataset='cifar100', relu_budget=15000):
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
    return student_model


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('arch', type=str, choices=ARCHITECTURES)
    parser.add_argument('outdir', type=str, help='folder to save model and training log)')
    parser.add_argument('savedir', type=str, help='folder to load model')
    parser.add_argument('--correlation_threshold', default=0.9, type=float, help='correlation threshold')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=160, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--finetune_epochs', default=20, type=int,
                        help='number of epochs for the finetuning')
    parser.add_argument('--batch', default=256, type=int, metavar='N',
                        help='batchsize (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--lr_milestones', default=[80, 120])
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--noise_sd', default=0.0, type=float,
                        help="standard deviation of Gaussian noise for weight augmentation")
    parser.add_argument('--gpu', default=0, type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--block_type', type=str, default='LearnableAlpha')
    parser.add_argument('--num_of_neighbors', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    logfilename = os.path.join(args.outdir, 'log.txt')
    os.makedirs(args.outdir, exist_ok=True)
    log(logfilename, "Dataset: {:}".format(args.dataset))
    log(logfilename, "Arch: {:}".format(args.arch))
    log(logfilename, "Correlation Threshold: {:}".format(args.correlation_threshold))
    # Loading the base_classifier
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    checkpoint = torch.load(
        './retraining_after_alpha_thresholding/cifar100/original/resnet18_in/snl_best_checkpoint_resnet18_in_cifar100_15000.pth.tar',
        map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()

    original_acc = model_inference(base_classifier, test_loader,
                                   device, display=True)
    print("Original Model Test Accuracy: {:.5}".format(original_acc))

    root = args.outdir
    visualizations_root = os.path.join(root, 'fsp_like_alg/visualizations/')
    os.makedirs(visualizations_root, exist_ok=True)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                layername = f'{layer_name}_block{block}_relu{relu_idx}'
                # cache activations
                log(logfilename, f"Starting process on layer: {layername}")
                log(logfilename, f"caching activations...")
                curr_block_in = get_input_activations_for_layer(base_classifier, test_loader, activation_in, layer_name, block, relu_idx)
                # print stats
                unique_activated_channels = np.unique(
                    np.where((base_classifier.get_submodule(layer_name)[block].get_submodule(
                        f'alpha{relu_idx}').alphas > args.threshold).cpu())[1])
                print(f"activated channels: {unique_activated_channels}")
                log(logfilename, f"SNL yielded {len(unique_activated_channels)} activated channels in layer {layername}...")
                log(logfilename, f"The activated channels in layer {layername} are: {unique_activated_channels}")

                # sml lets alpha= '1' for DReLU it computes.
                snl_prototypes_indices = np.where((base_classifier.get_submodule(layer_name)[block].get_submodule(
                        f'alpha{relu_idx}').alphas > args.threshold).cpu())[1:]
                # now, we focus on snl's prototypes DReLUs:
                prototypes_drelu = curr_block_in[..., snl_prototypes_indices[-3], snl_prototypes_indices[-2], snl_prototypes_indices[-1]]
                prototypes_drelu = 2 * prototypes_drelu - 1
                # we measure distance as the cosime similarity between two epochs.
                # Two DReLUs are considered similar if their dot product is high:
                cosine_similarity = normalize(prototypes_drelu.T) @ normalize(prototypes_drelu.T).T
                if CREATE_PLOT_ARTIFACTS:
                    plt.close('all')
                    plt.title(f'correlation sum vs prototype location\n{layername}')
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
                    os.makedirs(os.path.join(visualizations_root, layername, ), exist_ok=True)
                    plt.savefig(os.path.join(visualizations_root, layername, 'correlation_sum_stem_plot_iter0.png'))
                    plt.close('all')
                    plt.title(f'correlation matrix\n{layername}')
                    plt.imshow(cosine_similarity)
                    # plt.xticks(range(len(snl_prototypes_indices[0])),
                    #            [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                    #                snl_prototypes_indices[0],
                    #                snl_prototypes_indices[1],
                    #                snl_prototypes_indices[2])], rotation='vertical', fontsize='small')
                    # plt.yticks(range(len(snl_prototypes_indices[0])),
                    #            [f"({ch:03d}, {row:03d}, {col:03d})" for (ch, row, col) in zip(
                    #                snl_prototypes_indices[0],
                    #                snl_prototypes_indices[1],
                    #                snl_prototypes_indices[2])], fontsize='small')
                    # plt.grid(True)
                    plt.colorbar()
                    fig = plt.gcf()
                    fig.set_size_inches((15, 15))
                    plt.tight_layout()
                    plt.ylabel('correlation matrix')
                    os.makedirs(os.path.join(visualizations_root, layername, ), exist_ok=True)
                    plt.savefig(os.path.join(visualizations_root, layername, 'correlation_mat.png'))

                print(f"layer {layername} has {cosine_similarity.shape[0]} SNL prototypes")
                inducer_to_inducee = associate_inducers_with_inducees(cosine_similarity,
                                                                      args.correlation_threshold,
                                                                      visualizations_root, layername,
                                                                      snl_prototypes_indices)

                print(f"Reduced the number of prototypes from {len(snl_prototypes_indices[0])} to {len(inducer_to_inducee)}")
                log(logfilename, f"Our FSP-like algorithm reduced the number of prototypes from "
                                 f"{len(snl_prototypes_indices[0])} to {len(inducer_to_inducee)}")

                inducer_to_inducee_in_channel_row_and_col_terminology = {
                    (snl_prototypes_indices[0][inducer], snl_prototypes_indices[1][inducer], snl_prototypes_indices[2][inducer]): [
                        (snl_prototypes_indices[0][inducee], snl_prototypes_indices[1][inducee], snl_prototypes_indices[2][inducee])
                        for inducee in inducer_to_inducee[inducer]

                ]
                    for inducer in inducer_to_inducee
                }
                os.makedirs(os.path.join(root, 'associations', f'{layername}'), exist_ok=True)
                with open(os.path.join(root, 'associations', f'{layername}', 'inducer_to_inducee.pkl'), 'wb') as f:
                    pickle.dump(inducer_to_inducee, f)
                with open(os.path.join(root, 'associations', f'{layername}',
                                       'inducer_to_inducee_in_channel_row_and_col_terminology.pkl'), 'wb') as f:
                    pickle.dump(inducer_to_inducee_in_channel_row_and_col_terminology, f)

                # channels, rows, columns
                original_feature_size = curr_block_in.shape[-3], curr_block_in.shape[-2], curr_block_in.shape[-1]
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


                alphas = base_classifier.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}'
                                                                                        ).alphas.cpu().detach()
                setattr(base_classifier.get_submodule(layer_name)[block], f'alpha{relu_idx}',
                        InducedLearnableAlpha(alphas.cuda(), inducers_mask.cuda(), sparse_matrix.cuda()))
                original_acc = model_inference(base_classifier, test_loader,
                                               device, display=True)
                log(logfilename, f"After replacing {layername} in InducedReLU, accuracy is:"
                                 f" {original_acc:.3f} [%]")
                print("Original Model Test Accuracy: {:.5}".format(original_acc))
                outdir = os.path.join(root, 'checkpoints',  layername)
                os.makedirs(outdir, exist_ok=True)
                base_classifier = finetune_student_model(base_classifier, get_teacher_model(args), train_loader,
                                                         test_loader, outdir, epochs=args.finetune_epochs, momentum=0.9,
                                                         weight_decay=0.0005, arch=args.arch)
                original_acc = model_inference(base_classifier, test_loader,
                                               device, display=True)
                log(logfilename, f"After replacing {layername} in InducedReLU and finetuning, accuracy is:"
                                 f" {original_acc:.3f} [%]")


if __name__ == '__main__':
    main()
