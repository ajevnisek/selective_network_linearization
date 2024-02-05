import os
import argparse
from architectures_unstructured import get_architecture
from datasets import get_num_classes
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import get_dataset, model_inference
import matplotlib.pyplot as plt
import sklearn.metrics

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("snl_model_checkpoint_path", type=str, help='path to the checkpoint of the SNL model')
    args.add_argument("output_dir", type=str, help='the directory to which the plots shall be saved')
    args.add_argument("-a", "--arch", type=str, default="resnet18_in", help='the SNL-based architecture to use')
    args.add_argument("-d", "--dataset", type=str, default="cifar100", help='the dataset to use')
    args.add_argument("-b", "--batch_size", type=int, default=128, help='batch size for data loaders')
    args.add_argument("-w", "--workers", type=int, default=4, help='number of workers to use for data loaders')
    args.add_argument("-s", "--stride", type=int, default=1, help='the relevant stride for the chosen architecture')
    args.add_argument("-t", "--threshold", type=float, default=0.01, help='the threshold for alphas')
    args.add_argument("-k", "--block_type", type=str, default="LearnableAlpha", help='the block type to use in the SNL routine')
    args.add_argument("-n", "--num_of_neighbors", type=int, default=4, help='where applicable, the number of neighbors for a ReLU pruning routine')

    return args.parse_args()

device = torch.device("cuda")

def get_activation_drelu(name, cache_dict):
    def hook(model, input, output):
        if name not in cache_dict:
            cache_dict[name] = [(input[0] > 0).detach().cpu().byte()]
        else:
            cache_dict[name].append((input[0] > 0).detach().cpu().byte())

    return hook

def get_drelus_for_layer(model, data_loader, layer_name, block, relu_idx):
    cache_dict = {}
    outputs_list = []

    layername = f'{layer_name}_block{block}_relu{relu_idx}'
    hook_in = model.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        get_activation_drelu(layername, cache_dict))

    # The return value here does not matter, this inference run is just for caching purposes.
    model_inference(model, data_loader,
                                   device, display=True, outputs_list=outputs_list)
    drelus = torch.cat(cache_dict[layername], 0)
    outputs = torch.cat(outputs_list, 0)

    hook_in.remove()
    return drelus, outputs

def get_flattened_alphas(snl_model):

    alpha_list = []

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                alpha_list.append(snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.flatten())

    return alpha_list

def get_data_loaders(dataset, batch_size, workers):
    train_dataset = get_dataset(dataset, 'train')
    test_dataset = get_dataset(dataset, 'test')
    val_dataset = get_dataset(dataset, 'val')

    pin_memory = (dataset == "imagenet")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory) if val_dataset is not None else None

    return train_loader, val_loader, test_loader

def load_snl_model(arch, dataset, snl_model_checkpoint_path, args):
    snl_model = get_architecture(arch, dataset, device, args)
    checkpoint = torch.load(snl_model_checkpoint_path, map_location=device)
    snl_model.load_state_dict(checkpoint['state_dict'])

    return snl_model

def separate_features_by_class(features, labels, num_classes):

    features_by_class = []

    for label in range(num_classes):
        label_indices = torch.where(labels == label)[0]
        features_by_class.append(features[ : , label_indices])

    return features_by_class

def get_max_class_distance_matrix(layer, block, relu_idx, snl_model, activations_cache_loader, num_classes):
    layer_name = f'{layer}_block{block}_relu{relu_idx}'

    print(f"{layer_name}")

    drelus, outputs = get_drelus_for_layer(snl_model, activations_cache_loader, layer, block, relu_idx)
    data_shape = drelus.shape

    drelus = drelus.flatten(start_dim=1).T
    alphas = snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.flatten().cpu()

    relu_applied_indices = torch.where(alphas == 1)[0]
    prototypes_indices = np.where((snl_model.get_submodule(layer)[block].get_submodule(
                        f'alpha{relu_idx}').alphas == 1).cpu())[1:]

    snl_prototypes_drelus = torch.index_select(drelus, 0, relu_applied_indices)

    features_by_class = separate_features_by_class(snl_prototypes_drelus, outputs, num_classes)

    max_distance_matrix = torch.zeros(snl_prototypes_drelus.shape[0], snl_prototypes_drelus.shape[0])
    for feature in features_by_class:
        """
        When dealing with binary tensors, Hamming distance is equivalent to calculating the Eucledian distance and
        then applying a power of two. Calculating the Eucledian distance is much faster than specifying 'hamming'
        to sklearn's implemenetation, so that's what we're doing here.
        """
        current_class_distance_matrix = sklearn.metrics.pairwise_distances(feature) ** 2
        current_class_distance_matrix /= feature.shape[1] # Normalize to [0, 1]
        """
        As we have a list of Hamming distances per class, we simply take the element-wise maximum value of each
        tensor, which means that at the end, we arrive a tensor for which each element is the maximum Hamming distance
        among all classes.
        """
        max_distance_matrix = torch.max(max_distance_matrix, torch.from_numpy(current_class_distance_matrix))

    return max_distance_matrix, prototypes_indices

def show_layer_class_based_distance(layer, block, relu_idx, snl_model, activations_cache_loader, num_classes, out_dir):

    max_distance_matrix, _ = get_max_class_distance_matrix(layer, block, relu_idx, snl_model, activations_cache_loader, num_classes)

    plt.imshow(max_distance_matrix)
    plt.title(f'Normalized class-based Hamming distance for {layer_name}')
    plt.xlabel('Prototypes')
    plt.ylabel('Prototypes')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f'class_hamming_map_{layer_name}.png'))
    plt.clf()

    # Fill diagnoal with 1's so that when we calculated the minimum, we don't take the diagonal values into account.
    max_distance_matrix.fill_diagonal_(1)
    min_normalized_distance = torch.min(max_distance_matrix, dim=0)[0]

    plt.hist(min_normalized_distance)
    plt.title(f"Histogram of class-based minimal Hamming distances for {layer_name}")
    plt.xlabel("Min normalized Hamming distance")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, f'min_normalized_distance_{layer_name}.png'))
    plt.clf()

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.workers)

    activations_cache_loader = val_loader

    # Not all datasets have a validation dataset in their split, so we use the train set as the fallback option.
    if val_loader is None:
        activations_cache_loader = train_loader

    snl_model = load_snl_model(args.arch, args.dataset, args.snl_model_checkpoint_path, args)
    snl_model.eval()

    num_classes = get_num_classes(args.dataset)

    for layer in ['layer1','layer2', 'layer3','layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                show_layer_class_based_distance(layer, block, relu_idx, snl_model, activations_cache_loader, num_classes, args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
