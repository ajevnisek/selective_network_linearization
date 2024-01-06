import os
import argparse
from architectures_unstructured import get_architecture
import torch
from torch.utils.data import DataLoader
from utils import get_dataset, model_inference
import matplotlib.pyplot as plt

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

    layername = f'{layer_name}_block{block}_relu{relu_idx}'
    hook_in = model.get_submodule(layer_name)[block].get_submodule(f'alpha{relu_idx}').register_forward_hook(
        get_activation_drelu(layername, cache_dict))

    # The return value here does not matter, this inference run is just for caching purposes.
    model_inference(model, data_loader,
                                   device, display=True)
    drelus = torch.cat(cache_dict[layername], 0)
    hook_in.remove()
    return drelus

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


def show_layer_relu_statistics(layer, block, relu_idx, snl_model, activations_cache_loader, test_loader, out_dir, hammings):

    layer_name = f'{layer}_block{block}_relu{relu_idx}'

    print(f"{layer_name}")

    drelus = get_drelus_for_layer(snl_model, activations_cache_loader, layer, block, relu_idx)
    data_shape = drelus.shape

    B, C, W, H = data_shape[0], data_shape[1], data_shape[2], data_shape[3]

    drelus = drelus.flatten(start_dim=1).T.reshape(C, H, W, -1)

    # curr_block_in is of shape |LayerReLUs| x |ValSamples|
    alphas = snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.squeeze(dim=0)

    relu_applied_indices = torch.where(alphas == 1)

    relu_inputs = drelus[relu_applied_indices[0], relu_applied_indices[1], relu_applied_indices[2], :]

    normalized_hamming_list = []

    for curr_relu in range(relu_inputs.shape[0]):
        curr_item = torch.sum(relu_inputs[curr_relu]).item() / relu_inputs.shape[1]
        normalized_hamming_list.append(curr_item)
        hammings.append(curr_item)

    plt.hist(normalized_hamming_list, 100)
    plt.title(f"Histogram of normalized Hamming weight for layer {layer_name}")
    plt.xlabel("Normalized Hamming weight")
    plt.ylabel("Count")
    plt.savefig(os.path.join(out_dir, f"{layer_name}_hamming.png"))
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

    hammings = []

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                show_layer_relu_statistics(layer, block, relu_idx, snl_model, activations_cache_loader, test_loader, args.output_dir, hammings)

    plt.hist(hammings, 100)
    plt.title(f"Histogram of normalized Hamming weight for 30k SNL, ResNet18, CIFAR100")
    plt.xlabel("Normalized Hamming weight")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, f"overall_hamming.png"))


if __name__ == '__main__':
    args = parse_args()
    main(args)
