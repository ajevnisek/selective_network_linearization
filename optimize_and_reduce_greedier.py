import os
import argparse
from architectures_unstructured import get_architecture
import torch
from torch.utils.data import DataLoader
from collections import Counter
from utils import get_dataset, model_inference
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *

LOG_FILE_NAME = "log.txt"

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("snl_model_checkpoint_path", type=str, help='path to the checkpoint of the SNL model')
    args.add_argument("teacher_model_checkpoint_path", type=str, help='path to the checkpoint of the teacher model')
    args.add_argument("output_dir", type=str, help='the directory to which the final checkpoints will be saved')
    args.add_argument("-r", "--reduce_threshold", type=float, default=0.05, help='the train set percentage drop threshold for reducing alphas')
    args.add_argument("-l", "--step_reduce", type=int, default=100, help='the number of alphas to prune at a time')
    args.add_argument("-e", "--random_tries", type=int, default=100, help='the maximum number of random prune attempts per cycle')
    args.add_argument("-p", "--relu_target", type=int, default=15000, help='the target ReLU budget to arrive at')
    args.add_argument("-a", "--arch", type=str, default="resnet18_in", help='the SNL-based architecture to use')
    args.add_argument("-d", "--dataset", type=str, default="cifar100", help='the dataset to use')
    args.add_argument("-b", "--batch_size", type=int, default=128, help='batch size for data loaders')
    args.add_argument("-w", "--workers", type=int, default=4, help='number of workers to use for data loaders')
    args.add_argument("-s", "--stride", type=int, default=1, help='the relevant stride for the chosen architecture')
    args.add_argument("-t", "--threshold", type=float, default=0.01, help='the threshold for alphas')
    args.add_argument("-k", "--block_type", type=str, default="LearnableAlpha", help='the block type to use in the SNL routine')
    args.add_argument("-n", "--num_of_neighbors", type=int, default=4, help='where applicable, the number of neighbors for a ReLU pruning routine')
    args.add_argument("-f", "--finetune_epochs", type=int, default=20, help='number of epochs for finetune in case the accuracy dropped by some threshold')

    return args.parse_args()

device = torch.device("cuda")

class Resnet18ReLUInfo:

    def __init__(self, model):
        self.relu_info_dict = {}

        min_index, max_index = 0, 0
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block in [0, 1]:
                for relu_idx in [1, 2]:
                    B, C, H, W = model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.shape
                    max_index = B * C * H * W + min_index
                    self.relu_info_dict[(min_index, max_index)] = layer, block, relu_idx
                    min_index = max_index

    def get_relu_stats(self, flattened_relu_index):
        for k, v in self.relu_info_dict.items():
            min_index, max_index = k
            if min_index < flattened_relu_index < max_index:
                return v 
        return None

"""
This function performs a finetune KD loop for the student model. It does not return the student model itself because its weights are updated in-place.
"""
def finetune_student_model(student_model, teacher_model, train_loader, test_loader, outdir, epochs=20, lr=1e-3,
                           momentum=0.9, weight_decay=0.0005, arch='resnet18_in',
                           dataset='cifar100', relu_budget=15000):
    finetune_epoch = epochs

    optimizer = SGD(student_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

    # don't optimize for the alphas, we assume that their location is fixed.
    for name, param in student_model.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > 1e-2
            param.data = boolean_list.float()
            param.requires_grad = False

    log_file_path = os.path.join(outdir, LOG_FILE_NAME)

    log(log_file_path, "Finetuning the model")

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
                            f'finetune_best_checkpoint_{arch}_{dataset}_{relu_budget}.pth.tar'))

    log(log_file_path, "Final best Prec@1 = {}%".format(best_top1))
    return student_model

def get_training_accuracy(model, data_loader):
    return model_inference(model, data_loader, device, display=True)

def get_relu_count(snl_model):
    relu_count = 0
    for name, param in snl_model.named_parameters():
        if 'alpha' in name:
            relu_count += (param.data == 1).sum()
    if relu_count == 0:
        import ipdb; ipdb.set_trace()
    return relu_count

# WARNING: alpha_list MUST contain ALL alphas in order.
def apply_alphas(snl_model, alpha_list):

    alpha_list_index = 0

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.data = alpha_list[alpha_list_index].clone()
                alpha_list_index += 1

def get_flattened_alphas(snl_model):

    alpha_list = []

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                alpha_list.append(snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.flatten())

    return alpha_list

def apply_alpha_long_tensor(snl_model, alpha_flattened_tensor):

    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                B, C, H, W = snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.shape
                flattened_shape = C * H * W
                snl_model.get_submodule(layer)[block].get_submodule(f'alpha{relu_idx}').alphas.data = alpha_flattened_tensor[ : flattened_shape].reshape(B, C, H, W).clone()
                alpha_flattened_tensor = alpha_flattened_tensor[flattened_shape : ]

def do_prune_by_indices(snl_model, indices):      
    alphas = get_flattened_alphas(snl_model)

    alpha_long_tensor = torch.cat(alphas)

    alpha_long_tensor[indices] = 0

    apply_alpha_long_tensor(snl_model, alpha_long_tensor)

def get_accuracy_for_random_prune(snl_model, train_loader, reduce_step):

    alpha_long_tensor = torch.empty([])

    alphas = get_flattened_alphas(snl_model)

    alpha_long_tensor = torch.cat(alphas)

    indices_of_active_relus = torch.where(alpha_long_tensor == 1)[0]

    indices_to_indices_to_prune = torch.randperm(len(indices_of_active_relus))[ : reduce_step]

    indices_to_prune = indices_of_active_relus[indices_to_indices_to_prune]

    alpha_long_tensor[indices_to_prune] = 0

    apply_alpha_long_tensor(snl_model, alpha_long_tensor)

    accuracy = get_training_accuracy(snl_model, train_loader)

    # Restore alphas to possibly perform another random prune
    apply_alpha_long_tensor(snl_model, torch.cat(alphas))

    # We need to return the indices to prune in we case we decide to prune according to these indices after restoring
    return accuracy, indices_to_prune

def get_data_loaders(dataset, batch_size, workers):
    train_dataset = get_dataset(dataset, 'train')
    test_dataset = get_dataset(dataset, 'test')

    pin_memory = (dataset == "imagenet")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=workers, pin_memory=pin_memory)

    return train_loader, test_loader

def load_model(arch, dataset, model_checkpoint_path, args):
    model = get_architecture(arch, dataset, device, args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def optimize_and_reduce(snl_model, teacher_model, train_loader, test_loader, relu_target, random_tries, 
    reduce_threshold, reduce_step, out_dir, finetune_epochs):

    info_gen = Resnet18ReLUInfo(snl_model)
    current_accuracy = get_training_accuracy(snl_model, train_loader)
    alpha_count = get_relu_count(snl_model)
    stage = 0

    log_file_path = os.path.join(out_dir, LOG_FILE_NAME)
    print(f'logging in {log_file_path}')

    while alpha_count > relu_target:

        log(log_file_path, f"Current ReLU count: {alpha_count}")
        prev_accuracy = current_accuracy
        prune_by_threshold = False
        diff_indices_dict = {}

        for _ in range(random_tries):
            current_accuracy, indices = get_accuracy_for_random_prune(snl_model, train_loader, reduce_step)
            diff = prev_accuracy - current_accuracy

            if (prev_accuracy - current_accuracy) < reduce_threshold:
                prune_by_threshold = True
                break

            diff_indices_dict[diff] = indices

        if prune_by_threshold:
            indices_to_prune = indices
            log(log_file_path, f'[Stage: {stage:04d}] pruning by threshold...')
        else:
            indices_to_prune = diff_indices_dict[min(diff_indices_dict.keys())]
            log(log_file_path, f'[Stage: {stage:04d}] pruning by min diff...')

        # print stats
        stats = [info_gen.get_relu_stats(index) for index in indices_to_prune]
        log(log_file_path, f"[Stage: {stage:04d}] Pruned indices statistics: {Counter(stats)}")

        do_prune_by_indices(snl_model, indices_to_prune)
        snl_model = finetune_student_model(snl_model, teacher_model, train_loader, test_loader, out_dir, epochs=finetune_epochs)

        alpha_count = get_relu_count(snl_model)
        log(log_file_path, f'[Stage: {stage:04d}] Current budget = {alpha_count}')
        test_acc = model_inference(snl_model, test_loader, device, display=False)
        log(log_file_path, f"[Stage: {stage:04d}] Test acc: {test_acc} @ Budget: {alpha_count}")
        stage += 1

    inference_score = model_inference(snl_model, test_loader, device, display=False)
    log(log_file_path, f"Final test score: {inference_score}")

    torch.save(snl_model.state_dict(), os.path.join(out_dir, "optimized.pth"))

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, args.workers)

    teacher_model = load_model(args.arch, args.dataset, args.teacher_model_checkpoint_path, args)
    teacher_model.eval()

    snl_model = load_model(args.arch, args.dataset, args.snl_model_checkpoint_path, args)
    snl_model.eval()

    optimize_and_reduce(snl_model, teacher_model, train_loader, test_loader, args.relu_target, args.random_tries,
     args.reduce_threshold, args.step_reduce, args.output_dir, finetune_epochs=args.finetune_epochs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
