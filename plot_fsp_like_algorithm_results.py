import argparse
import numpy as np
import matplotlib.pyplot as plt

from math import ceil, floor
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Plot FSP like algorithm results')
    parser.add_argument('logfile', help='path to log file.')
    return parser.parse_args()


def main():
    path = parse_args().logfile
    with open(path, 'r') as f:
        data = f.read()

    after_finetuning = OrderedDict({})
    induce_reduction = [[], []]
    for line in data.splitlines():
        if 'in InducedReLU and finetuning, accuracy is:' in line:
            layername = line.split(' ')[2]
            accuracy = float(line.split(' ')[-2])
            after_finetuning[layername] = accuracy
        if 'Our FSP-like algorithm reduced the number of prototypes from' in line:
            induce_reduction[0].append(int(line.split(' ')[-3]))
            induce_reduction[1].append(int(line.split(' ')[-1]))


    plt.close('all')
    plt.suptitle(path.split('/')[-2].replace('_', ' ').capitalize())
    plt.subplot(3, 1, 1)
    plt.title('accuracy [%] vs the layer we currently replace')
    plt.plot(after_finetuning.values())
    plt.xticks(range(len(after_finetuning.keys())), after_finetuning.keys(), rotation='vertical')
    plt.ylabel('accuracy [%]')
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.bar(range(len(induce_reduction[0])), induce_reduction[0], width=0.3, label='before=SNL')
    #plt.bar([x + 0.3 for x in range(len(induce_reduction[1]))], induce_reduction[1], width=0.3, label='before')
    plt.xticks(range(len(induce_reduction[0])), after_finetuning.keys(), rotation='vertical')
    plt.bar([x + 0.3 for x in range(len(induce_reduction[1]))], induce_reduction[1], width=0.3, label='after=Ours', color='g')
    plt.legend()
    plt.grid(True)
    plt.ylabel('num prototypes')
    plt.title('num prototypes before and after our FPS-like algorithm')
    plt.subplot(3, 1, 3)
    plt.title('Number of ReLUs in the network')
    plt.ylabel('ReLU budget [#]')
    import numpy as np
    plt.plot([sum(induce_reduction[0])] * len(induce_reduction[0]), 'b', linewidth=2, label='before=SNL')
    plt.plot(np.cumsum(induce_reduction[1]) - np.cumsum(induce_reduction[0]) + sum(induce_reduction[0]), 'g', linewidth=2, label='after=Ours')
    plt.xticks(range(len(induce_reduction[0])), after_finetuning.keys(), rotation='vertical')
    max_value = max(sum(induce_reduction[0]),
                    max(np.cumsum(induce_reduction[1]) - np.cumsum(induce_reduction[0]) + sum(
                        induce_reduction[0])))
    min_value = min(sum(induce_reduction[0]),
                    min(np.cumsum(induce_reduction[1]) - np.cumsum(induce_reduction[0]) + sum(
                        induce_reduction[0])))
    step = 1e3
    max_ytick = int((ceil(max_value // step) + 1) * step) + step
    min_ytick = int((floor(min_value // 1e3) - 1) * 1e3)
    if (max_ytick - min_ytick ) / step < 4:
        step = 0.5 * step
    plt.yticks(range(int(min_ytick), int(max_ytick), int(step)))
    plt.grid(True)
    fig = plt.gcf()
    fig.set_size_inches((12, 10))
    plt.tight_layout()
    correlation_coeff = path.split('/')[-2].split('_')[-1]
    plt.savefig(path.replace('log.txt', f'accuracy_and_prototypes_reduction_{correlation_coeff}.png'))
    plt.close('all')
    plt.suptitle(path.split('/')[-2].replace('_', ' ').capitalize())
    relu_budget_ours = np.cumsum(induce_reduction[1]) - np.cumsum(induce_reduction[0]) + sum(induce_reduction[0])
    accuracy_ours = list(after_finetuning.values())
    snl_for_cifar100_with_val = [
        (5998.0, 57.13),
        (6988.0, 59.66),
        (7965.0, 60.98),
        (8987.0, 62.28),
        (9910.0, 63.33),
        (11831.0, 65.28),
        (12961.0, 65.51),
        (13974.0, 65.35),
        (14865.0, 66.0),
        # (24936.0, 68.53),
        # (29909.0, 69.12)
    ]
    snl_for_cifar100_original = [
        (5859.0, 57.65),
         (6896.0, 59.5),
         (7867.0, 60.48),
         (8967.0, 62.79),
         (9858.0, 64.16),
         (10796.0, 64.8),
         (11885.0, 65.57),
         (12784.0, 66.02),
         (13873.0, 67.0),
         (14621.0, 67.16)
    ]
    relu_budget_snl_cifar100_original = [x[0] for x in snl_for_cifar100_original]
    accuracy_snl_cifar100_original = [x[1] for x in snl_for_cifar100_original]
    relu_budget_snl_cifar100_with_val = [x[0] for x in snl_for_cifar100_with_val]
    accuracy_snl_cifar100_with_val = [x[1] for x in snl_for_cifar100_with_val]
    plt.close('all')
    plt.suptitle(path.split('/')[-2].replace('_', ' ').capitalize())
    plt.plot(relu_budget_snl_cifar100_original, accuracy_snl_cifar100_original, 'ob', linewidth=2,  markersize=10,
             label="SNL-CIFAR100-Original")
    plt.plot(relu_budget_snl_cifar100_with_val, accuracy_snl_cifar100_with_val, 'sr', linewidth=2, markersize=10,
             label="SNL-CIFAR100-with-Val")
    plt.plot(relu_budget_ours, accuracy_ours, '^g',  linewidth=2,   markersize=10, label="Ours")
    plt.xlabel('ReLU budget [#]')
    plt.ylabel('accuracy [#]')
    max_value = max(accuracy_ours + accuracy_snl_cifar100_original + accuracy_snl_cifar100_with_val)
    min_value = min(accuracy_ours + accuracy_snl_cifar100_original + accuracy_snl_cifar100_with_val)
    plt.yticks(range(floor(min_value), ceil(max_value) + 1, 1))
    # plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.replace('log.txt', f'accuracy_vs_relu_budget_{correlation_coeff}.png'))


if __name__ == '__main__':
    main()
