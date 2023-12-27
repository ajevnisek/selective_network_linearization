import argparse, pickle
import matplotlib.pyplot as plt
from math import sqrt
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Plot FSP like algorithm results')
    parser.add_argument('correlation_threshold', type=float, help='Correlation threshold.')
    return parser.parse_args()



def get_layernames():
    layernames = []
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in [0, 1]:
            for relu_idx in [1, 2]:
                layer_name = f'{layer}_block{block}_relu{relu_idx}'
                layernames.append(layer_name)
    return layernames

def main():
    correlation_threshold = parse_args().correlation_threshold
    inducer_to_purity_score = OrderedDict({})
    inducer_to_num_inducees = OrderedDict({})
    for layername in get_layernames():
        path = f'associate_prototypes/cifar100/resnet18_in/correlation_threshold_{correlation_threshold}/associations/{layername}/inducer_to_inducee_in_channel_row_and_col_terminology.pkl'
        d = pickle.load(open(path, 'rb'))
        for k in d:
            inducer_to_purity_score[f"{layername}_{k}"] = len([v for v in d[k] if v[0] == k[0]]) / len(d[k]) * 1.0
            inducer_to_num_inducees[f"{layername}_{k}"] = len(d[k])


    plt.close('all')
    plt.suptitle(f'Purity score = num of inducees in the same channel / num of overall inducees\ncorrelation_threshold={correlation_threshold:.2f}.')
    plt.subplot(2, 2, 1)
    plt.scatter(inducer_to_num_inducees.values(), inducer_to_purity_score.values())
    plt.xscale('log')
    plt.xlabel('num inducees')
    plt.ylabel('purity score')
    plt.title('purity score vs num inducees')
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.hist(inducer_to_num_inducees.values(), bins=int(sqrt(len(inducer_to_num_inducees.values()))))
    plt.title('num inducees histogram')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('num inducees')
    plt.ylabel('count [#]')
    plt.grid(True)
    if correlation_threshold == 0.7:
        plt.xticks([1, 10, 20, 30, 100, 150, 400, 1000], [1, 10, 20, 30, 100, 150, 400,  1000], rotation=-45)
    plt.subplot(2, 2, 3)
    plt.hist(inducer_to_purity_score.values(), bins=int(sqrt(len(inducer_to_purity_score.values()))), label='count')
    plt.title('purity scores histogram')
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('purity scores')
    plt.ylabel('count [#]')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.hist(inducer_to_purity_score.values(), bins=int(sqrt(len(inducer_to_purity_score.values()))), cumulative=True,
             density=True, label='CDF')
    plt.title('purity scores histogram')
    # plt.xscale('log')
    plt.xlabel('purity scores')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((10, 10))
    plt.tight_layout()
    plt.savefig(f'associate_prototypes/cifar100/resnet18_in/correlation_threshold_{correlation_threshold}/purity_scores_{correlation_threshold}.png')


if __name__ == '__main__':
    main()

