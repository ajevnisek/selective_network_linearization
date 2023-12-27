import os
from pprint import pprint
from collections import OrderedDict


def get_relu_budget_from_file(data):
    return float([l for l in data.splitlines() if l.startswith('After SNL Algorithm, the current ReLU Count:')][0].split(': ')[-1].split(',')[0])


def get_accuracy_from_file(data):
    return float([l for l in data.splitlines() if l.startswith('Final best Prec@1 = ')][0].split(' = ')[-1].split('%')[0])


root = 'snl_run_results_from_runai/cifar100/resnet18_in/'
# root = 'snl_run_results_from_runai/cifar100-new-split/resnet18_in/'
budgets = os.listdir(root)
budgets.sort(key=lambda x: int(x))

relu_budgets_to_accs = OrderedDict({})
for b in budgets:
    filepath = os.path.join(root, b, f'resnet18_in_unstructured_{b}.txt')
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r') as f:
        data = f.read()
    try:
        relu_budget = get_relu_budget_from_file(data)
        accuracy = get_accuracy_from_file(data)
    except:
        print(f"Skipping {filepath}...")
    relu_budgets_to_accs[relu_budget] = accuracy

pprint(relu_budgets_to_accs)
