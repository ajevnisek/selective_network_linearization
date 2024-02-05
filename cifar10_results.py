import os
import matplotlib.pyplot as plt


snl = [
    (12.9, 88.23),
    (14.9, 88.43),
    (25.0, 90.88),
    (29.8, 90.92),
    (40.0, 91.68),
    (49.4, 92.27),
    (60.0, 92.63),
    (69.8, 93.02),
    (79.1, 93.16),
    (99.9, 93.50),
    (150.0, 94.26),
    (180.0, 94.78),
    (300, 95.06),
    (400, 95.07),
    (500, 95.21),
]
deepreduce = [
            (36.0 , 88.5),
            (70.0, 90.0),
            (80.0, 90.5),
            # (99.8,89.32),
            # (99.9,86.7),
            (114.0, 92.7),
            (147.0, 93.16),
            (221.48,94.07),

]
cryptonas = [
            (50, 90.0),
            (86, 91.5),
            (100, 92.2),
            (334, 94),
            (500, 94.8),
]
our_method = [
    (6, 85.65),
    (9, 87.49),
    (14.5, 89.65),

]

plt.close('all')
plt.plot([relu_budget for (relu_budget, _) in our_method],
         [acc for (_, acc) in our_method], '-*b', markersize=10, linewidth=3, label='Our', )
plt.plot([relu_budget for (relu_budget, _) in snl],
         [acc for (_, acc) in snl], '--xr', linewidth=1.5, label='SNL')
plt.plot([relu_budget for (relu_budget, _) in deepreduce],
         [acc for (_, acc) in deepreduce],  '--^g', linewidth=1.5, label='DeepReduce')
plt.plot([relu_budget for (relu_budget, _) in cryptonas],
         [acc for (_, acc) in cryptonas],  '--Dk', linewidth=1.5, label='CryptoNAS')
plt.xscale('log')
plt.xticks([2**i for i in range(2, 10)], [2**i for i in range(2, 10)])
plt.grid(True)
plt.xlabel('ReLU Budget [K]')
plt.ylabel('Accuracy [%]')
# plt.legend(bbox_to_anchor=(0.0, 1.03), loc="lower left", borderaxespad=0, ncols=4)
plt.legend()
plt.title('CIFAR-10')
plt.tight_layout()
plt.savefig('/home/adminubuntu/Desktop/cifar10_results.png')
plt.show()