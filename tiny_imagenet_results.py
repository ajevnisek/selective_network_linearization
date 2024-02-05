import os
import matplotlib.pyplot as plt


snl = [
        (12.9, 66.53),
        (14.9, 67.17),
        (25.0, 70.05),
        (29.8, 70.18),
        (40.0, 72.35),
        (49.4, 73.75),
        (60.0, 74.85),
        (69.8, 75.29),
        (79.1, 75.72),
        (99.9, 75.94),
    (120.0, 76.35),
    (150.0, 77.35),
    (180.0, 77.65),
    (248.4, 77.7),
]
deepreduce = [
    (12.3, 64.97),
    (28.7, 68.68),
    (49.2, 69.50),
    (57.34, 72.68),
    (114.0, 74.72),
    (197.0, 75.51),
    (229.38, 76.22),

]
sphnx = [
(25.6, 66.13),
            (30.2, 67.37),
            (41.0, 68.23),
            (51.2, 69.57),
            (71.7, 71.06),
            (102.4, 72.90),
            (230, 74.93),
]
cryptonas = [
    (50, 63.5),
    (86, 66),
    (100, 68.5),
    (334, 76),
    (500, 77),
]
snip = [
            (12.9, 56.83),
            (14.9, 56.92),
            (25.0, 60.88),
            (29.8, 62.20),
            (40.0, 61.98),
            (49.4, 63.69),
            (60.0, 64.89),
            (69.8, 64.98),
            (79.1, 65.50),
            (99.9, 66.50),
]
our_method = [
    (6, 57.65),
    (9, 62.8),
    (12.9, 66.53),
    (15, 67.17),
    (15.8, 67.94),
    (25.0, 70.05),
    (29.8, 70.18),

]

plt.close('all')
plt.plot([relu_budget for (relu_budget, _) in our_method],
         [acc for (_, acc) in our_method], '-*b', markersize=10,  linewidth=3, label='Our', )
plt.plot([relu_budget for (relu_budget, _) in snl],
         [acc for (_, acc) in snl], '--xr', linewidth=1.5, label='SNL')
plt.plot([relu_budget for (relu_budget, _) in deepreduce],
         [acc for (_, acc) in deepreduce],  '--^g', linewidth=1.5, label='DeepReduce')
plt.plot([relu_budget for (relu_budget, _) in sphnx],
         [acc for (_, acc) in sphnx],  '--Dk', linewidth=1.5, label='Sphynx')
plt.plot([relu_budget for (relu_budget, _) in cryptonas],
         [acc for (_, acc) in cryptonas],  '--dm', linewidth=1.5, label='CryptoNAS')
plt.plot([relu_budget for (relu_budget, _) in snip],
         [acc for (_, acc) in snip],  '--4c', linewidth=1.5, label='SNIP')
plt.xscale('log')
plt.xticks([2**i for i in range(2, 10)], [2**i for i in range(2, 10)])
plt.grid(True)
plt.xlabel('ReLU Budget [K]')
plt.ylabel('Accuracy [%]')
# plt.legend(bbox_to_anchor=(0.0, 1.03), loc="lower left", borderaxespad=0, ncols=4)
plt.legend()
plt.title('CIFAR-100')
plt.tight_layout()
plt.savefig('/home/adminubuntu/Desktop/cifar100_results.png')
plt.savefig('/home/adminubuntu/Desktop/cifar100_results.pdf')

plt.show()