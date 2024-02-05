import os
import matplotlib.pyplot as plt


snl = [
    (60, 54.24),
    (100, 58.94),
    (200, 63.39),
    (300, 64.04),
    (400, 63.83),
    (500, 64.42),
]
deepreduce = [
             (12.9, 41.95),
             (24.6, 47.01),
             (28.67, 47.55),
             (49.16, 49.00),
             (57.35, 53.75),
             (98.3, 55.67),
             (114.69, 56.18),
             (200.0, 57.51),
             (230.0, 59.18),
             (400.0, 61.65),
             (459.0, 62.26),
             (917.0, 64.66),

]
sphnx = [
(102.4, 48.44),
            (204.8, 53.51),
            (286.7, 56.72),
            (491.5, 59.12),
            (614.4, 60.76),
]
our_method = [
    (59.1, 56.6),

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
plt.xscale('log')
plt.xticks([2**i for i in range(5, 11)], [2**i for i in range(5, 11)])
plt.grid(True)
plt.xlabel('ReLU Budget [K]')
plt.ylabel('Accuracy [%]')
# plt.legend(bbox_to_anchor=(0.0, 1.03), loc="lower left", borderaxespad=0, ncols=4)
plt.legend()
plt.title('Tiny-ImageNet')
plt.tight_layout()
plt.savefig('/home/adminubuntu/Desktop/tiny_imagenet_results.png')
plt.savefig('/home/adminubuntu/Desktop/tiny_imagenet_results.pdf')

plt.show()