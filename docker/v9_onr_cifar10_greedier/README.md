# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v9 -f docker/v9_onr_cifar10_greedier/Dockerfile .
docker tag snl-amir:v9 ajevnisek/snl-amir:v9
docker push ajevnisek/snl-amir:v9
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/optimize_and_reduce/exports/relu_target_15000-step_reduce_100-random_tries_100-reduce_threshold_0.5_cifar10_greedier.sh -e RUN_SCRIPT=/local_code/scripts/optimize_and_reduce/optimize_and_reduce_greedier_cifar10_runner.sh  -it ajevnisek/snl-amir:v9
```
On runai:
```shell
runai submit --name amir-snl-with-betas -g 1.0 -i ajevnisek/snl-amir:v6 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-alpha-beta-gamma/configs/snl_alpha_beta_gamma_diagonal_mask_30k_to_20k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-alpha-beta-gamma/snl_alpha_beta_gamma_diagonal_mask_runner.sh --pvc=storage:/storage
```

