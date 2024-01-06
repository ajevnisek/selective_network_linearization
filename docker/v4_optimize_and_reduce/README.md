# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v4 -f docker/v4_optimize_and_reduce/Dockerfile .
docker tag snl-amir:v4 ajevnisek/snl-amir:v4
docker push ajevnisek/snl-amir:v4
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/optimize_and_reduce/exports/relu_target_15000-step_reduce_100-random_tries_100-reduce_threshold_0.5.sh -e RUN_SCRIPT=/local_code/scripts/optimize_and_reduce/optimize_and_reduce_runner.sh  -it ajevnisek/snl-amir:v4
```
On runai:
```shell
runai submit --name amir-snl-with-betas -g 1.0 -i ajevnisek/snl-amir:v3 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_with_betas/original_cifar100/snl_with_betas_15000.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_with_alphas_and_betas_generic.sh --pvc=storage:/storage
```

