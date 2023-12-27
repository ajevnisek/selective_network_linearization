# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v2 -f docker/v2_snl_with_betas/Dockerfile .
docker tag snl-amir:v2 ajevnisek/snl-amir:v2
docker push ajevnisek/snl-amir:v2
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/snl_with_betas/cifar100-new-split/snl_with_betas_15000_local.sh -e RUN_SCRIPT=/local_code/scripts/snl_with_alphas_and_betas_generic.sh  -it snl-amir:v2
```
On runai:
```shell
runai submit --name amir-snl-with-betas -g 1.0 -i ajevnisek/snl-amir:v2 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_with_betas/original_cifar100/snl_with_betas_15000.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_with_alphas_and_betas_generic.sh --pvc=storage:/storage
```

