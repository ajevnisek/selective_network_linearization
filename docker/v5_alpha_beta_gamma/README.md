# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v5 -f docker/v5_alpha_beta_gamma/Dockerfile .
docker tag snl-amir:v5 ajevnisek/snl-amir:v5
docker push ajevnisek/snl-amir:v5
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/snl_alpha_beta_gamma_config.sh -e RUN_SCRIPT=/local_code/scripts/snl_alpha_beta_gamma.sh  -it ajevnisek/snl-amir:v5
```
On runai:
```shell
runai submit --name amir-snl-with-betas -g 1.0 -i ajevnisek/snl-amir:v5 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-alpha-beta-gamma/configs/snl_alpha_beta_gamma_30k_to_29k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-alpha-beta-gamma/snl_alpha_beta_gamma_runner.sh --pvc=storage:/storage
```

