# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v14 -f docker/v14_relu_autoencoder_multiple_layers_at_once/Dockerfile .
docker tag snl-amir:v14 ajevnisek/snl-amir:v14
docker push ajevnisek/snl-amir:v14
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_local.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/debug_runner.sh  -it ajevnisek/snl-amir:v14 /bin/bash
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_local.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/relus_autoencoder_runner_multiple_layers.sh  -it ajevnisek/snl-amir:v14

```
On runai:
```shell
runai submit --name amir-relus-autoencoder-all-at-once -g 1.0 -i ajevnisek/snl-amir:v14 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_all_at_once.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner_multiple_layers_all_at_once.sh --pvc=storage:/storage
```

