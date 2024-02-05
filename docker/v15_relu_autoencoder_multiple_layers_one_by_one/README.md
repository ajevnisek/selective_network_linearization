# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v15 -f docker/v15_relu_autoencoder_multiple_layers_one_by_one/Dockerfile .
docker tag snl-amir:v15 ajevnisek/snl-amir:v15
docker push ajevnisek/snl-amir:v15
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_one_by_one_local.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/debug_runner.sh  -it ajevnisek/snl-amir:v15 /bin/bash
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_one_by_one_local.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/relus_autoencoder_runner_multiple_layers_one_by_one.sh  -it ajevnisek/snl-amir:v15

```
On runai:
```shell
runai submit --name amir-relus-autoencoder-one-by-one -g 1.0 -i ajevnisek/snl-amir:v15 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_multiple_layers_one_by_one.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner_multiple_layers_one_by_one.sh --pvc=storage:/storage
```

