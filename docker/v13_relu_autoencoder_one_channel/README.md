# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v13 -f docker/v13_relu_autoencoder_one_channel/Dockerfile .
docker tag snl-amir:v13 ajevnisek/snl-amir:v13
docker push ajevnisek/snl-amir:v13
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/debug_runner.sh  -it ajevnisek/snl-amir:v13 /bin/bash
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer.sh  -e RUN_SCRIPT=/local_code/scripts/relu_autoencoder_one_channel/relus_autoencoder_runner.sh  -it ajevnisek/snl-amir:v13

```
On runai:
```shell
runai submit --name amir-relus-autoencoder-layer1-1-2-smooth-drelu -g 1.0 -i ajevnisek/snl-amir:v13 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer_1_1_2_smooth-drelu.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner.sh --pvc=storage:/storage
runai submit --name amir-relus-autoencoder-layer1-1-2-drelu -g 1.0 -i ajevnisek/snl-amir:v13 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer_1_1_2_drelu.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner.sh --pvc=storage:/storage
runai submit --name amir-relus-autoencoder-layer1-1-2-relu -g 1.0 -i ajevnisek/snl-amir:v13 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer_1_1_2_relu.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner.sh --pvc=storage:/storage
runai submit --name amir-relus-autoencoder-layer1-1-2-smooth-drelu-medium-budget -g 1.0 -i ajevnisek/snl-amir:v13 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer_1_1_2_smooth-drelu_medium_budget.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner.sh --pvc=storage:/storage
runai submit --name amir-relus-autoencoder-layer1-1-2-smooth-drelu-large-budget -g 1.0 -i ajevnisek/snl-amir:v13 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/exports/relu_autoencoder_single_layer_1_1_2_smooth-drelu_larger_budget.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/relu_autoencoder_one_channel/relus_autoencoder_runner.sh --pvc=storage:/storage
```

