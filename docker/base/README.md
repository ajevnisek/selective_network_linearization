# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v0 -f docker/base/Dockerfile .
docker tag snl-amir:v0 ajevnisek/snl-amir:v0
docker push ajevnisek/snl-amir:v0
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/  -it snl-amir:v0 /bin/bash
```
On runai:
```shell
runai submit --name amir-nlr-demo -g 0.5 -i ajevnisek/our_method_non_local_relu:v1 -e CONFIG_YAML=/storage/jevnisek/non_local_relu/configs/vanilla_5k_runai.yaml --pvc=storage:/storage
```
