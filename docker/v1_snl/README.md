# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v1 -f docker/v1_snl/Dockerfile .
docker tag snl-amir:v1 ajevnisek/snl-amir:v1
docker push ajevnisek/snl-amir:v1
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/example_exports.sh -e RUN_SCRIPT=/local_code/scripts/snl_resnet_single_line.sh  -it snl-amir:v1
```
On runai:
```shell
runai submit --name amir-snl-9k -g 1.0 -i ajevnisek/snl-amir:v1 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_9000.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl_resnet_single_line.sh --pvc=storage:/storage
```
