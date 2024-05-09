# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v2_snl_with_cka -f docker/v2_snl_with_cka/Dockerfile .
docker tag snl-amir:v2_snl_with_cka ajevnisek/snl-amir:v2_snl_with_cka
docker push ajevnisek/snl-amir:v2_snl_with_cka
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/example_exports_snl_with_cka.sh -e RUN_SCRIPT=/local_code/scripts/snl_with_cka_resnet_single_line_local.sh  -it snl-amir:v2_snl_with_cka
```
On runai:
```shell
runai submit --name amir-snl-9k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/vanilla_snl/snl_9000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_resnet_single_line.sh --pvc=storage:/storage
runai submit --name amir-snl-15k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/vanilla_snl/snl_15000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_resnet_single_line.sh --pvc=storage:/storage
runai submit --name amir-snl-30k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/vanilla_snl/snl_30000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_resnet_single_line.sh --pvc=storage:/storage

runai submit --name amir-snl-with-cka-9k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_9000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage
runai submit --name amir-snl-with-cka-15k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_15000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage
runai submit --name amir-snl-with-cka-30k -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_30000.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage

runai submit --name amir-snl-with-cka-9k-larger-weight -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_9000_higher_factor.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage
runai submit --name amir-snl-with-cka-15k-larger-weight -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_15000_higher_factor.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage
runai submit --name amir-snl-with-cka-30k-larger-weight -g 1.0 -i ajevnisek/snl-amir:v2_snl_with_cka -e EXPORT_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/snl_with_cka/snl_with_cka_30000_higher_factor.sh -e RUN_SCRIPT=/storage/jevnisek/snl_with_cka/scripts_and_configs/runners/snl_with_cka_resnet_single_line_remote.sh --pvc=storage:/storage
```
