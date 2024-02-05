# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v7 -f docker/v7_squash_betas/Dockerfile .
docker tag snl-amir:v7 ajevnisek/snl-amir:v7
docker push ajevnisek/snl-amir:v7
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/snl_abc_with_squash/snl_abc_with_squash_start_30k_end_29k.sh -e RUN_SCRIPT=/local_code/scripts/snl_abc_with_squash.sh  -it ajevnisek/snl-amir:v7
```
On runai:
```shell
runai submit --name amir-snl-abc-squash-9k -g 1.0 -i ajevnisek/snl-amir:v7 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/exports/snl_abc_with_squash_start_30k_end_9k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/snl_abc_with_squash.sh --pvc=storage:/storage
runai submit --name amir-snl-abc-squash-15k -g 1.0 -i ajevnisek/snl-amir:v7 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/exports/snl_abc_with_squash_start_30k_end_15k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/snl_abc_with_squash.sh --pvc=storage:/storage
runai submit --name amir-snl-abc-squash-20k -g 1.0 -i ajevnisek/snl-amir:v7 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/exports/snl_abc_with_squash_start_30k_end_20k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/snl_abc_with_squash.sh --pvc=storage:/storage
runai submit --name amir-snl-abc-squash-29k -g 1.0 -i ajevnisek/snl-amir:v7 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/exports/snl_abc_with_squash_start_30k_end_29k.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/snl-abc-squash/snl_abc_with_squash.sh --pvc=storage:/storage

```

