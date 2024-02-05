# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v8 -f docker/v8_alternating_alphas_and_beta/Dockerfile .
docker tag snl-amir:v8 ajevnisek/snl-amir:v8
docker push ajevnisek/snl-amir:v8
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/exports/snl_abc_with_squash/snl_abc_with_squash_start_30k_end_29k.sh -e RUN_SCRIPT=/local_code/scripts/snl_abc_with_squash.sh  -it ajevnisek/snl-amir:v8
```
On runai:
```shell
runai delete job amir-alternate-ab-freeze-a-and-w-little-noise
runai delete job amir-alternate-ab-freeze-a-and-w-zero-noise
runai delete job amir-alternate-ab-freeze-only-a-little-noise
runai delete job amir-alternate-ab-freeze-only-a-zero-noise
runai delete job amir-alternate-ab-freeze-a-and-w-loads-of-noise
runai delete job amir-alternate-ab-freeze-a-and-w-xavier-init
runai delete job amir-alternate-ab-freeze-only-a-xavier-init
runai submit --name amir-alternate-ab-freeze-a-and-w-little-noise -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-zero-noise -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-little-noise -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-zero-noise -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-loads-of-noise -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_loads_of_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-xavier-init -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner_xavier_init.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-xavier-init -g 1.0 -i ajevnisek/snl-amir:v8 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner_xavier_init.sh --pvc=storage:/storage

```

