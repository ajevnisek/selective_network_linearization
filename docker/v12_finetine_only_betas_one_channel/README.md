# Docker instructions
1. Build the docker:
```shell
docker build -t snl-amir:v12 -f docker/v12_finetine_only_betas_one_channel/Dockerfile .
docker tag snl-amir:v12 ajevnisek/snl-amir:v12
docker push ajevnisek/snl-amir:v12
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/finetune_only_betas_one_channel/exports/freeze_only_alphas_little_noise_local.sh  -e RUN_SCRIPT=/local_code/scripts/finetune_only_betas_one_channel/debug_runner.sh  -it ajevnisek/snl-amir:v12
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/finetune_only_betas_one_channel/exports/freeze_only_alphas_little_noise_local.sh  -e RUN_SCRIPT=/local_code/scripts/finetune_only_betas_one_channel/finetune_only_betas_one_channel_runner.sh  -it ajevnisek/snl-amir:v12
docker run --gpus all -v $(pwd):/local_code/ -e EXPORT_SCRIPT=/local_code/scripts/alternating_alphas_betas_and_gammas/exports/freeze_alphas_and_weights_little_noise_local.sh  -e RUN_SCRIPT=/local_code/scripts/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh  -it ajevnisek/snl-amir:v12

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
runai submit --name amir-alternate-ab-freeze-a-and-w-little-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-zero-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-little-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-zero-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-loads-of-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_loads_of_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-a-and-w-xavier-init -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_alphas_and_weights_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner_xavier_init.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-freeze-only-a-xavier-init -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/exports/freeze_only_alphas_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_and_betas/alternating_alphas_and_betas_runner_xavier_init.sh --pvc=storage:/storage


runai delete job amir-alternate-ab-and-c-freeze-a-and-w-little-noise
runai delete job amir-alternate-ab-and-c-freeze-a-and-w-zero-noise
runai delete job amir-alternate-ab-and-c-freeze-only-a-little-noise
runai delete job amir-alternate-ab-and-c-freeze-only-a-zero-noise
runai delete job amir-alternate-ab-and-c-freeze-a-and-w-loads-of-noise
runai delete job amir-alternate-ab-and-c-freeze-a-and-w-xavier-init
runai delete job amir-alternate-ab-and-c-freeze-only-a-xavier-init
runai submit --name amir-alternate-ab-and-c-freeze-a-and-w-little-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_alphas_and_weights_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-a-and-w-zero-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_alphas_and_weights_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-only-a-little-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_only_alphas_little_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-only-a-zero-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_only_alphas_zero_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-a-and-w-loads-of-noise -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_alphas_and_weights_loads_of_noise.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-a-and-w-xavier-init -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_alphas_and_weights_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner_xavier_init.sh --pvc=storage:/storage
runai submit --name amir-alternate-ab-and-c-freeze-only-a-xavier-init -g 1.0 -i ajevnisek/snl-amir:v12 -e EXPORT_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/exports/freeze_only_alphas_xavier_init.sh -e RUN_SCRIPT=/storage/jevnisek/snl/scripts_and_configs/alternating_alphas_betas_and_gammas/alternating_alphas_betas_and_gammas_runner_xavier_init.sh --pvc=storage:/storage

```

