import os

ROOT = 'scripts/optimize_and_reduce/exports'


def generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target):
	return f"""
SNL_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_tiny_imagenet_80000.pth
TEACHER_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_tiny_imagenet.pth


REDUCE_THRESHOLD={reduce_threshold}
STEP_REDUCE={step_reduce}
RANDOM_TRIES={random_tries}
RELU_TARGET={relu_target}


OUTPUT_DIR=/storage/vladi/snl/optimize_and_reduce/outputs/TINY_IMAGENET_RELU_TARGET_$RELU_TARGET-REDUCE_THRESHOLD_$REDUCE_THRESHOLD-RANDOM_TRIES_$RANDOM_TRIES-STEP_REDUCE_$STEP_REDUCE
"""

def generate_runai_command(job_index, export_filename):
	return f"""runai submit --name vladi-onr-tiny-imagenet-greedier-{job_index} --gpu 1.0 -e EXPORT_SCRIPT=/storage/vladi/snl/optimize_and_reduce/scripts/{export_filename} -e RUN_SCRIPT=/storage/vladi/snl/optimize_and_reduce/scripts/optimize_and_reduce_greedier_tiny_imagenet_runner.sh  -i ajevnisek/snl-amir:v10 --pvc=storage:/storage --large-shm"""


def generate_files():
	runai_bash_lines = []
	counter = 0
	for reduce_threshold in [0.3]:
		for step_reduce in [100]:
			for random_tries in [50]:
				for relu_target in [59100]:
					filename = f"relu_target_{relu_target}-step_reduce_{step_reduce}-random_tries_{random_tries}-reduce_threshold_{reduce_threshold}_tiny_imagenet_greedier.sh"
					runai_bash_lines.append(generate_runai_command(counter, filename))
					with open(os.path.join(ROOT, filename), 'w') as f:
						f.write(generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target))
					counter += 1
	with open('scripts/optimize_and_reduce/runai_bash_lines_tiny_imagenet_greedier.sh', 'w') as f:
		f.write('\n'.join(runai_bash_lines))



if __name__ == '__main__':
	generate_files()

