import os

ROOT = 'scripts/optimize_and_reduce/exports'


def generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target):
	return f"""
SNL_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100_30000.pth
TEACHER_MODEL_CHECKPOINT_PATH=/storage/vladi/snl/optimize_and_reduce/checkpoints/resnet18_cifar100.pth


REDUCE_THRESHOLD={reduce_threshold}
STEP_REDUCE={step_reduce}
RANDOM_TRIES={random_tries}
RELU_TARGET={relu_target}


OUTPUT_DIR=/storage/vladi/snl/optimize_and_reduce/outputs/RELU_TARGET_$RELU_TARGET-REDUCE_THRESHOLD_$REDUCE_THRESHOLD-RANDOM_TRIES_$RANDOM_TRIES-STEP_REDUCE_$STEP_REDUCE
"""

def generate_runai_command(job_index, export_filename):
	return f"""runai submit --name vladi-onr-{job_index} --gpu 1.0 -e EXPORT_SCRIPT=/storage/vladi/snl/optimize_and_reduce/scripts/{export_filename} -e RUN_SCRIPT=/storage/vladi/snl/optimize_and_reduce/scripts/optimize_and_reduce_runner.sh  -i ajevnisek/snl-amir:v4 --pvc=storage:/storage --large-shm"""


def generate_files():
	runai_bash_lines = []
	counter = 0
	for reduce_threshold in [0.5, 1.0, 1.5]:
		for step_reduce in [100, 200, 500, 1000]:
			for random_tries in [20, 50, 100]:
				for relu_target in [9000, 15000, 20000]:
					filename = f"relu_target_{relu_target}-step_reduce_{step_reduce}-random_tries_{random_tries}-reduce_threshold_{reduce_threshold}.sh"
					runai_bash_lines.append(generate_runai_command(counter, filename))
					with open(os.path.join(ROOT, filename), 'w') as f:
						f.write(generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target))
					counter += 1
	with open('scripts/optimize_and_reduce/runai_bash_lines.sh', 'w') as f:
		f.write('\n'.join(runai_bash_lines))



if __name__ == '__main__':
	generate_files()

