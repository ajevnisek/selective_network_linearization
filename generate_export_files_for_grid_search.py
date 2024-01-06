import os

ROOT = 'scripts/optimize_and_reduce/exports'


def generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target):
	return f"""SNL_MODEL_CHECKPOINT_PATH=checkpoints/resnet18_cifar100_30000.pth
TEACHER_MODEL_CHECKPOINT_PATH=checkpoints/resnet18_cifar100.pth
OUTPUT_DIR=checkpoints/docker_sandbox/optimize_and_reduce

REDUCE_THRESHOLD={reduce_threshold}
STEP_REDUCE={step_reduce}
RANDOM_TRIES={random_tries}
RELU_TARGET={relu_target}
"""

def generate_files():
	for reduce_threshold in [0.5, 1.0, 1.5]:
		for step_reduce in [100, 200, 500, 1000]:
			for random_tries in [20, 50, 100]:
				for relu_target in [9000, 15000, 20000]:
					filename = f"relu_target_{relu_target}-step_reduce_{step_reduce}-random_tries_{random_tries}-reduce_threshold_{reduce_threshold}.sh"
					with open(os.path.join(ROOT, filename), 'w') as f:
						f.write(generate_filecontent(reduce_threshold, step_reduce, random_tries, relu_target))

if __name__ == '__main__':
	generate_files()
