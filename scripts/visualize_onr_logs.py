import argparse
import os
import matplotlib
import matplotlib.pyplot as plt

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('onr_logs_dir', type=str, help='O&R runai logs directory')
	parser.add_argument('output_dir', type=str, help='the directory to which the visualizations will be saved')

	return parser.parse_args()

def extract_data_from_log_file(log_file_path):
	with open(log_file_path, 'r') as f:
		lines = f.readlines()

	if len(lines) == 0:
		return

	params_line = lines[10]
	accuracy_line = lines[-1]

	dash_split = params_line.split('-')

	relu_target = int(dash_split[0].split('_')[-1])
	reduce_threshold = float(dash_split[1].split('_')[-1])
	random_tries = int(dash_split[2].split('_')[-1])
	reduce_step = int(dash_split[3].split('_')[-1].split('/')[0])
	accuracy = float(accuracy_line.split(' ')[-1][: -1])

	return relu_target, reduce_threshold, random_tries, reduce_step, accuracy

def visualize_stem_for_budget(budget, budget_data_list, output_dir):
	x_str_list = []
	accuracies = []
	for elem in budget_data_list:
		accuracy, reduce_threshold, random_tries, reduce_step = elem
		curr_x_str = f'{reduce_threshold}_{random_tries}_{reduce_step}'
		x_str_list.append(curr_x_str)
		accuracies.append(accuracy)

	plt.figure(figsize=(15, 10))
	matplotlib.rc('xtick', labelsize=5)
	matplotlib.rc('ytick', labelsize=7)
	plt.stem(x_str_list, accuracies)
	plt.tight_layout()
	plt.title(f'O&R experiments for budget of {budget} ReLUs')
	plt.xlabel('Configuration')
	plt.ylabel('Accuracy')
	plt.yticks(range(1, 70, 2))
	#plt.subplots_adjust(bottom=0.01, top=0.01,left=0.01)
	plt.savefig(os.path.join(output_dir, f'onr_plot_{budget}_relus.png'), bbox_inches='tight')
	plt.clf()

def process_log_files(onr_logs_dir, output_dir):
	data_dict = {}
	file_names = os.listdir(onr_logs_dir)

	for file_name in file_names:
		curr_log_file_path = os.path.join(onr_logs_dir, file_name)
		extracted_data = extract_data_from_log_file(curr_log_file_path)
		if extracted_data is None:
			continue

		relu_target, reduce_threshold, random_tries, reduce_step, accuracy = extracted_data

		dict_element = (accuracy, reduce_threshold, random_tries, reduce_step)
		if relu_target not in data_dict:
			data_dict[relu_target] = [dict_element]
		else:
			data_dict[relu_target].append(dict_element)

	for budget in data_dict.keys():
		visualize_stem_for_budget(budget, data_dict[budget], output_dir)


def main(args):
	os.makedirs(args.output_dir, exist_ok=True)
	process_log_files(args.onr_logs_dir, args.output_dir)

if __name__ == "__main__":
	args = parse_args()
	main(args)