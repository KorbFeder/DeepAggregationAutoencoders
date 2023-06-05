import time
import yaml
import os

from globals.folder_names import LOG_FOLDER, IMAGE_FOLDER
from typing import Dict

def create_experiment_log_dir(config: Dict) -> str:
	# create folder structure 
	experiment_name = config['path']['experiment_name']
	time_stamp = time.time()
	experiment_dir_name = f'{str(time_stamp)}-{experiment_name}'
	root_experiment_folder = os.path.join(os.path.abspath(os.curdir), LOG_FOLDER, experiment_dir_name)
	dir_name_log = os.path.join(root_experiment_folder, LOG_FOLDER)
	dir_name_image = os.path.join(root_experiment_folder, IMAGE_FOLDER)
	os.makedirs(dir_name_log)
	os.makedirs(dir_name_image)

	# copy config into folder
	with open(os.path.join(root_experiment_folder, 'config.yaml'), 'w') as outfile:
		yaml.safe_dump(config, outfile, default_flow_style=False)

	return root_experiment_folder
