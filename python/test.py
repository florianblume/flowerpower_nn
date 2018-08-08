import training
from util import util
from model import training_config
import shutil
import os
import json
import numpy as np
import glob

def incremental_training(epochs, model, exp_no):
    for i in range(2, 10):
        # Start experiment
        try:
            print("Starting training with model " + model + " experiment " + exp_no + "-" + str(i))
            experiment_path = "./data/experiments/model" + model + "/obj_01/experiment_" + exp_no + "_" + str(i) + "/training"
            config_path = os.path.join(experiment_path, "config.json")
            config = training_config.TrainingConfig()
            config.parse_config_from_json_file(config_path)
            config.EPOCHS = [int(np.sum(epochs[:i]))]
            training.train(experiment_path, config)

            print("Writing trained weights to next experiment.")
            # Set weights on next experiment
            current_training_results_folder = [folder for folder in os.listdir(experiment_path) if folder.startswith("obj_01_")][0]
            current_weights_folder_path = os.path.join(experiment_path, current_training_results_folder)
            next_experiment_path = "./data/experiments/model" + model + "/obj_01/experiment_" + exp_no + "_" + str(i + 1) + "/training"
            next_weights_folder_path = os.path.join(next_experiment_path, current_training_results_folder)
            shutil.copytree(current_weights_folder_path, next_weights_folder_path)

            weight_entries = os.listdir(next_weights_folder_path)
            util.sort_list_by_num_in_string_entries(weight_entries)
            weights_file = weight_entries[-1]
            next_starting_weights_file_path = os.path.join(next_weights_folder_path, weights_file)

            next_config_path = os.path.join(next_experiment_path, "config.json")

            with open(next_config_path, "r") as config_file:
                config = json.load(config_file)
            
            config["WEIGHTS_PATH"] = next_starting_weights_file_path
            config["EPOCHS"] = [int(np.sum(epochs[:i + 1]))]

            with open(next_config_path, "w") as config_file:
                json.dump(config, config_file)
        except Exception as e:
            print(e)
            log_data.append("Error model " + model + " experiment " + exp_no + "-" + str(i))
            with open('log.json', 'w') as log_file:
                json.dump(log_data, log_file)

log_data = []

"""
configs = [name for name in glob.glob('./data/experiments/**/config.json', recursive=True) if not any(x in name for x in ['val', 'inference', 'second_side', 'l2', 'sgd'])]
configs = configs[9:len(configs) - 1]

for _config in configs:
    config = training_config.TrainingConfig()
    config.parse_config_from_json_file(_config)
    config.WEIGHTS_PATH = ''
    epochs = config.EPOCHS
    config.EPOCHS = [max(epochs)]
    config.LAYERS_TO_TRAIN = ['all']
    config.LEARNING_RATE = [0.001]
    try:
        print("Starting training with config {}".format(_config))
        training.train(os.path.dirname(_config), config)
    except Exception as e:
        print(e)
        log_data.append("Error {}".format(_config))
"""

long_epochs = [15, 15, 14, 13, 12, 11, 10, 9, 8, 7]
short_epochs = [5, 5, 4, 4, 4, 3, 3, 3, 2, 2]

#incremental_training(long_epochs, '1', '7')
#incremental_training(short_epochs, '1', '8')
#incremental_training(long_epochs, '5', '7')
incremental_training(short_epochs, '5', '8')