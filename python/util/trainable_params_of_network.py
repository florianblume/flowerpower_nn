import os
import shutil
import json
import cv2
import numpy as np
import math
from collections import OrderedDict
import tifffile as tiff

import inference as inference_script
import util
from ..model import inference_config
from ..model import model_util

def inference(config):
    weights_path = config.WEIGHTS_PATH 
    output_path = ""
    model = model_util.get_model(config.MODEL)
    network_model = model.FlowerPowerCNN('inference', config, output_path)
    network_model.load_weights(weights_path, by_name=True)
    print("Architecture {} has {} trainable parameters.".format(config.MODEL, network_model.get_number_of_trainable_variables()))

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to run the FlowerPower network.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    config = inference_config.InferenceConfig()
    config.parse_config_from_json_file(arguments.config)
    inference(config)