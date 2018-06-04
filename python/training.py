import os
import shutil
import json
import cv2
import numpy as np
import importlib
from random import shuffle

import util.util as util
import util.tless_inout as inout
import renderer.renderer as renderer
import tifffile as tiff
import matplotlib.pyplot as plt

from model import dataset
# Model will be imported based on the one requested in the config
#from model import model
from model import training_config

def train(config):

    # We do not retrieve the color from the config (it should not be specified anyway)
    # because we render our own segmentation images using white color

    image_extension = config.IMAGE_EXTENSION 
    object_model_path = config.OBJECT_MODEL_PATH 
    ground_truth_path = config.GT_PATH 
    data_path = config.DATA_PATH 
    weights_path = config.WEIGHTS_PATH 
    output_path = config.OUTPUT_PATH

    assert os.path.exists(object_model_path), "The object model file {} does not exist.".format(object_model_path)
    assert os.path.exists(ground_truth_path), "The ground-truth file {} does not exist.".format(ground_truth_path)

    if weights_path != "":
        assert os.path.exists(weights_path), "The weights file {} does not exist.".format(weights_path)

    # The paths where we store the generated object coordinate images as well as
    # the cropped original and segmentation images
    images_path = os.path.join(data_path, "images")
    segmentations_path = os.path.join(data_path, "segmentations")
    obj_coords_path = os.path.join(data_path, "obj_coords")

    # Retrieve the rendered images to add it to the datasets
    images = util.get_files_at_path_of_extensions(images_path, [image_extension])
    util.sort_list_by_num_in_string_entries(images)
    segmentation_renderings = util.get_files_at_path_of_extensions(segmentations_path, ['png'])
    util.sort_list_by_num_in_string_entries(segmentation_renderings)
    obj_coordinate_renderings = util.get_files_at_path_of_extensions(obj_coords_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(obj_coordinate_renderings)

    print("Populating datasets.")

    train_dataset = dataset.Dataset()
    val_dataset = dataset.Dataset()

    # Open the json files that hold the filenames for the respective datasets
    with open(config.TRAIN_FILE, 'r') as train_file, open(config.VAL_FILE, 'r') as val_file:
        train_filenames = json.load(train_file)
        val_filenames = json.load(val_file)
        # Fill training dict
        for i, image in enumerate(images):
            segmentation_image = segmentation_renderings[i]
            loaded_segmentation_image = cv2.imread(os.path.join(segmentations_path, segmentation_image))
            # We do not want to scale object coordinates in the network because that creates
            # imprecisions. I.e. we can only pad object coordinates to fill the image size
            # but not resize them. This way, the object coordinates would not fit in the
            # batch arrays when width or height exceed the dimensions specified in the config.
            if loaded_segmentation_image.shape[0] > config.IMAGE_DIM or \
               loaded_segmentation_image.shape[1] > config.IMAGE_DIM:
                raise Exception("Image dimension exceeds image dim specified in config. \
                    File: {}".format(image))

            object_coordinate_image = obj_coordinate_renderings[i]
            # Check both cases, it might be that the image is not to be added at all
            if image in train_filenames:
                train_dataset.add_training_example(
                        os.path.join(images_path, image),
                        os.path.join(segmentations_path, segmentation_image),
                        os.path.join(obj_coords_path, object_coordinate_image))
            elif image in val_filenames:
                val_dataset.add_training_example(
                        os.path.join(images_path, image),
                        os.path.join(segmentations_path, segmentation_image),
                        os.path.join(obj_coords_path, object_coordinate_image))

    print("Added {} images for training and {} images for validation.".format(train_dataset.size(), val_dataset.size()))

    # Here we import the request model
    model = importlib.import_module("model." + config.MODEL + ".model")
    network_model = model.FlowerPowerCNN('training', config, output_path)
    if weights_path != "":
        network_model.load_weights(weights_path, by_name=True, exclude=config.LAYERS_TO_EXCLUDE_FROM_WEIGHT_LOADING)

    print("Starting training.")
    network_model.train(train_dataset, val_dataset, config)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to train the FlowerPower network.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    config = training_config.TrainingConfig()
    config.parse_config_from_json_file(arguments.config)
    train(config)