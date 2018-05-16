import os
import shutil
import json
import cv2
import numpy as np
from random import shuffle

import util.util as util

from model import dataset
from model import model
from model import config

def ransac(prediction, cam_info, num_iterations=10):
    obj_coords = prediction['obj_coords']
    step_y = prediction['step_y']
    step_x = prediction['step_x']
    indices = prediction != [np.nan, np.nan, np.nan]
    pairs = []
    for index in indices:
        pairs.append({"2d" : index, "3d" : prediction[index]})

def inference(config):

	images_path = config.IMAGES_PATH
	image_extension = config.IMAGE_EXTENSION 
	segmentation_images_path = config.SEGMENTATION_IMAGES_PATH
	segmentation_image_extension = config.SEGMENTATION_IMAGE_EXTENSION
    segmentation_color = config.SEGMENTATION_COLOR
    object_model_path = config.OBJECT_MODEL_PATH 
    cam_info_path = config.CAM_INFO_PATH 
    weights_path = config.WEIGHTS_PATH 
    batch_size = config.BATCH_SIZE
    ransac_iterations = config.RANSAC_ITERATIONS
    output_file = config.OUTPUT_FILE

    assert os.path.exists(images_path), \
    		"The images path {} does not exist.".format(images_path)
    assert image_extension in ['png', 'jpg', 'jpeg'], \
    		"Unkown image extension."
    assert os.path.exists(segmentation_images_path), \
    		"The segmentation images path {} does not exist.".format(segmentation_images_path)
    assert segmentation_image_extension in ['png', 'jpg', 'jpeg'], \
    		"Unkown segmentation image extension."
    assert os.path.exists(object_model_path), \
    		"The object model file {} does not exist.".format(object_model_path)
    assert os.path.exists(cam_info_path), \
    		"The camera info file {} does not exist.".format(cam_info_path)
    assert os.path.exists(weights_path), \
    		"The weights file {} does not exist.".format(weights_path)

    if not os.path.exists(output_path):
    	os.makedirs(output_path)

    image_paths = util.get_files_at_path_of_extensions(images_path, image_extension)
    image_paths = util.sort_list_by_num_in_string_entries(images)
    segmentation_image_paths = util.get_files_at_path_of_extensions(segmentation_images_path, segmentation_image_extension)
    segmentation_image_paths = util.sort_list_by_num_in_string_entries(segmentation_images)

    images = []
    segmentation_images = []
    cropped_segmentation_images = []
    # Bounding boxes
    bbs = []

    # Prepare data, i.e. crop images to the segmentation mask
    for index in range(len(image_paths)):
        image_path = image_paths[index]
        image = cv2.imread(image_path)
        segmentation_image_path = segmentation_image_paths[index]
        segmentation_image =cv2.imread(segmentation_image_path)
        image, frame = util.crop_image_on_segmentation_color(
                        image, segmentation_image, segmentation_color, return_frame=True)
        bbs.append(frame)
        cropped_segmentation_image = util.crop_image_on_segmentation_color(
                                segmentation_image, segmentation_image, segmentation_color)
        images.append(image)
        segmentation_images.append(segmentation_image)
        cropped_segmentation_images.append(cropped_segmentation_image)

    # No support for batching yet
    config.BATCH_SIZE = 1
    print("Running network inference.")
    network_model = model.FlowerPowerCNN('inference', config, output_path)
    results = []
    # We only store the filename + extension
    object_model_path = os.path.basename(object_model_path)

    with open(cam_info_path, "r") as cam_info:
        for i in range(len(images)):
            key = os.path.basename(images_paths[i])
            prediction = model.predict([images[i]], [cropped_segmentation_images[i]], verbose=1)
            pose = ransac(prediction, cam_info[key], ransac_iterations)
            results.append({key : [{"R" : pose.R.flatten().tolist(), 
                                    "t" : pose.t.flatten().tolist(), 
                                    "bb" : bbs[i].flatten().tolist(), 
                                    "obj" : object_model_path}]})

    print("Writing results to {}".format(output_file))
    with open(output_file, "w") as json_file:
        json.dump(results, json_file)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to run the FlowerPower network.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    train_config = config.parse_config_from_json_file(arguments.config)
    inference(train_config)