import os
import shutil
import json
import cv2
import math
import importlib
import numpy as np
from random import randint

import util.util as util

from model import dataset
# We load the model later dynamically based on what is requested in the config
#from model import model
from model import inference_config

def inference(base_path, config):

    images_path = config.IMAGES_PATH
    image_extension = config.IMAGE_EXTENSION 
    segmentation_images_path = config.SEGMENTATION_IMAGES_PATH
    segmentation_image_extension = config.SEGMENTATION_IMAGE_EXTENSION
    segmentation_color = config.SEGMENTATION_COLOR
    object_model_path = config.OBJECT_MODEL_PATH 
    weights_path = config.WEIGHTS_PATH 
    batch_size = config.BATCH_SIZE
    image_list = os.path.join(base_path, config.IMAGE_LIST)
    output_path = os.path.join(base_path, config.OUTPUT_PATH)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    assert os.path.exists(weights_path), \
            "The weights file {} does not exist.".format(weights_path)

    image_paths = util.get_files_at_path_of_extensions(images_path, image_extension)
    util.sort_list_by_num_in_string_entries(image_paths)
    segmentation_image_paths = util.get_files_at_path_of_extensions(segmentation_images_path, segmentation_image_extension)
    util.sort_list_by_num_in_string_entries(segmentation_image_paths)

    images = []
    segmentation_images = []
    cropped_segmentation_images = []
    # Bounding boxes
    bbs = []

    print("Preparing data.")

        # TODO: Support file name list

    # Prepare data, i.e. crop images to the segmentation mask
    with open(image_list, "r") as loaded_image_list:
        images_to_process = json.load(loaded_image_list)
        image_paths = [image_path for image_path in image_paths if image_path in images_to_process]

    for index in range(len(image_paths)):
        image_path = image_paths[index]
        image = cv2.imread(os.path.join(images_path, image_path))
        segmentation_image_path = segmentation_image_paths[index]
        segmentation_image =cv2.imread(os.path.join(segmentation_images_path, segmentation_image_path))
        image, frame = util.crop_image_on_segmentation_color(
                        image, segmentation_image, segmentation_color, return_frame=True)
        bbs.append(frame)
        cropped_segmentation_image = util.crop_image_on_segmentation_color(
                                segmentation_image, segmentation_image, segmentation_color)
        images.append(image)
        segmentation_images.append(segmentation_image)
        cropped_segmentation_images.append(cropped_segmentation_image)

    # Otherwise datatype is int64 which is not JSON serializable
    bbs = np.array(bbs).astype(np.int32)

    print("Running network inference.")
    # Here we import the request model
    model = importlib.import_module("model." + config.MODEL + ".model")
    network_model = model.FlowerPowerCNN('inference', config, output_path)
    network_model.load_weights(weights_path, by_name=True)

    results = []
    # We only store the filename + extension
    object_model_path = os.path.basename(object_model_path)

    current_batch = 0
    batch_size = min(batch_size, len(images))
    # TODO: Check if it is necessary for the network to know the batch size...
    #       It seems that batch size is just a sanity check
    # Set batch size for the network to the current batch size
    network_model.config.BATCH_SIZE = batch_size

    while current_batch < len(images):
        # Account for that the number of images might not be divisible by the batch size
        batch_size = min(batch_size, len(images) - current_batch)
        batch_start = current_batch
        current_batch += batch_size
        batch_end = current_batch
        network_model.config.BATCH_SIZE = batch_size
        predictions = network_model.predict(images[batch_start:batch_end], cropped_segmentation_images[batch_start:batch_end], verbose=1)
        for index in range(len(predictions)):
            results.append({    "prediction" : predictions[index], 
                                "image" : image_paths[batch_start + index], 
                                "segmentation_image" : segmentation_image_paths[batch_start + index],
                                "object_model" : object_model_path,
                                "bb" : bbs[batch_start + index]
                            })

    return results