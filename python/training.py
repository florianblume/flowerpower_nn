import os
import shutil
import cv2
import json
import numpy as np
from random import shuffle

import util.util as util
import util.tless_inout as inout
import renderer.renderer as renderer
import tifffile as tiff
import matplotlib.pyplot as plt

from model import dataset
from model import model
from model import config

OBJ_COORD_FILE_EXTENSION = "_obj_coordinates.tiff"
SEG_FILE_EXTENSION = "_segmentation.png"


def generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, temp_data_path, regenerate_data):

    with open(ground_truth_path, 'r') as gt_data_file, open(cam_info_path, 'r') as cam_info_file:
        gt_data = json.load(gt_data_file)
        cam_info = json.load(cam_info_file)
        for image_filename in gt_data:
            image_filename_without_extension = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_path, image_filename)

            gts_for_image = gt_data[image_filename]

            for gt_entry in range(len(gts_for_image)):
                gt = gts_for_image[gt_entry]
                # Filter out ground-truth entries that are interesting to us
                object_model_name = os.path.basename(object_model_path)
                if object_model_name == gt['obj']:
                    ################## TODO: add support for .obj files ###################

                    object_model = inout.load_ply(object_model_path)
                    # Rotation matrix was flattend to store it in a json
                    R = np.array(gt['R']).reshape(3, 3)
                    t = np.array(gt['t'])
                    image_cam_info = cam_info[image_filename]
                    # Same goes for camera matrix
                    K = np.array(image_cam_info['K']).reshape(3, 3)
                    image = cv2.imread(image_path)

                    # Render the object coordinates ground truth and store it as tiff image
                    renderings = renderer.render(object_model, (image.shape[0], image.shape[1]), 
                                                                   K, R, t, 
                                                                   mode=[renderer.RENDERING_MODE_OBJ_COORDS, 
                                                                   renderer.RENDERING_MODE_SEGMENTATION])

                    object_coordinates_rendering = renderings[renderer.RENDERING_MODE_OBJ_COORDS].astype(np.float16)
                    object_coordinates_rendering_path = image_filename_without_extension + OBJ_COORD_FILE_EXTENSION
                    object_coordinates_rendering_path = os.path.join(temp_data_path, object_coordinates_rendering_path)
                    tiff.imsave(object_coordinates_rendering_path, object_coordinates_rendering)

                    segmentation_rendering = renderings[renderer.RENDERING_MODE_SEGMENTATION]
                    segmentation_rendering_path = image_filename_without_extension + SEG_FILE_EXTENSION
                    segmentation_rendering_path = os.path.join(temp_data_path, segmentation_rendering_path)
                    cv2.imwrite(segmentation_rendering_path, segmentation_rendering)


def train(config):

    images_path = config.IMAGES_PATH
    image_extension = config.IMAGE_EXTENSION 
    object_model_path = config.OBJECT_MODEL_PATH 
    ground_truth_path = config.GT_PATH 
    cam_info_path = config.CAM_INFO_PATH 
    temp_data_path = config.DATA_PATH 
    regenerate_data = config.REGENERATE_DATA 
    weights_path = config.WEIGHTS_PATH 
    output_path = config.OUTPUT_PATH

    assert os.path.exists(images_path), "The specified images path does not exist."
    assert os.path.exists(object_model_path), "The specified object model file does not exist."
    assert os.path.exists(ground_truth_path), "The specified ground-truth file does not exist."
    assert os.path.exists(cam_info_path), "The specified camera info file does not exist."

    if weights_path != "":
        assert os.path.exists(weights_path)

    if regenerate_data:
        if os.path.exists(temp_data_path):
            shutil.rmtree(temp_data_path)
            
        os.makedirs(temp_data_path)

        plt.ioff() # Turn interactive plotting off
        generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, temp_data_path, regenerate_data)

    # Retrieve the rendered images to add it to the datasets
    images = util.get_files_at_path_of_extensions(images_path, [image_extension])
    util.sort_list_by_num_in_string_entries(images)
    obj_coordinate_renderings = util.get_files_at_path_of_extensions(temp_data_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(obj_coordinate_renderings)
    segmentation_renderings = util.get_files_at_path_of_extensions(temp_data_path, ['png'])
    util.sort_list_by_num_in_string_entries(segmentation_renderings)

    train_dataset = model.dataset.Dataset()
    val_dataset = model.dataset.Dataset()

    indices = range(len(images))
    shuffle(indices)

    for i in indices[:len(indices) * config.TRAIN_VAL_RATIO]:
        train_dataset.add_image(images[i])
        train_dataset.add_segmentation_image(segmentation_renderings[i])
        train_dataset.add_obj_coord_image(obj_coordinate_renderings[i])

    for i in indices[len(indices) * config.TRAIN_VAL_RATIO : len(indices) - 1]:
        val_dataset.add_image(images[i])
        val_dataset.add_segmentation_image(segmentation_renderings[i])
        val_dataset.add_obj_coord_image(obj_coordinate_renderings[i])

    network_model = model.FlowerPowerCNN('training', config, output_path)
    if weights_path != "":
        network_model.load_weights(weights_path, config.LAYERS_TO_EXCLUDE_FROM_WEIGHT_LOADING)

    network_model.train(train_dataset, val_dataset, config.LEARNING_RATE, config.EPOCHS, config.LAYERS_TO_TRAIN)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to train the FlowerPower network.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    train_config = config.parse_config_from_json_file(arguments.config)
    train(train_config)