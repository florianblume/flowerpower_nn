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

OBJ_COORD_FILE_EXTENSION = "_obj_coordinates.tiff"
SEG_FILE_EXTENSION = "_segmentation.png"


def generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, temp_data_path, regenerate_data):

    with open(ground_truth_path, 'r') as gt_data_file, open(cam_info_path, 'r') as cam_info_file:

        # The paths where to store the results
        temp_path_images = os.path.join(temp_data_path, "images")
        temp_path_segmentation_images = os.path.join(temp_data_path, "segmentation_images")
        temp_path_obj_coord_images = os.path.join(temp_data_path, "obj_coord_images")

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

                    # Render the segmentation image first, to crop all images to the segmentation mask
                    segmentation_rendering = renderings[renderer.RENDERING_MODE_SEGMENTATION]
                    # On the borders of the object the segmentation color is not 255 but above 0
                    segmentation_rendering_indices = segmentation_rendering > 0
                    segmentation_rendering[segmentation_rendering_indices] = 255
                    cropped_segmentation = uti.crop_image_on_segmentation_color(segmentation_rendering, 
                                                                              segmentation_rendering,
                                                                              [255, 255, 255])
                    segmentation_rendering_path = image_filename_without_extension + SEG_FILE_EXTENSION
                    segmentation_rendering_path = os.path.join(temp_path_segmentation_images, segmentation_rendering_path)
                    cv2.imwrite(segmentation_rendering_path, cropped_segmentation)

                    # Render, crop and save object coordinates
                    object_coordinates_rendering = renderings[renderer.RENDERING_MODE_OBJ_COORDS].astype(np.float16)
                    # TODO: use desired color from config
                    object_coordinates = uti.crop_image_on_segmentation_color(object_coordinates_rendering, 
                                                                              segmentation_rendering,
                                                                              [255, 255, 255])
                    object_coordinates_rendering_path = image_filename_without_extension + OBJ_COORD_FILE_EXTENSION
                    object_coordinates_rendering_path = os.path.join(temp_path_obj_coord_images, object_coordinates_rendering_path)
                    tiff.imsave(object_coordinates_rendering_path, object_coordinates)

                    # Save the original image in a cropped version as well
                    cropped_image = uti.crop_image_on_segmentation_color(image, 
                                                                         segmentation_rendering,
                                                                         [255, 255, 255])
                    cropped_image_path = os.path.join(temp_path_images, image_filename)
                    cv2.imwrite(segmentation_rendering_path, cropped_image)


def train(config):

    # We do not retrieve the color from the config (it should not be specified anyway)
    # because we render our own segmentation images using white color

    image_extension = config.IMAGE_EXTENSION 
    object_model_path = config.OBJECT_MODEL_PATH 
    ground_truth_path = config.GT_PATH 
    cam_info_path = config.CAM_INFO_PATH 
    temp_data_path = config.DATA_PATH 
    regenerate_data = config.REGENERATE_DATA 
    weights_path = config.WEIGHTS_PATH 
    output_path = config.OUTPUT_PATH

    assert os.path.exists(object_model_path), "The object model file {} does not exist.".format(object_model_path)
    assert os.path.exists(ground_truth_path), "The ground-truth file {} does not exist.".format(ground_truth_path)
    assert os.path.exists(cam_info_path), "The camera info file {} does not exist.".format(cam_info_path)

    if weights_path != "":
        assert os.path.exists(weights_path), "The weights file {} does not exist.".format(weights_path)

    # The paths where we store the generated object coordinate images as well as
    # the cropped original and segmentation images
    temp_path_images = os.path.join(temp_data_path, "images")
    temp_path_segmentation_images = os.path.join(temp_data_path, "segmentation_images")
    temp_path_obj_coord_images = os.path.join(temp_data_path, "obj_coord_images")

    if regenerate_data:

        print("Generating training data.")

        if os.path.exists(temp_data_path):
            shutil.rmtree(temp_data_path)
            
        os.makedirs(temp_path_images)
        os.makedirs(temp_path_segmentation_images)
        os.makedirs(temp_path_obj_coord_images)

        plt.ioff() # Turn interactive plotting off
        generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, temp_data_path, regenerate_data)

    # Retrieve the rendered images to add it to the datasets
    images = util.get_files_at_path_of_extensions(temp_path_images, [image_extension])
    util.sort_list_by_num_in_string_entries(images)
    segmentation_renderings = util.get_files_at_path_of_extensions(temp_path_segmentation_images, ['png'])
    util.sort_list_by_num_in_string_entries(segmentation_renderings)
    obj_coordinate_renderings = util.get_files_at_path_of_extensions(temp_path_obj_coord_images, ['tiff'])
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
            object_coordinate_image = obj_coordinate_renderings[i]
            # Check both cases, it might be that the image is not to be added at all
            if image in train_filenames:
                train_dataset.add_training_example(
                        os.path.join(temp_path_images, image),
                        os.path.join(temp_path_segmentation_images, segmentation_image),
                        os.path.join(temp_path_obj_coord_images, object_coordinate_image))
            elif image in val_filenames:
                val_dataset.add_training_example(
                        os.path.join(temp_path_images, image),
                        os.path.join(temp_path_segmentation_images, segmentation_image),
                        os.path.join(temp_path_obj_coord_images, object_coordinate_image))

    print("Added {} images for training and {} images for validation.".format(train_dataset.size(), val_dataset.size()))

    # Here we import the request model
    model = importlib.import_module("model." + config.MODEL + ".model")
    network_model = model.FlowerPowerCNN('training', config, output_path)
    if weights_path != "":
        network_model.load_weights(weights_path, config.LAYERS_TO_EXCLUDE_FROM_WEIGHT_LOADING)

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