import os
import shutil
import json
import cv2
import numpy as np
import importlib
from random import shuffle
from collections import OrderedDict

import util
import tless_inout as inout
from renderer import renderer
import tifffile as tiff
import matplotlib.pyplot as plt

def generate_data(images_path, image_extension, object_models_path, object_model_name, ground_truth_path, 
                  cam_info_path, segmentation_color, output_path):

    print("Generating training data.")

    segmentations_output_path = os.path.join(output_path, 'segmentations')

    if os.path.exists(segmentations_output_path):
        shutil.rmtree(segmentations_output_path)
        
    os.makedirs(segmentations_output_path)

    # To process only images that actually exist
    existing_images = util.get_files_at_path_of_extensions(images_path, [image_extension])

    plt.ioff() # Turn interactive plotting off

    with open(ground_truth_path, 'r') as gt_data_file, \
         open(cam_info_path, 'r') as cam_info_file:
        cam_info = json.load(cam_info_file)

        gt_data = OrderedDict(sorted(json.load(gt_data_file).items(), key=lambda t: t[0]))
        for image_filename in gt_data:
            if not image_filename in existing_images:
              continue

            print("Processing file {}".format(image_filename))
            image_filename_without_extension = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_path, image_filename)
            image = cv2.imread(image_path)
            image_cam_info = cam_info[image_filename]
            # Same goes for camera matrix
            K = np.array(image_cam_info['K']).reshape(3, 3)

            gts_for_image = gt_data[image_filename]
            if len([gt for gt in gts_for_image if gt['obj'] == object_model_name]) > 1:
              print("Warning: found multiple ground truth entries for the object."
                    " Using only the first one.")

            # We have one object model that we actually want to render the segmentation for
            # the other ones may overlay the object and have to be rendered as well
            desired_obj_model = None
            misc_obj_models = []

            for gt_entry in range(len(gts_for_image)):
                gt = gts_for_image[gt_entry]
                # Rotation matrix was flattend to store it in a json
                R = np.array(gt['R']).reshape(3, 3)
                t = np.array(gt['t'])
                object_model = inout.load_ply(os.path.join(object_models_path, gt['obj']))
                object_model['R'] = R
                object_model['t'] = t
                if object_model_name == gt['obj'] and desired_obj_model is None:
                  # We found the first entry for our object model, i.e. this is the one
                  # we want to render a segmentation mask for
                  desired_obj_model = object_model
                else:
                  misc_obj_models.append(object_model)


            segmentation_rendering_path = image_filename_without_extension + "_segmentation.png"
            segmentation_rendering_path = os.path.join(segmentations_output_path, 
                                                    segmentation_rendering_path)

            if desired_obj_model is None:
              # No entry in the scene for our object model, i.e. we write out
              # and empty segmentation image because the network checks for
              # the color and skips the image if the segmentation color is
              # not present
              cv2.imwrite(segmentation_rendering_path, np.zeros(image.shape, np.uint8))
              continue

            main_surface_color = [[255, 255, 255]]
            misc_surface_colors = np.repeat([[0, 0, 0]], len(misc_obj_models), axis=0)
            surface_colors = np.concatenate([main_surface_color, misc_surface_colors])
            # Render the object coordinates ground truth and store it as tiff image
            rendering = renderer.render(image.shape[:2], 
                                         K,
                                         [desired_obj_model] + misc_obj_models, 
                                         surface_colors,
                                         modes=['segmentation'])

            cv2.imwrite(segmentation_rendering_path, rendering['segmentation'])

if __name__ == '__main__':
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to'
                                                  ' create segmentation masks using ground'
                                                  ' truth poses.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the generation config.")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        generate_data(config["IMAGES_PATH"], 
                  config["IMAGE_EXTENSION"], 
                  config["OBJECT_MODELS_PATH"],
                  config["OBJECT_MODEL"],
                  config["GROUND_TRUTH_PATH"],
                  config["CAM_INFO_PATH"],
                  config["SEGMENTATION_COLOR"],
                  config["OUTPUT_PATH"]
                  )