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
                  inferred_poses_path, cam_info_path, segmentation_color, output_path):

    print("Generating training data.")

    bounding_box_output_path = os.path.join(output_path, 'bounding_boxes')

    if os.path.exists(bounding_box_output_path):
        shutil.rmtree(bounding_box_output_path)
        
    os.makedirs(bounding_box_output_path)

    # To process only images that actually exist
    existing_images = util.get_files_at_path_of_extensions(images_path, [image_extension])

    plt.ioff() # Turn interactive plotting off

    with open(ground_truth_path, 'r') as gt_data_file, \
        open(inferred_poses_path, 'r') as inferred_data_file, \
         open(cam_info_path, 'r') as cam_info_file:
        cam_info = json.load(cam_info_file)

        gt_data = OrderedDict(sorted(json.load(gt_data_file).items(), key=lambda t: t[0]))
        inferred_data = OrderedDict(sorted(json.load(inferred_data_file).items(), key=lambda t: t[0]))
        for image_filename in gt_data:
            if image_filename not in inferred_data or image_filename not in existing_images:
              continue

            print("Processing file {}".format(image_filename))
            image_filename_without_extension = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_path, image_filename)
            image = cv2.imread(image_path)
            image_cam_info = cam_info[image_filename]
            # Same goes for camera matrix
            K = np.array(image_cam_info['K']).reshape(3, 3)

            gts_for_image = gt_data[image_filename]
            inferred_poses_for_image = inferred_data[image_filename]

            object_models = []

            for gt_entry in gts_for_image:

                # We only want to render ground-truth poses when there is a corresponding inferred pose
                object_found = False
                for pose in inferred_poses_for_image:
                    if pose['obj'] == gt_entry['obj']:
                        object_found = True
                if not object_found:
                    continue

                # Rotation matrix was flattend to store it in a json
                R = np.array(gt_entry['R']).reshape(3, 3)
                t = np.array(gt_entry['t'])
                object_model = inout.load_ply(os.path.join(object_models_path, gt_entry['obj']))
                object_model['R'] = R
                object_model['t'] = t
                object_models.append(object_model)

            gt_surface_colors = np.repeat([[0, 255, 0, 125]], len(object_models), axis=0)

            for pose in inferred_poses_for_image:
                # Rotation matrix was flattend to store it in a json
                R = np.array(pose['R']).reshape(3, 3)
                t = np.array(pose['t'])
                object_model = inout.load_ply(os.path.join(object_models_path, pose['obj']))
                object_model['R'] = R
                object_model['t'] = t
                object_models.append(object_model)

            inferred_surface_colors = np.repeat([[0, 0, 255, 125]], len(object_models), axis=0)
            surface_colors = np.concatenate([gt_surface_colors, inferred_surface_colors])

            # Render the object coordinates ground truth and store it as tiff image
            rendering = renderer.render((image.shape[0], image.shape[1]), 
                                         K,
                                         object_models, 
                                         surface_colors,
                                         modes=['bounding_boxes'])

            bounding_box_rendering_path = image_filename_without_extension + "_bounding_boxes.png"
            bounding_box_rendering_path = os.path.join(bounding_box_output_path, 
                                                    bounding_box_rendering_path)
            rendering = rendering['bounding_boxes']
            pose_incides = np.where((rendering > [0, 0, 0]).any(axis = 2))
            image[pose_incides] = 0.3 * image[pose_incides] + 0.7 * rendering[pose_incides]
            cv2.imwrite(bounding_box_rendering_path, image)

if __name__ == '__main__':
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to'
                                                  ' render bounding boxes of poses to visualize the result.')
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
                  config["INFERRED_POSES_PATH"],
                  config["CAM_INFO_PATH"],
                  config["SEGMENTATION_COLOR"],
                  config["OUTPUT_PATH"]
                  )