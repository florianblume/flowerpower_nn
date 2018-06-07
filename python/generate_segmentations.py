import os
import shutil
import json
import cv2
import numpy as np
import importlib
from random import shuffle
from collections import OrderedDict

import util.util as util
import util.tless_inout as inout
import renderer.renderer_segmentations_with_misc_objs as renderer
import tifffile as tiff
import matplotlib.pyplot as plt

def generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, segmentation_color, output_path):

    print("Generating training data.")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    os.makedirs(os.path.join(output_path, "images"))
    os.makedirs(os.path.join(output_path, "segmentations"))

    # To process only images that actually exist
    existing_images = util.get_files_at_path_of_extensions(images_path, [image_extension])

    plt.ioff() # Turn interactive plotting off

    cam_info_output_path = os.path.join(output_path, "info.json")

    with open(ground_truth_path, 'r') as gt_data_file, \
         open(cam_info_path, 'r') as cam_info_file, \
         open(cam_info_output_path, 'w') as cam_info_output_file:

        # The paths where to store the results
        images_output_path = os.path.join(output_path, "images")
        segmentations_output_path = os.path.join(output_path, "segmentations")

        gt_data = OrderedDict(sorted(json.load(gt_data_file).items(), key=lambda t: t[0]))
        cam_info = json.load(cam_info_file)
        new_cam_info = {}
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
            object_model_name = os.path.basename(object_model_path)
            if len([gt for gt in gts_for_image if gt['obj'] == object_model_name]) > 1:
              print("Warning: found multiple ground truth entries for the object."
                    " Using only the first one.")

            # We have one object model that we actually want to render the segmentation for
            # the other ones may overlay the object and have to be rendered as well
            desired_obj_model = None
            misc_obj_models = []

            for gt_entry in range(len(gts_for_image)):
                gt = gts_for_image[gt_entry]
                object_model = inout.load_ply(object_model_path)
                # Rotation matrix was flattend to store it in a json
                R = np.array(gt['R']).reshape(3, 3)
                t = np.array(gt['t'])
                obj_dict = {"obj" : object_model,
                            "R" : R,
                            "t" : t}
                if object_model_name == gt['obj'] and desired_obj_model is None:
                  # We found the first entry for our object model, i.e. this is the one
                  # we want to render a segmentation mask for
                  desired_obj_model = obj_dict
                else:
                  misc_obj_models.append(obj_dict)


            segmentation_rendering_path = "segmentation_" + image_filename_without_extension + ".png"
            segmentation_rendering_path = os.path.join(segmentations_output_path, 
                                                    segmentation_rendering_path)

            if desired_obj_model is None:
              # No entry in the scene for our object model, i.e. we write out
              # and empty segmentation image because the network checks for
              # the color and skips the image if the segmentation color is
              # not present
              cv2.imwrite(segmentation_rendering_path, np.zeros(image.shape, np.uint8))
              continue

            # Render the object coordinates ground truth and store it as tiff image
            segmentation = renderer.render(desired_obj_model, 
                                         misc_obj_models, 
                                         (image.shape[0], image.shape[1]), 
                                         K,
                                         segmentation_color=segmentation_color)

            # We need to write the crops into the new camera info file because the principal points 
            # changes when we crop the image
            cropped_segmentation, crop_frame = util.crop_image_on_segmentation_color(
                                                          segmentation, 
                                                          segmentation,
                                                          segmentation_color, return_frame=True)
            cv2.imwrite(segmentation_rendering_path, cropped_segmentation)

            # Save the original image in a cropped version as well
            cropped_image = util.crop_image_on_segmentation_color(image, 
                                                                 segmentation,
                                                                 segmentation_color)
            cropped_image_path = os.path.join(images_output_path, image_filename)
            cv2.imwrite(cropped_image_path, cropped_image)

            # Update camera matrix
            # I.e. the principal point has to be adjusted by shifting it by the crop offset
            K[0][2] = K[0][2] - crop_frame[1]
            K[1][2] = K[1][2] - crop_frame[0]
            image_cam_info['K'] = K.flatten().tolist()
            new_cam_info[image_filename] = image_cam_info
        json.dump(OrderedDict(sorted(new_cam_info.items(), key=lambda t: t[0])), cam_info_output_file)

if __name__ == '__main__':
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to'
                                                  ' create segmentation masks using ground'
                                                  ' truth for images that the network is'
                                                  ' to be run on in inference mode.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the generation config.")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        generate_data(config["IMAGES_PATH"], 
                  config["IMAGE_EXTENSION"], 
                  config["OBJECT_MODEL_PATH"],
                  config["GROUND_TRUTH_PATH"],
                  config["CAM_INFO_PATH"],
                  config["SEGMENTATION_COLOR"],
                  config["OUTPUT_PATH"]
                  )