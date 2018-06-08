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
import renderer.renderer as renderer
import tifffile as tiff
import matplotlib.pyplot as plt

def generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, segmentation_color, output_path):

    print("Generating training data.")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    os.makedirs(os.path.join(output_path, "images"))
    os.makedirs(os.path.join(output_path, "segmentations"))
    os.makedirs(os.path.join(output_path, "obj_coords"))

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
        obj_coords_output_path = os.path.join(output_path, "obj_coords")

        gt_data = OrderedDict(sorted(json.load(gt_data_file).items(), key=lambda t: t[0]))
        cam_info = json.load(cam_info_file)
        new_cam_info = {}
        for image_filename in gt_data:
            if not image_filename in existing_images:
              continue

            print("Processing file {}".format(image_filename))
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
                                                                   renderer.RENDERING_MODE_SEGMENTATION],
                                                                   segmentation_color=segmentation_color)

                    # Render the segmentation image first, to crop all images to the segmentation mask
                    segmentation_rendering = renderings[renderer.RENDERING_MODE_SEGMENTATION]
                    # We need to write the crops into the new camera info file because the principal points 
                    # changes when we crop the image
                    cropped_segmentation, crop_frame = util.crop_image_on_segmentation_color(
                                                                  segmentation_rendering, 
                                                                  segmentation_rendering,
                                                                  segmentation_color, return_frame=True)
                    segmentation_rendering_path = "segmentation_" + image_filename_without_extension + ".png"
                    segmentation_rendering_path = os.path.join(segmentations_output_path, 
                                                            segmentation_rendering_path)
                    cv2.imwrite(segmentation_rendering_path, cropped_segmentation)

                    # Render, crop and save object coordinates
                    object_coordinates_rendering = renderings[renderer.RENDERING_MODE_OBJ_COORDS]\
                                                              .astype(np.float16)

                    object_coordinates = util.crop_image_on_segmentation_color(
                                                                    object_coordinates_rendering, 
                                                                    segmentation_rendering,
                                                                    segmentation_color)
                    object_coordinates_rendering_path = "obj_coords_" + image_filename_without_extension + ".tiff"
                    object_coordinates_rendering_path = os.path.join(obj_coords_output_path, 
                                                            object_coordinates_rendering_path)
                    tiff.imsave(object_coordinates_rendering_path, object_coordinates)

                    # Save the original image in a cropped version as well
                    cropped_image = util.crop_image_on_segmentation_color(image, 
                                                                         segmentation_rendering,
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
    parser = argparse.ArgumentParser(description='This script provides functionality to train the FlowerPower network.')
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