import os
import shutil
import cv2
import json
import numpy as np

import util.util as util
import util.tless_inout as inout
import renderer.renderer as renderer
import tifffile as tiff
import matplotlib.pyplot as plt

from model import dataset
#from model import model
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


def main(images_path, image_extension, object_model_path, ground_truth_path, cam_info_path, temp_data_path, regenerate_data):

    assert os.path.exists(images_path), "The specified images path does not exist."
    assert os.path.exists(object_model_path), "The specified object model file does not exist."
    assert os.path.exists(ground_truth_path), "The specified ground-truth file does not exist."
    assert os.path.exists(cam_info_path), "The specified camera info file does not exist."

    if regenerate_data:
        if os.path.exists(temp_data_path):
            shutil.rmtree(temp_data_path)
            
        os.makedirs(temp_data_path)

        plt.ioff() # Turn interactive plotting off
        generate_data(images_path, image_extension, object_model_path, ground_truth_path, 
                  cam_info_path, temp_data_path, regenerate_data)

    images = util.get_files_at_path_of_extensions(images_path, [image_extension])
    util.sort_list_by_num_in_string_entries(images)
    obj_coordinate_renderings = util.get_files_at_path_of_extensions(temp_data_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(obj_coordinate_renderings)
    segmentation_renderings = util.get_files_at_path_of_extensions(temp_data_path, ['png'])
    util.sort_list_by_num_in_string_entries(segmentation_renderings)

    dataset = model.dataset.Dataset()

    for i in range(len(images)):
        dataset.add_image(images[i])
        dataset.add_segmentation_image(segmentation_renderings[i])
        dataset.add_obj_coord_image(obj_coordinate_renderings[i])

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to train the FlowerPower network.')
    parser.add_argument("--images_path",
                        required=True,
                        help="The path to the images.")
    parser.add_argument("--image_extension",
                        required=True,
                        help="The extension of the images to use in training.")
    parser.add_argument("--object_model_path",
                        required=True,
                        help="The path to the 3D object to train for.")
    parser.add_argument("--cam_info_path",
                        required=True,
                        help="The path to the camera info file.")
    parser.add_argument("--gt_path",
                        required=True,
                        help="The path to the ground-truth annotations file.")
    parser.add_argument("--data_path",
                        required=True,
                        help="The path where the taining process can create the necessary renderings at \
                        or will try to load existing renderings from.")
    """
    # TODO: Add argument for list of image filenames to process
    # TODO: Add argument for path to pre-trained weights
    parser.add_argument("--log_path",
                        required=True,
                        help="The path to the folder where the logs and the resulting weights are to be stored.")
    """
    parser.add_argument("--regenerate_data",
                        required=False,
                        action='store_true',
                        help="If set, the already produced depth map renderings and cropped images will be removed \
                        and regenerated.")

    # More arguments related to training, etc. are to follow
    arguments = parser.parse_args()
    main(arguments.images_path, arguments.image_extension, arguments.object_model_path,
         arguments.gt_path, arguments.cam_info_path, arguments.data_path, True)