import os
import shutil
import json
import cv2
import numpy as np
import math
from collections import OrderedDict
import tifffile as tiff

import inference as inference_script
import util.util as util
from model import inference_config

def compute_reprojection(prediction, rvec, tvec, cam_info):
    prediction = prediction[prediction != [0, 0, 0]]
    prediction = prediction.reshape((int(prediction.shape[0] / 3), 3))
    reprojection, _ = cv2.projectPoints(prediction,
                                        rvec,
                                        tvec,
                                        np.array(cam_info['K']).reshape(3, 3),
                                        None)
    return reprojection

def ransac(prediction, imsize, K):
    obj_coords = prediction['obj_coords']

    step_y = prediction['step_y']
    step_x = prediction['step_x']

    image_points, object_points = util.pair_object_coords_with_index(obj_coords, imsize, step_y, step_x)
    retval, rvec, tvec, inliers  = cv2.solvePnPRansac(object_points, 
                              image_points, 
                              K, 
                              None,
                              iterationsCount=100
    )

    # If z value is negative we need to flip the translation vector, this can happen when the z axis
    # is the viewing axis of the camera
    if tvec[2] < 0:
        tvec = tvec * -1

    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    projected_points = projected_points.reshape(-1, 2)
    error = np.sum(np.abs(image_points - projected_points) ** 2)

    return retval, rvec, tvec, inliers, np.sqrt(error / float(len(image_points)))

def inference_with_config_path(config_string):
    config = inference_config.InferenceConfig()
    config.parse_config_from_json_file(config_string)
    inference(config)

def inference(config):
    base_path = os.path.dirname(arguments.config)
    cam_info_path = config.CAM_INFO_PATH
    object_model_path = config.OBJECT_MODEL_PATH
    merge_mode = config.MERGE_MODE
    output_file = os.path.join(base_path, config.OUTPUT_FILE)

    assert merge_mode in ["overwrite", "append", "replace"], "Unkown merge mode"

    assert os.path.exists(cam_info_path), \
            "The camera info file {} does not exist.".format(cam_info_path)

    results = inference_script.inference(base_path, config)

    if (merge_mode == "append" or merge_mode == "replace") and os.path.exists(output_file):
        with open(output_file, "r") as json_file:
            try:
                converted_results = json.load(json_file)
            except ValueError:
                print("No existing poses loaded because the JSON file could not be read. Overwriting the file.")
                converted_results = OrderedDict()
    else:
        converted_results = OrderedDict()

    with open(cam_info_path, "r") as cam_info_file:
        cam_info = json.load(cam_info_file)
        print("Predicting poses.")
        for result in results:
            key = result["image"]
            prediction = result["prediction"]
            image = cv2.imread(os.path.join(config.IMAGES_PATH, key))
            bb = result["bb"]
            camera_matrix = np.array(cam_info[key]['K']).reshape(3, 3)
            # Adjust c_x of the camera matrix, i.e. principal point x
            camera_matrix[0][2] = camera_matrix[0][2] - bb[1]
            # Adjust c_y of the camera matrix, i.e. principal point y
            camera_matrix[1][2] = camera_matrix[1][2] - bb[0]
            _, rvec, tvec, inliers, reprojection_error = ransac(prediction, image.shape, camera_matrix)
            rotation_matrix = cv2.Rodrigues(rvec)[0]
            translation_vector = tvec
            result_dict = {"R" : rotation_matrix.flatten().tolist(), 
                                    "t" : translation_vector.flatten().tolist(), 
                                    "bb" : bb.flatten().tolist(), 
                                    "obj" : os.path.basename(object_model_path),
                                    "ransac_inliers" : len(inliers.squeeze()),
                                    "reprojection_error" : reprojection_error}
            if merge_mode == "append" and key in converted_results:
                # Only when the user wants to append the pose and there already are some
                # poses for the image
                converted_results[key].append(result_dict)
            else:
                # For overwrite and replace we replace all the content whether
                # it existed or not
                converted_results[key] = [result_dict]

    print("Writing results to {}".format(output_file))
    with open(output_file, "w") as json_file:
        json.dump(OrderedDict(sorted(converted_results.items())), json_file)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to run the FlowerPower network '
                                                  'and compute the poses implied by the coordinate predictions.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    config = inference_config.InferenceConfig()
    config.parse_config_from_json_file(arguments.config)
    inference(config)