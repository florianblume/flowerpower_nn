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

def ransac(prediction, imsize, cam_info):
    obj_coords = prediction['obj_coords']

    step_y = prediction['step_y']
    step_x = prediction['step_x']

    image_points, object_points = util.pair_object_coords_with_index(obj_coords, imsize, step_y, step_x)
    retval, rvec, tvec, inliers  = cv2.solvePnPRansac(object_points, 
                              image_points, 
                              np.array(cam_info['K']).reshape(3, 3), 
                              None,
                              iterationsCount=100
    )

    # If z value is negative we need to flip the translation vector
    if tvec[2] < 0:
        tvec = tvec * -1

    return retval, rvec, tvec

def inference(base_path, config):

    cam_info_path = config.CAM_INFO_PATH
    object_model_path = config.OBJECT_MODEL_PATH
    output_file = os.path.join(base_path, config.OUTPUT_FILE)

    assert os.path.exists(cam_info_path), \
            "The camera info file {} does not exist.".format(cam_info_path)

    results = inference_script.inference(base_path, config)
    converted_results = OrderedDict()

    with open(cam_info_path, "r") as cam_info_file:
        cam_info = json.load(cam_info_file)
        print("Predicting poses.")
        for result in results:
            key = result["image"]
            prediction = result["prediction"]
            image = cv2.imread(os.path.join(config.IMAGES_PATH, key))
            pose = ransac(prediction, image.shape, cam_info[key])
            rotation_matrix = cv2.Rodrigues(pose[1])[0]
            # Translation is inverted somehow
            translation_vector = pose[2]
            converted_results[key] = [{"R" : rotation_matrix.flatten().tolist(), 
                                    "t" : translation_vector.flatten().tolist(), 
                                    "bb" : result["bb"].flatten().tolist(), 
                                    "obj" : os.path.basename(object_model_path)}]

    print("Writing results to {}".format(output_file))
    with open(output_file, "w") as json_file:
        json.dump(converted_results, json_file)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to run the FlowerPower network.')
    parser.add_argument("--config",
                        required=True,
                        help="The path to the config file.")
    arguments = parser.parse_args()
    config = inference_config.InferenceConfig()
    config.parse_config_from_json_file(arguments.config)
    inference(os.path.dirname(arguments.config), config)