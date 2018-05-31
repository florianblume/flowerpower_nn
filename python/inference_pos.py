import os
import shutil
import json
import cv2
import numpy as np
import math

import inference as model_inference
import util.util as util
from model import inference_config

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                   0                   ],
                    [0,         math.cos(theta[0]),  math.sin(theta[0])  ],
                    [0,         -math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]),    0,      -math.sin(theta[1])  ],
                    [0,                     1,      0                    ],
                    [math.sin(theta[1]),    0,      math.cos(theta[1])   ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    math.sin(theta[2]),     0],
                    [-math.sin(theta[2]),   math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(np.dot( R_x, R_y ), R_z)
 
    return R

def ransac(prediction, imsize, cam_info):
    obj_coords = prediction['obj_coords']

    step_y = prediction['step_y']
    step_x = prediction['step_x']

    image_points, object_points = util.pair_object_coords_with_index(obj_coords, imsize, step_y, step_x)

    retval, rvec, tvec, inliers  = cv2.solvePnPRansac(object_points, 
                              image_points, 
                              np.array(cam_info['K']).reshape(3, 3), 
                              None,
                              iterationsCount=1000
    )
    return retval, rvec, tvec

def inference(config):

    cam_info_path = config.CAM_INFO_PATH
    object_model_path = config.OBJECT_MODEL_PATH
    output_file = config.OUTPUT_FILE

    assert os.path.exists(cam_info_path), \
            "The camera info file {} does not exist.".format(cam_info_path)

    results = model_inference.inference(config)
    converted_results = {}

    with open(cam_info_path, "r") as cam_info_file:
        cam_info = json.load(cam_info_file)
        for result in results:
            key = result["image"]
            print(key)
            prediction = result["prediction"]
            image = cv2.imread(os.path.join(config.IMAGES_PATH, key))
            pose = ransac(prediction, image.shape, cam_info[key])
            rotation_matrix = eulerAnglesToRotationMatrix(pose[1])
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
    inference(config)