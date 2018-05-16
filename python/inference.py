import os
import shutil
import json
import cv2
import math
import numpy as np
from random import randint

import util.util as util

from model import dataset
from model import model
from model import inference_config

def eulerAnglesToRotationMatrix(theta) :

    ### From https://www.learnopencv.com/rotation-matrix-to-euler-angles/
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def ransac(prediction, cam_info):
    obj_coords = prediction['obj_coords']
    step_y = prediction['step_y']
    step_x = prediction['step_x']
    indices = np.where(np.any(obj_coords != 0, axis=2))
    object_points = []
    image_points = []
    for index in range(len(indices[0])):
        object_points.append(obj_coords[indices[0][index], indices[1][index]])
        image_points.append([indices[0][index], indices[1][index]])
    object_points = np.array(object_points).astype(np.float32)
    image_points = np.array(image_points).astype(np.float32)
    retval, rvec, tvec, inliers  = cv2.solvePnPRansac(object_points, 
                              image_points, 
                              np.array(cam_info['K']).reshape(3, 3), 
                              np.zeros(4)
    )
    return retval, rvec, tvec

def inference(config):

    images_path = config.IMAGES_PATH
    image_extension = config.IMAGE_EXTENSION 
    segmentation_images_path = config.SEGMENTATION_IMAGES_PATH
    segmentation_image_extension = config.SEGMENTATION_IMAGE_EXTENSION
    segmentation_color = config.SEGMENTATION_COLOR
    object_model_path = config.OBJECT_MODEL_PATH 
    cam_info_path = config.CAM_INFO_PATH 
    weights_path = config.WEIGHTS_PATH 
    batch_size = config.BATCH_SIZE
    limit = config.LIMIT
    output_path = config.OUTPUT_PATH
    output_file = config.OUTPUT_FILE

    assert os.path.exists(images_path), \
    		"The images path {} does not exist.".format(images_path)
    assert image_extension in ['png', 'jpg', 'jpeg'], \
    		"Unkown image extension."
    assert os.path.exists(segmentation_images_path), \
    		"The segmentation images path {} does not exist.".format(segmentation_images_path)
    assert segmentation_image_extension in ['png', 'jpg', 'jpeg'], \
    		"Unkown segmentation image extension."
    assert os.path.exists(object_model_path), \
    		"The object model file {} does not exist.".format(object_model_path)
    assert os.path.exists(cam_info_path), \
    		"The camera info file {} does not exist.".format(cam_info_path)
    assert os.path.exists(weights_path), \
    		"The weights file {} does not exist.".format(weights_path)

    image_paths = util.get_files_at_path_of_extensions(images_path, image_extension)
    util.sort_list_by_num_in_string_entries(image_paths)
    segmentation_image_paths = util.get_files_at_path_of_extensions(segmentation_images_path, segmentation_image_extension)
    util.sort_list_by_num_in_string_entries(segmentation_image_paths)

    images = []
    segmentation_images = []
    cropped_segmentation_images = []
    # Bounding boxes
    bbs = []

    print("Preparing data.")

    if limit == 0:
        limit = len(images_paths)

        # TODO: Support file name list

    # Prepare data, i.e. crop images to the segmentation mask
    for index in range(min(len(image_paths), limit)):
        image_path = image_paths[index]
        image = cv2.imread(os.path.join(images_path, image_path))
        segmentation_image_path = segmentation_image_paths[index]
        segmentation_image =cv2.imread(os.path.join(segmentation_images_path, segmentation_image_path))
        image, frame = util.crop_image_on_segmentation_color(
                        image, segmentation_image, segmentation_color, return_frame=True)
        bbs.append(frame)
        cropped_segmentation_image = util.crop_image_on_segmentation_color(
                                segmentation_image, segmentation_image, segmentation_color)
        images.append(image)
        segmentation_images.append(segmentation_image)
        cropped_segmentation_images.append(cropped_segmentation_image)

    # Otherwise datatype is int64 which is not JSON serializable
    bbs = np.array(bbs).astype(np.int32)

    # No support for batching yet
    config.BATCH_SIZE = 1
    print("Running network inference.")
    network_model = model.FlowerPowerCNN('inference', config, output_path)
    results = []
    # We only store the filename + extension
    object_model_path = os.path.basename(object_model_path)

    with open(cam_info_path, "r") as cam_info_file:
        cam_info = json.load(cam_info_file)
        for i in range(len(images)):
            key = image_paths[i]
            prediction = network_model.predict([images[i]], [cropped_segmentation_images[i]], verbose=1)
            # Network returns list as it is suitable for batching
            pose = ransac(prediction[0], cam_info[key])
            rotation_matrix = eulerAnglesToRotationMatrix(pose[1])
            translation_vector = pose[2]
            results.append({key : [{"R" : rotation_matrix.flatten().tolist(), 
                                    "t" : translation_vector.flatten().tolist(), 
                                    "bb" : bbs[i].flatten().tolist(), 
                                    "obj" : object_model_path}]})

    print("Writing results to {}".format(output_file))
    with open(output_file, "w") as json_file:
        json.dump(results, json_file)

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