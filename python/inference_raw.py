import os
import shutil
import json
import cv2
import numpy as np
import tifffile as tiff

import inference as model_inference
from model import model_util
from model import inference_config

def inference(config):
    results = model_inference.inference(config)

    # Path exsitence check during inference
    segmentation_images_path = config.SEGMENTATION_IMAGES_PATH
    output_path = config.OUTPUT_PATH

    for i, result in enumerate(results):
        image_filename = result["image"]
        image_filename_without_extension = os.path.splitext(image_filename)[0]
        # TODO: only 0 as long as only one image is detected and images are not batched for inference
        prediction = result["prediction"]["obj_coords"]
        segmentation_image_filename = result["segmentation_image"]
        segmentation_image = cv2.imread(os.path.join(segmentation_images_path, segmentation_image_filename))
        shrunk_segmentation_image = model_util.shrink_image_with_step_size(segmentation_image, prediction.shape)
        final_image = np.zeros(prediction.shape, dtype=np.float16)
        indices = np.where(shrunk_segmentation_image == config.SEGMENTATION_COLOR)
        final_image[indices] = prediction[indices]
        tiff.imsave(os.path.join(output_path, "obj_coordinates_{}.tiff".format(image_filename_without_extension)),
                    final_image)

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