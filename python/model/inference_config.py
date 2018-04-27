# Configuration of the neural network

import config.Config

class InferenceConfig(config.Config):

    # The path to the images
    IMAGES_PATH = ""

    # The extension of the images
    IMAGE_EXTENSION = "jpg"

    # The path to the segmentation images
    SEGMENTATION_IMAGES_PATH = ""

    # The color for the object model in the segmentation image
    SEGMENTATION_COLOR = [255, 255, 255]

    # The path to the weights to use for inference
    WEIGHTS_PATH = ""

    # The path to the camera info file related to the images
    CAM_INFO_PATH = ""

    # The path where the network can create necessary data
    # This is not the log folder where the network will store
    # its weights, etc, but only the rendered images etc
    OUTPUT_PATH = ""