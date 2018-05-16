# Configuration of the neural network

import config.Config

class InferenceConfig(config.Config):

    # The path to the images
    IMAGES_PATH = ""

    # The extension of the images
    IMAGE_EXTENSION = ""

    # The path to the segmentation images
    SEGMENTATION_IMAGES_PATH = ""

    # The extension of the images
    SEGMENTATION_IMAGE_EXTENSION = ""

    # The color for the object model in the segmentation image
    SEGMENTATION_COLOR = [255, 255, 255]
    
    IMAGE_DIM = 500

    # The path to the weights to use for inference
    WEIGHTS_PATH = ""

    # Batch size is GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1

    IMAGES_PER_GPU = 26

    # The path to the camera info file related to the images
    CAM_INFO_PATH = ""

    # The file that the results are going to be written to
    OUTPUT_FILE = ""

    # The number of RANSAC iterations to compute the final pose
    # per image
    RANSAC_ITERATIONS = 10