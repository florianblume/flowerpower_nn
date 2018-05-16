import numpy as np

class Config:

    def parse_config_from_json_file(self, json_file):
        import os
        import json

        with open(json_file, 'r') as opened_json_file:
            config_data = json.load(opened_json_file)
            for entry in config_data:
                setattr(self, entry, config_data[entry])
            self.__init__()

    NAME = "Default"  # Override in config class

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Strides for the ResNet graph, where applicable
    STRIDES = 2

    # Maximum size of the images
    IMAGE_DIM = 300

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_DIM, self.IMAGE_DIM, 3])