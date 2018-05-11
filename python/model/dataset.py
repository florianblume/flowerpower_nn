# This class provides a container for all data necessary to train the neural network.

import cv2
import tifffile as tiff
import numpy as np

class Dataset:

    # TODO: convert to use training examples
    def __init__(self):
        self.training_examples = []

    def add_training_example(self, image_path, segmentation_image_path, obj_coord_image_path):
        self.training_examples.append({"image" : image_path,
                                  "segmentation" : segmentation_image_path,
                                  "obj_coord" : obj_coord_image_path})

    def get_image(self, image_id):
        return self.training_examples[image_id]["image"]

    def load_image(self, im_id):
        return cv2.imread(self.get_image(im_id))

    def get_image_ids(self):
        return [i for i in range(len(self.training_examples))]

    def get_segmentation_image(self, seg_im_id):
        return self.training_examples[seg_im_id]["segmentation"]

    def load_segmentation_image(self, seg_im_id):
        return cv2.imread(self.get_segmentation_image(seg_im_id))

    def get_obj_coord_image(self, obj_coord_image):
        return self.training_examples[obj_coord_image]["obj_coord"]

    def load_obj_coord_image(self, obj_coord_image_id):
        return tiff.imread(self.get_obj_coord_image(obj_coord_image_id)).astype(np.float32)

    def size(self):
        return len(self.training_examples)