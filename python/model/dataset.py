# This class provides a container for all data necessary to train the neural network.

import matplotlib.image
import tifffile as tiff

class Dataset:

    # The full paths to the images
    images = []

    # The full paths to the segmentation images
    segmentation_images = []

    # The object coordinates ground truth values
    obj_coord_images = []

    def add_image(self, image_path):
        self.images.append(image_path)

    def get_image(self, image_id):
        return self.images[image_id]

    def load_image(self, im_id):
        return matplotlib.image.imread(self.images[im_id])

    def get_image_ids(self):
        return [i for i in range(len(self.images))]

    def add_segmentation_image(self, segmentation_image_path):
        self.segmentation_images.append(segmentation_image_path)

    def load_segmentation_image(self, seg_im_id):
        return matplotlib.image.imread(self.segmentation_images[seg_im_id])

    def add_obj_coord_image(self, obj_coord_image):
        self.obj_coord_images.append(obj_coord_image)

    def load_obj_coord_image(self, obj_coord_image_id):
        return tiff.imread(self.obj_coord_images[obj_coord_image_id])

    def verify(self):
        return len(self.images) == len(self.segmentation_images) == len(self.obj_coord_images)

    def size(self):
        return len(self.images)