# This class provides a container for all data necessary to train the neural network.

import matplotlib.image

class Dataset:

    # The full paths to the images
    images = []

    # The full paths to the segmentation images
    segmentation_images = []

    # The object coordinates ground truth values
    obj_coord_images = []

    def add_image(image_path):
        images.append(image_path)

    def load_image(im_id):
        return matplotlib.image.imread(images[im_id])

    def get_image_ids():
        return [i for i in range(len(images))]

    def add_segmentation_image(segmentation_image_path):
        segmentation_images.append(segmentation_image_path)

    def load_segmentation_image(seg_im_id):
        return matplotlib.image.imread(segmentation_images[seg_im_id])

    def add_obj_coord_image(obj_coord_image):
        obj_coord_image.append(obj_coord_image)

    def load_obj_coord_image(obj_coord_image_id):
        return matplotlib.image.imread(obj_coord_images[obj_coord_image_id])

    def verify():
        return len(images) == len(segmentation_images) == len(obj_coord_gts)

    def size():
        return len(images)