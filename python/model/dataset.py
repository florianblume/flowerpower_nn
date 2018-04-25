# This class provides a container for all data necessary to train the neural network.

import matplotlib.image

class Dataset:

    # The full paths to the images
    images = []

    # The full paths to the segmentation images
    segmentation_images = []

    # The object coordinates ground truth values
    obj_coord_gts = []

    def add_image(image_path):
        images.append(image_path)

    def get_image(im_id):
        return matplotlib.image.imread(images[im_id])

    def add_segmentation_image(segmentation_image_path):
        segmentation_images.append(segmentation_image_path)

    def get_segmentation_image(seg_im_id):
        return matplotlib.image.imread(segmentation_images[seg_im_id])

    def add_obj_coord_gt(obj_coord_gt):
        obj_coord_gts.append(obj_coord_gt)

    def get_obj_coord_gt(obj_coord_gt_id):
        return gt_io.read_gt(obj_coord_gts[obj_coord_gt_id])

    def verify():
        return len(images) == len(segmentation_images) == len(obj_coord_gts)

    def size():
        return len(images)