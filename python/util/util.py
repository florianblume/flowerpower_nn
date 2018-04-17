""" Utility functions for stuff necessary for the network.
"""

import numpy as np
import os

def crop_image_on_segmentation_color(image, segmentation_mask, color):
    """ This function returns the rectangle that results when cropping the mask
    """
    mask = np.zeros([segmentation_mask.shape[0], segmentation_mask.shape[1]], dtype=np.uint8)
    indices = segmentation_mask == color
    mask[indices] = 255
    return image[np.ix_(mask.any(255), mask.any(0))]

def get_files_at_path_of_extensions(images_path, extensions):
    import os
    return [fn for fn in os.listdir(images_path) if any(fn.endswith(ext) for ext in extensions)]

class GroundTruthIO:
    import cv2

    data_key = 'data'

    def load_gt(gt_path):        
        fs_read = cv2.FileStorage(gt_path, cv2.FILE_STORAGE_READ)
        image = fs_read.getNode(data_key).mat()      
        fs_read.release()
        return image

    def store_gt(gt_path, gt):
        fs_write = cv2.FileStorage(gt_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write(data_key, vis_rgb)
        fs_write.release()