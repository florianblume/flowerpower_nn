""" Utility functions for stuff necessary for the network.
"""

import numpy as np
import os
import re

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

def sort_list_by_num_in_string_entries(list_of_strings):
    list_of_strings.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])