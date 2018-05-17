""" Utility functions for stuff necessary for the network.
"""

import numpy as np
import os
import re

def merge_gts(gt1, gt2):
    pass

def crop_image_on_segmentation_color(image, segmentation_mask, color, return_frame=False):
    """ This function returns the rectangle that results when cropping the mask
    """
    indices = np.where(segmentation_mask == color)
    y_indices = indices[0]
    y_start = np.min(y_indices)
    y_end = np.max(y_indices)
    x_indices = indices[1]
    x_start = np.min(x_indices)
    x_end = np.max(x_indices)
    cropped_image = image[y_start : y_end + 1, x_start : x_end + 1]
    if return_frame:
        return cropped_image, (y_start, y_end + 1, x_start, x_end + 1)
    else:
        return cropped_image

def get_files_at_path_of_extensions(path, extensions):
    import os
    return [fn for fn in os.listdir(path) if any(fn.endswith(ext) for ext in extensions)]

def sort_list_by_num_in_string_entries(list_of_strings):
    list_of_strings.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

def is_number(n, _type):
    try:
        _type(n)
    except ValueError:
        return False
    return True

def convert_printed_to_numpy_array(printed_array, _type):
    temp = printed_array.split("[")
    # Flatten array
    temp = [s.split("]") for s in temp]
    temp = [s for x in temp for s in x]
    temp = [s.split(" ") for s in temp]
    temp = [s for x in temp for s in x]
    temp = [_type(n) for n in temp if is_number(n, _type)]
    return np.array(temp)