""" Utility functions for stuff necessary for the network.
"""

import numpy as np
import os
import re
import math

from ..model import model_util

def pair_object_coords_with_index(image, original_im_size, step_y, step_x):
    """ This function pairs the object coords in a detection image with their
    actual index in the original image. The image shrinks during inference
    thus the index of the final detection does not correspond with the index
    in the source image. 
    """
    steps_y, steps_x = model_util.create_index_array_from_step_sizes(step_y, 
                                                          original_im_size[0],
                                                          original_im_size[0] - 1,
                                                          step_x,  
                                                          original_im_size[1],
                                                          original_im_size[1] - 1)

    object_points = []
    image_points = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            obj_coord = image[i][j]
            if np.any(obj_coord != 0):
                # If all coords are 0, then we are outside of the segmentation mask
                object_points.append(obj_coord)
                image_points.append([steps_y[i], steps_x[j]])

    object_points = np.array(object_points).astype(np.float32)
    image_points = np.array(image_points).astype(np.float32)
    return image_points, object_points

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
        return cropped_image, (y_start, x_start, y_end + 1, x_end + 1)
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