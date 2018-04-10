""" This script provides functionality to work with and modify images.
"""

import numpy as np
import os
import cv2

def crop_image_on_segmentation_color(image, segmentation_mask, color):
    """ This function returns the rectangle that results when cropping the mask
    """
    mask = np.zeros([segmentation_mask.shape[0], segmentation_mask.shape[1]], dtype=np.uint8)
    indices = segmentation_mask == color
    mask[indices] = 255
    return image[np.ix_(mask.any(255), mask.any(0))]

def object_coordinates_from_depth_image(depth_image, f_x, f_y, c_x, c_y):
    """ This function takes in a rendered depth image and computes the 3D object
    coordinates per pixel with respect to the given camera matrix. 

    Args:
        depth_image ((width, height)): the depth image
        f_x (float): focal length x
        f_y (float): focal length y
        c_x (float): principal point x
        c_y (float): principal point y

    Returns:
        A two-dimensional array of (X, Y, Z) tuples that represent the object 
        coordinate at the (x, y) location.
    """
    # We need the indices to calculate the resulting 3D point
    indices_x = np.arange(depth_image.shape[0])
    indices_y = np.arange(depth_image.shape[1])
    # 3-tuple that is going to hold the coordinates
    coordinates = (depth_image.shape[0], depth_image.shape[1], 3)

    coordinates[:,:,0] = (indices_x - c_x).multiply(depth_image) / f_x
    coordinates[:,:,1] = (indices_y - c_y).multiply(depth_image) / f_y
    coordinates[:,:,2] = depth_image

    return coordinates