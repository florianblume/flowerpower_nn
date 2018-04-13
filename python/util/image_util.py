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

def object_coordinates_from_depth_image(depth_image, K, R, t):
    """ This function takes in a rendered depth image and computes the 3D object
    coordinates per pixel with respect to the given camera matrix. 

    Args:
        depth_image ((height, width)): the depth image
        K  ((3, 3)): Intrinsic camera matrix
        R  ((3, 3)): Rotation matrix of the camera
        t     ((3)): Translation vector of the camera

    Returns:
        A two-dimensional array of (X, Y, Z) tuples that represent the object 
        coordinate at the (x, y) location.
    """
    obj_coordinates = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), np.float)
    K = np.array(K)
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
    R = np.array(R)
    R_inv = np.linalg.inv(R)
    t = np.array(t)
    print(t.T[0])
    R_t = np.hstack((R, t))
    R_t = np.vstack((R_t, [0, 0, 0, 1]))
    R_t_inv = np.linalg.pinv(R_t)
    P_inv = np.linalg.pinv(np.vstack((R_t, [0, 0, 0, 1])))

    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                z = depth_image[v, u]
                x = z * (u - c_x) / f_x
                y = z * (v - c_y) / f_y
                X = np.array([x, y, z])
                obj_coordinates[v, u] = X
            else:
                obj_coordinates[v, u] = 0

    return obj_coordinates