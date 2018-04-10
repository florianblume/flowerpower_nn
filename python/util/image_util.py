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
        depth_image ((width, height)): the depth image
        K  ((3, 3)): Intrinsic camera matrix
        R  ((3, 3)): Rotation matrix of the camera
        t     ((3)): Translation vector of the camera

    Returns:
        A two-dimensional array of (X, Y, Z) tuples that represent the object 
        coordinate at the (x, y) location.
    """

    # Camera parameters
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # Construct the matrix that is to be multplied inversely to the points to switch
    # from camera space to object space
    M = np.concatenate((R, t), axis=1)
    bottom_row = np.zeros(4)
    bottom_row[3] = 1
    bottom_row = [bottom_row]
    # Add the bottom axis to convert the matrix to homogeneous coordinates
    M = np.concatenate((M, bottom_row), axis=0)
    # The matrix was formerly multiplied onto the object coordiantes, thus we have to
    # inverse-mutliply it onto the rendered points to get from camera space to object space
    M = np.linalg.inv(M)

    # We need the indices to calculate the resulting 3D point
    indices_x = np.arange(depth_image.shape[0])
    indices_y = np.arange(depth_image.shape[1])
    # 3-tuple that is going to hold the coordinates
    cam_coordinates = np.zeros((depth_image.shape[0], depth_image.shape[1], 4), dtype=np.float)

    print(cam_coordinates[:])

    cam_coordinates[:,:,0] = np.multiply(indices_x - c_x, depth_image) / f_x
    cam_coordinates[:,:,1] = np.multiply(indices_y - c_y, depth_image) / f_y
    cam_coordinates[:,:,2] = depth_image
    # To be in homogeneous coordinates
    cam_coordinates[:,:,3] = 1

    obj_coordinates = [depth_image.shape[0], depth_image.shape[1], 3]
    obj_coordinates = np.tensordot(M, cam_coordinates, axes=([1],[2]))

    return obj_coordinates