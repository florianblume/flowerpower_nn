import os
import shutil
import json
import cv2
import numpy as np
from numpy.linalg import inv
import math

import util.util as util
import tifffile as tiff

def get_summary(mean_errors, inliers, angle_errors, distance_errors):
    mean_errors.sort()
    inliers.sort(reverse=True)
    angle_errors.sort()
    distance_errors.sort()
    mean = {'pixel_error' : np.mean(mean_errors).astype(np.float64),
            'inlier' : np.mean(inliers),
            'angle_error' : np.mean(angle_errors).astype(np.float64),
            'distance_error' : np.mean(distance_errors)}
    len_mean_errors = len(mean_errors)
    len_inliers = len(inliers)
    len_angle_errors = len(angle_errors)
    len_distance_errors = len(distance_errors)

    median = {'25' : {}, '50' : {}, '75' : {}}


    if len_mean_errors > 0:
        median['25']['pixel_error'] = mean_errors[int(len_mean_errors * 0.25)].astype(np.float64)
        median['50']['pixel_error'] = mean_errors[int(len_mean_errors * 0.50)].astype(np.float64)
        median['75']['pixel_error'] = mean_errors[int(len_mean_errors * 0.75)].astype(np.float64)

    if len_inliers > 0:
        median['25']['inliers'] = inliers[int(len_inliers * 0.25)]
        median['50']['inliers'] = inliers[int(len_inliers * 0.50)]
        median['75']['inliers'] = inliers[int(len_inliers * 0.75)]

    if len_angle_errors > 0:
        median['25']['angle_error'] = angle_errors[int(len_angle_errors * 0.25)]
        median['50']['angle_error'] = angle_errors[int(len_angle_errors * 0.50)]
        median['75']['angle_error'] = angle_errors[int(len_angle_errors * 0.75)]

    if len_distance_errors > 0:
        median['25']['distance_error'] = distance_errors[int(len_distance_errors * 0.25)]
        median['50']['distance_error'] = distance_errors[int(len_distance_errors * 0.50)]
        median['75']['distance_error'] = distance_errors[int(len_distance_errors * 0.75)]

    return mean, median

def calculate_metrics(gt_images_path, pred_images_path, gt_path, obj, 
                        pred_path, image_extension, output_file):
    gt_images_files = util.get_files_at_path_of_extensions(gt_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(gt_images_files)
    pred_images_files = util.get_files_at_path_of_extensions(pred_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(pred_images_files)

    results = {'images' : {}}
    # Store these extra to be able to compute mean and median later
    mean_errors = []
    inliers = []
    angle_errors = []
    distance_errors = []

    with open(gt_path, 'r') as gt_file, \
         open(pred_path, 'r') as pred_file, \
         open(output_file, 'w') as output:
        gt_data = json.load(gt_file)
        pred_data = json.load(pred_file)
        # Compute mean pixel error and inlier count
        for pred_image_file in pred_images_files:
            result = {}
            if pred_image_file in gt_images_files:
                gt = tiff.imread(os.path.join(gt_images_path, pred_image_file))
                pred = tiff.imread(os.path.join(pred_images_path, pred_image_file))
                shrinked_gt = util.shrink_image_with_step_size(gt, pred.shape)
                diff = np.absolute(shrinked_gt - pred)
                inlier = np.where((diff < 2) & (diff > 0))
                inlier_count = int(inlier[0].shape[0] / 3.0)
                mean = np.mean(diff)
                mean_errors.append(mean)
                inliers.append(inlier_count)
                result['pixel_error'] = mean.astype(np.float64)
                result['inliers'] = inlier_count

            # We can still calculate the error of the pose even if we do not have
            # the ground truth object coordiantes
            original_image_file = pred_image_file.split("obj_coords_")[1]
            original_image_file = original_image_file.split(".tiff")[0]
            original_image_file = original_image_file + "." + image_extension
            if original_image_file in gt_data:
                gt_entry_for_image = gt_data[original_image_file]
                for gt_entry in gt_entry_for_image:
                    if gt_entry['obj'] == obj:
                        pred_entry = next((x for x in pred_data[original_image_file] if x['obj'] == obj), None)
                        gt_rot_matrix = np.array(gt_entry['R']).reshape(3, 3)
                        gt_t_vec = np.array(gt_entry['t'])
                        pred_rot_matrix = np.array(pred_entry['R']).reshape(3, 3)
                        pred_t_vec = np.array(pred_entry['t'])
                        distance = np.sqrt(np.sum(np.square(gt_t_vec - pred_t_vec)))
                        matrix_diff = np.dot(pred_rot_matrix, inv(gt_rot_matrix))
                        rotation_vector = cv2.Rodrigues(matrix_diff)[0]
                        # See https://www.ctcms.nist.gov/~langer/oof2man/RegisteredClass-Rodrigues.html
                        rotation_vector_mag = np.sqrt(np.sum(np.square(rotation_vector)))
                        angle = math.atan(rotation_vector_mag) * 2
                        angle_errors.append(angle)
                        distance_errors.append(distance)
                        result['angle_error'] = angle
                        result['distance_error'] = distance

            results['images'][original_image_file] = result
        mean, median = get_summary(mean_errors, inliers, angle_errors, distance_errors)
        results['mean'] = mean
        results['median'] = median
        print("Result store at {}.".format(output_file))
        json.dump(results, output)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='This script provides functionality to calculate metrics for a set of '
                    'object coordinate predictions and ground truth object coordinates.')
    parser.add_argument("--gt_images",
                        required=True,
                        help="The path to the ground truth object coordinates.")
    parser.add_argument("--pred_images",
                        required=True,
                        help="The path to the predicted object coordinates.")
    parser.add_argument("--gt",
                        required=True,
                        help="The path to the ground truth poses file.")
    parser.add_argument("--pred",
                        required=True,
                        help="The path to the predicted poses file.")
    parser.add_argument("--obj",
                        required=True,
                        help="The name of the object model to calculate the metrics for.")
    parser.add_argument("--image_extension",
                        required=True,
                        help="The extension of the original (i.e. the photographs) images."
                             " Specify without the dot.")
    parser.add_argument("--output_file",
                        required=True,
                        help="The path where to store the results.")
    args = parser.parse_args()
    calculate_metrics(args.gt_images, args.pred_images, args.gt, args.obj, 
                        args.pred, args.image_extension, args.output_file)