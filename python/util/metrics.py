import os
import shutil
import json
import cv2
import numpy as np
import math
from shutil import copyfile

from collections import OrderedDict

from . import util
from . import tless_inout as inout
import tifffile as tiff

<<<<<<< HEAD
def get_summary(mean_errors, coordinate_inliers, angle_errors, distance_errors, pose_errors, accepted):
=======
def get_summary(mean_errors, coordinate_inliers, angle_errors, distance_errors, pose_errors):
>>>>>>> c3056213b77ae3f9838bab893d28a20f53ee350d
    mean_errors.sort()
    coordinate_inliers.sort(reverse=True)
    angle_errors.sort()
    distance_errors.sort()
    pose_errors.sort()

    mean = {'pixel_error' : np.mean(mean_errors).astype(np.float64),
            'coordinate_inliers' : np.mean(coordinate_inliers),
            'angle_error' : np.mean(angle_errors).astype(np.float64),
            'distance_error' : np.mean(distance_errors),
            'pose_error' : np.mean(pose_errors)}
    len_mean_errors = len(mean_errors)
    len_coordinate_inliers = len(coordinate_inliers)
    len_angle_errors = len(angle_errors)
    len_distance_errors = len(distance_errors)
    len_pose_errors = len(pose_errors)

    median = {'25' : {}, '50' : {}, '75' : {}}


    if len_mean_errors > 0:
        median['25']['pixel_error'] = mean_errors[int(len_mean_errors * 0.25)].astype(np.float64)
        median['50']['pixel_error'] = mean_errors[int(len_mean_errors * 0.50)].astype(np.float64)
        median['75']['pixel_error'] = mean_errors[int(len_mean_errors * 0.75)].astype(np.float64)

    if len_coordinate_inliers > 0:
        median['25']['inliers'] = coordinate_inliers[int(len_coordinate_inliers * 0.25)]
        median['50']['inliers'] = coordinate_inliers[int(len_coordinate_inliers * 0.50)]
        median['75']['inliers'] = coordinate_inliers[int(len_coordinate_inliers * 0.75)]

    if len_angle_errors > 0:
        median['25']['angle_error'] = angle_errors[int(len_angle_errors * 0.25)]
        median['50']['angle_error'] = angle_errors[int(len_angle_errors * 0.50)]
        median['75']['angle_error'] = angle_errors[int(len_angle_errors * 0.75)]

    if len_distance_errors > 0:
        median['25']['distance_error'] = distance_errors[int(len_distance_errors * 0.25)]
        median['50']['distance_error'] = distance_errors[int(len_distance_errors * 0.50)]
        median['75']['distance_error'] = distance_errors[int(len_distance_errors * 0.75)]

    if len_pose_errors > 0:
        median['25']['pose_error'] = pose_errors[int(len_pose_errors * 0.25)]
        median['50']['pose_error'] = pose_errors[int(len_pose_errors * 0.50)]
        median['75']['pose_error'] = pose_errors[int(len_pose_errors * 0.75)]

    return mean, median

def calculate_metrics(gt_images_path, pred_images_path, gt_path, obj_path, 
                        pred_path, image_extension, output_file):
    gt_images_files = util.get_files_at_path_of_extensions(gt_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(gt_images_files)
    pred_images_files = util.get_files_at_path_of_extensions(pred_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(pred_images_files)

    results = {'images' : {}}
    # Store these extra to be able to compute mean and median later
    mean_errors = []
    coordinate_inliers = []
    angle_errors = []
    distance_errors = []
    pose_errors = []
<<<<<<< HEAD
    accepted = []
=======
>>>>>>> c3056213b77ae3f9838bab893d28a20f53ee350d

    loaded_obj = inout.load_ply(obj_path)
    mesh_points = loaded_obj['pts']
    x_max = np.amax(mesh_points[:,0]) 
    x_min = np.amin(mesh_points[:,0])
    y_max = np.amax(mesh_points[:,1]) 
    y_min = np.amin(mesh_points[:,1])
    z_max = np.amax(mesh_points[:,2]) 
    z_min = np.amin(mesh_points[:,2])
    # Diameter of object and khs fixed value to be able to compute Hinterstoisser et al. metric
    d = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    khs = 0.1
    ones = np.ones([mesh_points.shape[0]])
    mesh_points = np.array(mesh_points).T
    mesh_points = np.concatenate([mesh_points, [ones]])

    obj = os.path.basename(obj_path)

    with open(gt_path, 'r') as gt_file, \
         open(pred_path, 'r') as pred_file, \
         open(output_file, 'w') as output:
        gt_data = json.load(gt_file)
        pred_data = json.load(pred_file)
        # Compute mean pixel error and inlier count
        for pred_image_file in pred_images_files:
            if pred_image_file.startswith('obj_coords_'):
                split = pred_image_file.split('obj_coords_')[1].split('.tiff')[0]
                copyfile(os.path.join(pred_images_path, pred_image_file), os.path.join(pred_images_path, split + '_obj_coords.tiff'))
                os.remove(os.path.join(pred_images_path, pred_image_file))
                pred_image_file = split + '_obj_coords.tiff'
            result = {}
            if pred_image_file in gt_images_files:
                gt = tiff.imread(os.path.join(gt_images_path, pred_image_file))
                pred = tiff.imread(os.path.join(pred_images_path, pred_image_file))
                shrinked_gt = util.shrink_image_with_step_size(gt, pred.shape)
                # TODO use only valid locations according to segmentation mask
                diff = np.absolute(shrinked_gt - pred)
                summed_diff = np.linalg.norm(diff, axis=2)
                coordinate_inlier = np.where((summed_diff < 2) & (summed_diff > 0))
                coordinate_inlier_count = int(coordinate_inlier[0].shape[0] / 3.0)
                mean = np.mean(summed_diff)
                mean_errors.append(mean)
                coordinate_inliers.append(coordinate_inlier_count)
                result['pixel_error'] = mean.astype(np.float64)
                result['coordinate_inliers'] = coordinate_inlier_count
                image_file_name = pred_image_file.split("_obj_coords.tiff")[0]
            else:
                print("Could not find corresponding groundtruth object coordinates for {}.".format(pred_image_file))

        # We can still calculate the error of the pose even if we do not have
        # the ground truth object coordiantes or the predicted object coordinates
        for pred_image in pred_data:
            result = results['images'][pred_image] if pred_image in results['images'] else {}
            if pred_image in gt_data:
                pred_entries_for_image = pred_data[pred_image]
                gt_entries_for_image = gt_data[pred_image]
                for pred_entry in [entry for entry in pred_entries_for_image if entry['obj'] == obj]:
                    for gt_entry in gt_entries_for_image:
                        if gt_entry['obj'] == pred_entry['obj']:
                            # GT pose
                            gt_rot_matrix = np.array(gt_entry['R']).reshape(3, 3)
                            gt_t_vec = np.array(gt_entry['t'])
                            gt_transform = np.vstack([gt_rot_matrix, gt_t_vec]).T
                            # Predicted pose
                            pred_rot_matrix = np.array(pred_entry['R']).reshape(3, 3)
                            pred_t_vec = np.array(pred_entry['t'])
                            pred_transform = np.vstack([pred_rot_matrix, pred_t_vec]).T
                            # Hinterstoisser et al. pose error
                            pose_error = np.dot(gt_transform, mesh_points) - np.dot(pred_transform, mesh_points)
                            pose_error = np.linalg.norm(pose_error, axis=0)
                            pose_error = np.mean(pose_error)
                            pose_errors.append(pose_error)
                            result['pose_error'] = pose_error
                            result['pose_accepted'] = bool(pose_error <= khs * d)
                            accepted.append(result['pose_accepted'])
                            # Euclidean distance error
                            distance = np.linalg.norm(gt_t_vec - pred_t_vec)))
                            distance_errors.append(distance)
                            result['distance_error'] = distance
                            # Rotation angle error
                            matrix_diff = np.dot(gt_rot_matrix.T, pred_rot_matrix)
                            angle = np.rad2deg(np.arccos((np.trace(matrix_diff) - 1) / 2))
                            angle_errors.append(angle)
                            result['angle_error'] = angle
                            results['images'][pred_image] = result
                            gt_entry_found = True
                    if not gt_entry_found:
                        print("Could not find corresponding ground truth entry for {}.".format(pred_image))
            else:
                print("Could not find corresponding ground truth entry for {}.".format(pred_image))

        mean, median = get_summary(mean_errors, coordinate_inliers, angle_errors, \
                                distance_errors, pose_errors, accepted)
        results['mean'] = mean
        results['median'] = median
        results['pose_inliers_percentage'] = (np.where(np.array(accepted) == True)[0].shape[0] / float(len(accepted))) * 100
        results['images'] = OrderedDict(sorted(results['images'].items()))
        print("Result stored at {}.".format(output_file))
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
    parser.add_argument("--obj_path",
                        required=True,
                        help="The full path to the object model filename with extension.")
    parser.add_argument("--image_extension",
                        required=True,
                        help="The extension of the original (i.e. the photographs) images."
                             " Specify without the dot.")
    parser.add_argument("--output_file",
                        required=True,
                        help="The path where to store the results.")
    args = parser.parse_args()
    calculate_metrics(args.gt_images, args.pred_images, args.gt, args.obj_path, 
                        args.pred, args.image_extension, args.output_file)