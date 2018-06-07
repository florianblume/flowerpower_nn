import os
import shutil
import json
import cv2
import numpy as np

import util.util as util
import tifffile as tiff

def calculate_metrics(gt_images_path, pred_images_path, gt_path, pred_path):
    gt_images_files = util.get_files_at_path_of_extensions(gt_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(gt_images_files)
    pred_images_files = util.get_files_at_path_of_extensions(pred_images_path, ['tiff'])
    util.sort_list_by_num_in_string_entries(pred_images_files)

    # Compute mean pixel error and inlier count
    for pred_image_file in pred_images_files:
        if pred_images_file in gt_images_files:
            gt = tiff.imread(os.path.join(gt_images_path, pred_images_file))
            pred = tiff.imread(os.path.join(pred_images_path, pred_images_file))
            shrinked_gt = util.shrink_image_with_step_size(gt, pred.shape)
            diff = np.absolute(gt - pred)
            inlier = np.where(diff < 2)
            print(inlier)

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
    parser.add_argument("--outupt_file",
                        required=True,
                        help="The path where to store the results.")
    args = parser.parse_args()
    train(args.gt_images, args.pred_images, args.gt, args.pred)