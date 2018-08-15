import os
import shutil
import json
import cv2
import tifffile as tiff
import numpy as np
from collections import OrderedDict

import util

def segmentation_iou_and_reprojection_error(gt_seg, pred_seg, color, output_file):
    result = OrderedDict()

    gt_segmentation_images = util.get_files_at_path_of_extensions(gt_seg, ['png'])
    pred_segmentation_images = util.get_files_at_path_of_extensions(pred_seg, ['png'])

    util.sort_list_by_num_in_string_entries(gt_segmentation_images)
    util.sort_list_by_num_in_string_entries(pred_segmentation_images)

    for i in range(len(pred_segmentation_images)):
        segmentation_image_name = pred_segmentation_images[i]
        if segmentation_image_name in gt_segmentation_images:
            image_filename = pred_segmentation_images[i].split("_segmentation.png")[0] + ".jpg"

            # Compute segmentation IOU
            gt_segmentation_image = cv2.imread(os.path.join(gt_seg, segmentation_image_name))
            pred_segmentation_image = cv2.imread(os.path.join(pred_seg, segmentation_image_name))
            # Check if rendered segmentation mask is empty for some reason
            if pred_segmentation_image is not None:
                print('Processing {}'.format(image_filename))
                max_shape = [max(gt_segmentation_image.shape[0], pred_segmentation_image.shape[0]),
                            max(gt_segmentation_image.shape[1], pred_segmentation_image.shape[1])]

                # We do not need three dimensions, we only need a 2D boolean mask where the segmentation
                # is equal to the desired color
                gt_segmentation_image_bool = np.zeros(max_shape)
                gt_indices = np.all(gt_segmentation_image == color, axis=2)
                # Fill mask to match largest size
                gt_padding = ((0, max_shape[0] - gt_segmentation_image.shape[0]), (0, max_shape[1] - gt_segmentation_image.shape[1]))
                gt_indices = np.pad(gt_indices, gt_padding, 'constant', constant_values=False)
                gt_segmentation_image_bool[gt_indices] = 1

                pred_indices = np.all(pred_segmentation_image == color, axis=2)
                pred_padding = ((0, max_shape[0] - pred_segmentation_image.shape[0]), (0, max_shape[1] - pred_segmentation_image.shape[1]))
                pred_indices = np.pad(pred_indices, pred_padding, 'constant', constant_values=False)
                pred_segmentation_image_bool = np.zeros(max_shape)
                pred_segmentation_image_bool[pred_indices] = 1

                intersection = gt_segmentation_image_bool * pred_segmentation_image_bool
                union = (gt_segmentation_image_bool + pred_segmentation_image_bool) / 2
                iou = intersection.sum() / float(union.sum())
                result[image_filename] = {}
                result[image_filename]["iou"] = iou
        else:
            print('No corresponding ground-truth segmentation found for {}'.format(segmentation_image_name))

    with open(output_file, 'w') as out_file:
        json.dump(result, out_file)

if __name__ == '__main__':
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to compute the IoU between'
                                                 ' a set of segmentation images and simultaneously the reprojection error'
                                                 ' of 6D poses implied by the object coordinate predictions. The script assumes'
                                                 ' that the number of segmentation images in the two folders matches and that'
                                                 ' the first object coordinate image belongs to the first segmentation image etc.')
    parser.add_argument("--gt_seg",
                        required=True,
                        help="The path to the first folder with segmentation images.")
    parser.add_argument("--pred_seg",
                        required=True,
                        help="The path to the second folder with segmentation images.")
    parser.add_argument("--color",
                        required=True,
                        nargs=3, 
                        type=int,
                        help="The color in the segmentation images to compare.")
    parser.add_argument("--output_file",
                        required=True,
                        help="The file that the results are to be written to.")
    args = parser.parse_args()
    segmentation_iou_and_reprojection_error(args.gt_seg, args.pred_seg, args.color, args.output_file)