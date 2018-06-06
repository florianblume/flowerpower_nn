import tifffile as tiff
import cv2
import numpy as np
import os

import util

def visualize_errors(gt_images_path, prediction_images_path, output_path, output_path_float=None):
    gt_images = util.get_files_at_path_of_extensions(gt_images_path, [".tiff"])
    util.sort_list_by_num_in_string_entries(gt_images)
    prediction_images = util.get_files_at_path_of_extensions(prediction_images_path, [".tiff"])
    util.sort_list_by_num_in_string_entries(prediction_images)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # The threshold to use as maximum (i.e. 255 in RGB) because using the
    # maximum in the prediction itself might make smaller errors vanish
    # if the maximum is a lot larger than the rest of the coordinates.
    coord_threshold = 10
    for index in range(len(prediction_images)):
        prediction_image = prediction_images[index]
        if not prediction_image in gt_images:
            print("Could not find corresponding ground truth image.")
            continue
        gt_image = gt_images[gt_images.index(prediction_image)]
        output_filename = os.path.splitext(gt_image)[0]
        output_filename_float = os.path.join(output_path, output_filename + "_error.tiff")
        output_filename = os.path.join(output_path, output_filename + "_error.png")
        gt_image_loaded = tiff.imread(os.path.join(gt_images_path, gt_image))
        prediction_image_loaded = tiff.imread(os.path.join(prediction_images_path, prediction_image))
        # Bring ground truth to same size, by taking ever other (or i-th) pixel because
        # that's essentially what the network does when the output size is smaller than the
        # input size
        gt_image_loaded = util.shrink_image_with_step_size(gt_image_loaded, prediction_image_loaded.shape)
        diff = gt_image_loaded - prediction_image_loaded
        diff = np.absolute(diff)
        if output_path_float:
            tiff.imwrite(output_filename_float, diff)
        diff = (diff / coord_threshold) * 255
        diff[diff > 255] = 255
        diff = diff.astype(np.int32)
        cv2.imwrite(output_filename, diff)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to visualize errors.')
    parser.add_argument("--gt",
                        required=True,
                        help="The path to the ground truth images.")
    parser.add_argument("--pred",
                        required=True,
                        help="The path to the predicted images.")
    parser.add_argument("--output_path",
                        required=True,
                        help="The path where to store the results.")
    parser.add_argument("--output_path_f",
                        required=False,
                        help="If specified, also saves the float image at the path.")
    args = parser.parse_args()
    visualize_errors(args.gt, args.pred, args.output_path, args.output_path_f)