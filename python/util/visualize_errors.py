import tifffile as tiff
import cv2
import numpy as np
import os

import util

def visualize_errors(images1_path, images2_path, output_path, output_path_float=None):
    images1 = util.get_files_at_path_of_extensions(images1_path, [".tiff"])
    util.sort_list_by_num_in_string_entries(images1)
    images2 = util.get_files_at_path_of_extensions(images2_path, [".tiff"])
    util.sort_list_by_num_in_string_entries(images2)
    for index in range(len(images1)):
        image1 = images1[index]
        image2 = images2[index]
        output_filename = os.path.splitext(image1)[0]
        output_filename_float = os.path.join(output_path, output_filename + "_error.tiff")
        output_filename = os.path.join(output_path, output_filename + "_error.png")
        image1_loaded = tiff.imread(os.path.join(images1_path, image1))
        image2_loaded = tiff.imread(os.path.join(images2_path, image2))
        # Bring ground truth to same size, by taking ever other (or i-th) pixel because
        # that's essentially what the network does when the output size is smaller than the
        # input size
        image1_loaded = util.shrink_image_with_step_size(image1_loaded, image2_loaded.shape)
        diff = image1_loaded - image2_loaded
        if output_path_float:
            tiff.imwrite(output_filename_float, diff)
        max_r = np.amax(diff[:,:,0])
        max_g = np.amax(diff[:,:,1])
        max_b = np.amax(diff[:,:,2])
        diff = (diff[:,:] / [max_r, max_g, max_b]) * 255
        diff = diff.astype(np.int32)
        #cv2.imwrite(output_filename, diff)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to visualize errors.')
    parser.add_argument("--images1",
                        required=True,
                        help="The path to the ground truth images.")
    parser.add_argument("--images2",
                        required=True,
                        help="The path to the inference images.")
    parser.add_argument("--output_path",
                        required=True,
                        help="The path where to store the results.")
    parser.add_argument("--output_path_float",
                        required=False,
                        help="If specified, also saves the float image at the path.")
    args = parser.parse_args()
    visualize_errors(args.images1, args.images2, args.output_path, args.output_path_float)