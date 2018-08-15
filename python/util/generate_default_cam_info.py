def convert_tless_gt(images_path, image_extension, fovx, fovy, output_path):
    import os
    import cv2
    import json
    import util
    import math
    from collections import OrderedDict

    assert os.path.exists(images_path), "Images path does not exist."
    assert image_extension in ['png', 'jpg', 'jpeg', 'tiff'], "Unkown image file format."


    with open(output_path, "w") as json_info:

        cam_info = OrderedDict()

        image_filenames = util.get_files_at_path_of_extensions(images_path, [image_extension])
        util.sort_list_by_num_in_string_entries(image_filenames)

        for filename in image_filenames:
            cam_info[filename] = {}
            image  = cv2.imread(os.path.join(images_path, filename))
            s = image.shape
            cam_info[filename] = {'K' : [s[1] / math.tan(fovx / 2),                         0, s[1] / float(2), 
                                                                 0, s[0] / math.tan(fovy / 2), s[0] / float(2), 
                                                                 0,                         0,               1]}

        json.dump(cam_info, json_info)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script creates a camera info file with approximate parameters.')
    parser.add_argument("--images_path",
                        required=True,
                        help="The path to the images.")
    parser.add_argument("--image_extension",
                        required=True,
                        help="The extension of the images.")
    parser.add_argument("--fovx",
                        required=True,
                        type=float,
                        help="The horizontal viewing angle.")
    parser.add_argument("--fovy",
                        required=True,
                        type=float,
                        help="The vertical viewing angle.")
    parser.add_argument("--output_path",
                        required=True,
                        help="The path where to output the result to.")

    args = parser.parse_args()
    convert_tless_gt(args.images_path, args.image_extension, args.fovx, args.fovy, args.output_path)