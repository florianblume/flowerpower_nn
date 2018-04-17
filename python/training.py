def main(images_path): #, objects_path, ground_truth_path, color_map_path, regenerate_data):
    import os
    import shutil
    import cv2
    import numpy as np

    import util.util as util
    import renderer.render_images as render_images
    import renderer.renderer as renderer
    import renderer.inout as inout
    import tifffile as tiff

    assert os.path.exists(images_path), "The specified images path does not exist."
    #assert os.path.exists(objects_path), "The specified objects path does not exist."
    #assert os.path.exists(ground_truth_path), "The specified ground-truth path does not exist."
    #assert os.path.exists(color_map_path), "The specified color map path does not exist."

    data_path = os.path.join(images_path, "generated_data")
    info_path = os.path.join(images_path, "train_canon", "01", "info.yml")
    gt_path = os.path.join(images_path, "train_canon", "01", "gt.yml")

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    # Create the folder that is going to hold our rendered depth images as well as the cropped RGB images
    os.makedirs(data_path)
    render_images.render_images(images_path, data_path, range(1, 2), 100, 
                                [renderer.RENDERING_MODE_OBJ_COORDS], 
                                draw_image=False)

    obj_info = inout.load_info(info_path)
    obj_gt = inout.load_gt(gt_path)
    image_filenames = util.get_files_at_path_of_extensions(data_path, ['tiff'])
    for filename in image_filenames:
        image = tiff.imread(os.path.join(data_path, filename))
        print(filename)
        print(image[image > 0])


if __name__ == '__main__':
    import argparse

    """

    TODO: Add creation of tensorflow record and pass this on to the network 

    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to train the PENN network.')
    parser.add_argument("--images_path",
                        required=True,
                        help="The path to the images.")
    """
    parser.add_argument("--objects_path",
                        required=True,
                        help="The path to the 3D objects.")
    parser.add_argument("--ground_truth_path",
                        required=True,
                        help="The path to the ground-truth annotations file.")
    """
    # TODO: Add argument for map from object model filename to index
    # TODO: Add argument for list of image filenames to process
    # TODO: Add argument for path to pre-trained weights
    """
    parser.add_argument("--color_map_path",
                        required=True,
                        help="The path to the file that stores the mapping of object index to color code.")
    parser.add_argument("--log_path",
                        required=True,
                        help="The path to the folder where the logs and the resulting weights are to be stored.")
    parser.add_argument("--regenerate_data",
                        required=False,
                        action='store_true',
                        help="If set, the already produced depth map renderings and cropped images will be removed\
                        and regenerated.")
    """
    # More arguments related to training, etc. are to follow
    arguments = parser.parse_args()
    main(arguments.images_path)