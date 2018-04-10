def main(images_path, objects_path, ground_truth_path, color_map_path, regenerate_data):
    import os
    import shutil

    import util.util as util
    import util.renderer.renderer_images as renderer

    assert os.path.exists(images_path), "The specified images path does not exist."
    assert os.path.exists(objects_path), "The specified objects path does not exist."
    assert os.path.exists(ground_truth_path), "The specified ground-truth path does not exist."
    assert os.path.exists(color_map_path), "The specified color map path does not exist."

    data_path = os.path.join(images_path, "generated_data")

    if regenerate_data:
        shutil.rmtree(data_path)
        # Create the folder that is going to hold our rendered depth images as well as the cropped RGB images
        os.makedirs(data_path)
        renderer.render_images(images_path, data_path, range(1, 2), 100, 'depth', draw_image=False)

    image_filenames = util.get_files_at_path_of_extension(images_path, ['png', 'bmp', 'jpg', 'jpeg'])

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to train the PENN network.')
    parser.add_argument("--images_path",
                        required=True,
                        help="The path to the images.")
    parser.add_argument("--objects_path",
                        required=True,
                        help="The path to the 3D objects.")
    parser.add_argument("--ground_truth_path",
                        required=True,
                        help="The path to the ground-truth annotations file.")
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
    # More arguments related to training, etc. are to follow
    args = parser.parse_args()
    main(*args)