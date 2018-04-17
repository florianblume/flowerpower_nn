# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A script to render 3D object models into the training images. The models are
# rendered at the 6D poses that are associated with the training images.
# The visualizations are saved into the folder specified by "output_path".

from . import inout, renderer, misc
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def render_images(data_path, output_path, obj_ids, im_step, mode=renderer.RENDERING_MODES, draw_image=False):
    device = 'canon' # options: 'primesense', 'kinect', 'canon'
    model_type = 'cad' # options: 'cad', 'reconst'

    # Paths to the elements of the T-LESS dataset
    model_path_mask = os.path.join(data_path, 'models_' + model_type, 'obj_{:02d}.ply')
    obj_info_path_mask = os.path.join(data_path, 'train_{}', '{:02d}', 'info.yml')
    obj_gt_path_mask = os.path.join(data_path, 'train_{}', '{:02d}', 'gt.yml')
    rgb_path_mask = os.path.join(data_path, 'train_{}', '{:02d}', 'rgb', '{:04d}.{}')
    depth_path_mask = os.path.join(data_path, 'train_{}', '{:02d}', 'depth', '{:04d}.png')
    rgb_ext = {'primesense': 'png', 'kinect': 'png', 'canon': 'jpg'}
    obj_colors_path = os.path.join(data_path, 'models_' + model_type, 'obj_rgb.txt')
    vis_rgb_path_mask = os.path.join(output_path, '{:02d}_{}_{}_{:04d}_{}.tiff')
    vis_rgb_binary_path_mask = os.path.join(output_path, '{:02d}_{}_{}_{:04d}_{}_binary.txt')
    vis_depth_path_mask = os.path.join(output_path, '{:02d}_{}_{}_{:04d}_depth_diff.png')

    misc.ensure_dir(output_path)
    obj_colors = inout.load_colors(obj_colors_path)

    plt.ioff() # Turn interactive plotting off

    for obj_id in obj_ids:

        # Load object model
        model_path = model_path_mask.format(obj_id)
        model = inout.load_ply(model_path)

        # Load info about the templates (including camera parameters etc.)
        obj_info_path = obj_info_path_mask.format(device, obj_id)
        obj_info = inout.load_info(obj_info_path)

        obj_gt_path = obj_gt_path_mask.format(device, obj_id)
        obj_gt = inout.load_gt(obj_gt_path)

        for im_id in obj_info.keys():
            if im_id % im_step != 0:
                continue
            print('obj: ' + str(obj_id) + ', device: ' + device + ', im_id: ' + str(im_id))

            im_info = obj_info[im_id]
            im_gt = obj_gt[im_id]

            # Get intrinsic camera parameters and object pose
            K = im_info['cam_K']
            R = im_gt[0]['cam_R_m2c']
            t = im_gt[0]['cam_t_m2c']

            # Visualization #1
            #-----------------------------------------------------------------------
            # Load RGB image
            rgb_path = rgb_path_mask.format(device, obj_id, im_id, rgb_ext[device])
            rgb = cv2.imread(rgb_path)

            # Render RGB image of the object model at the pose associated with
            # the training image into a
            # surf_color = obj_colors[obj_id]
            surf_color = (1, 0, 0)
            im_size = (rgb.shape[1], rgb.shape[0])
            rendered = renderer.render(model, im_size, K, R, t,
                                      surf_color=surf_color, mode=mode)
            for rendering_mode in mode:
                image = rendered[rendering_mode]
                vis_rgb = image

                ############# We don't need this
                # Draw the bounding box of the object
                # vis_rgb = misc.draw_rect(vis_rgb, im_gt[0]['obj_bb'])

                # Save the visualization
                #vis_rgb[vis_rgb > 255] = 255
                vis_rgb_path = vis_rgb_path_mask.format(obj_id, device, model_type, im_id, rendering_mode)
                print(vis_rgb[vis_rgb > 0])
                fs_write = cv2.FileStorage(os.path.join(output_path, '{}.yml'.format(im_id)), cv2.FILE_STORAGE_WRITE)
                fs_write.write("floatdata", vis_rgb)
                fs_write.release()

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Render object coordinates of objects at groundtruth poses .')
    parser.add_argument("--data_path",
                        required=True,
                        help="The path to the T-Less dataset.")
    parser.add_argument("--output_path",
                        required=True,
                        help="The path to the output folder.")
    parser.add_argument("--obj_ids",
                        required=True,
                        nargs='+',
                        type=int,
                        help="The IDs of the objects to render. Specify all IDs that you need, .e.g 1 2 or 1 3 4.")
    parser.add_argument("--im_step",
                        required=True,
                        type=int,
                        help="Every im_step-th image will be rendered only.")
    parser.add_argument("--mode",
                        required=True,
                        nargs='+',
                        help="Specifies the mode for the renderer. This can be 'rgb', 'depth' or 'obj_coords' or any combination.")
    parser.add_argument("--draw_image",
                        required=False,
                        action='store_true',
                        help="Indicates whether the actual image of the object is to be rendered into the background.")
    args = parser.parse_args()
    render_images(args.data_path, args.output_path, args.obj_ids, args.im_step, args.mode, args.draw_image)