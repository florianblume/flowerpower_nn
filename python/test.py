import tifffile as tiff
import matplotlib.image
import util.util as util
import os

images_path = "/home/florian/git/flowerpower_nn/assets/tless/train_canon/01/rgb"
data_path = "/home/florian/git/flowerpower_nn/assets/tless/train_canon/01/generated_data"
output_path = "/home/florian/git/flowerpower_nn/assets/tless/train_canon/01/generated_data_cropped"

images = util.get_files_at_path_of_extensions(images_path, "jpg")
segmentation_images = util.get_files_at_path_of_extensions(data_path, "png")
obj_coord_images = util.get_files_at_path_of_extensions(data_path, "tiff")

util.sort_list_by_num_in_string_entries(images)
util.sort_list_by_num_in_string_entries(segmentation_images)
util.sort_list_by_num_in_string_entries(obj_coord_images)

for index, image in enumerate(images):
    segmentation_image = segmentation_images[index]
    obj_coord_image = obj_coord_images[index]

    loaded_image = matplotlib.image.imread(os.path.join(images_path, image))
    loaded_segmentation_image = matplotlib.image.imread(os.path.join(data_path, segmentation_image))
    loaded_obj_coord_image = tiff.imread(os.path.join(data_path, obj_coord_image))

    print("Cropping image {}".format(image))
    image_croped = util.crop_image_on_segmentation_color(loaded_image, loaded_segmentation_image, [1, 1, 1])
    print("Cropping image {}".format(segmentation_image))
    segmentation_image_cropped = util.crop_image_on_segmentation_color(loaded_segmentation_image, loaded_segmentation_image, [1, 1, 1])

    matplotlib.image.imsave(os.path.join(output_path, image), image_croped)
    matplotlib.image.imsave(os.path.join(output_path, segmentation_image), segmentation_image_cropped)
    tiff.imsave(os.path.join(output_path, obj_coord_image), loaded_obj_coord_image)
