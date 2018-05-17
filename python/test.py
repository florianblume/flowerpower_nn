import cv2
import tifffile as tiff
from util import util
import os

input_images = "assets/tless/train_canon/01/rgb"
input_segmentation_images = "assets/tless/train_canon/01/generated_data/segmentation_images"
input_obj_coord_images = "assets/tless/train_canon/01/generated_data/obj_coord_images"

output_images = "assets/tless/train_canon/01/generated_data_cropped/images"
output_segmentation_images = "assets/tless/train_canon/01/generated_data_cropped/segmentation_images"
output_obj_coord_images = "assets/tless/train_canon/01/generated_data_cropped/obj_coord_images"

image_files = util.get_files_at_path_of_extensions(input_images, ["jpg"])
util.sort_list_by_num_in_string_entries(image_files) 

segmentation_image_files = util.get_files_at_path_of_extensions(input_segmentation_images, ["png"])
util.sort_list_by_num_in_string_entries(segmentation_image_files)

obj_coord_image_files = util.get_files_at_path_of_extensions(input_obj_coord_images, ["tiff"])
util.sort_list_by_num_in_string_entries(obj_coord_image_files)

for index in range(len(image_files)):
	segmentation_image_file = segmentation_image_files[index]
	image_file = image_files[index]
	obj_coord_image_file = obj_coord_image_files[index]
	segmentation_image = cv2.imread(os.path.join(input_segmentation_images, segmentation_image_file))
	segmentation_image_indices = segmentation_image > 0
	segmentation_image[segmentation_image_indices] = 255

	image = cv2.imread(os.path.join(input_images, image_file))
	#obj_coord_image = tiff.imread(os.path.join(input_obj_coord_images, obj_coord_image_file))

	cropped_image = util.crop_image_on_segmentation_color(image, segmentation_image, [255, 255, 255])
	cropped_segmentation_image = util.crop_image_on_segmentation_color(segmentation_image, segmentation_image, [255, 255, 255])
	#cropped_obj_coord_image = util.crop_image_on_segmentation_color(obj_coord_image, segmentation_image, [255, 255, 255])

	print("Cropping {}, {} and {}".format(image_file, segmentation_image_file, obj_coord_image_file))

	cv2.imwrite(os.path.join(output_images, image_file), cropped_image)
	cv2.imwrite(os.path.join(output_segmentation_images, segmentation_image_file), cropped_segmentation_image)
	#tiff.imsave(os.path.join(output_obj_coord_images, obj_coord_image_file), cropped_obj_coord_image)
