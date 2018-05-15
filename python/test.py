import scipy.misc
import numpy as np
import tifffile as tiff

image = tiff.imread("./assets/tless/train_canon/01/generated_data_cropped/obj_coord_images/0900_obj_coordinates.tiff")
h, w = image.shape[:2]
scale = 1.3
print(image.shape)
test = image[:,:,0]
print(test)
print(test.shape)
channel_0 = scipy.misc.imresize(image[:,:,0], (int(scale * h), int(scale * w)), mode="F")
channel_1 = scipy.misc.imresize(image[:,:,1], (int(scale * h), int(scale * w)), mode="F")
channel_2 = scipy.misc.imresize(image[:,:,2], (int(scale * h), int(scale * w)), mode="F")
new_image = np.stack([channel_0, channel_1, channel_2], axis=2)
tiff.imsave("/home/florian/git/flowerpower_nn/assets/tless/train_canon/01/training/test/coord_2.tiff", new_image.astype(np.float16))