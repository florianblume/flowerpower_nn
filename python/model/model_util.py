import scipy.misc
import numpy as np

def create_index_array_from_step_sizes(step_y, end_y, max_y, step_x, end_x, max_x):

    steps_y = np.arange(step_y / 2, end_y, step_y)
    steps_x = np.arange(step_x / 2, end_x, step_x)

    steps_y = steps_y.astype(np.int32)
    steps_x = steps_x.astype(np.int32)

    # Numpy sometimes INCLUDES the limit in np.arange for floating point values, i.e.
    # we have to check whether we exceed the image's limits
    steps_y_last = len(steps_y) - 1
    steps_y[steps_y_last] = min(steps_y[steps_y_last], max_y)
    steps_x_last = len(steps_x) - 1
    steps_x[steps_x_last] = min(steps_x[steps_x_last], max_x)

    return steps_y, steps_x


def shrink_image_with_step_size(image, target_shape):
    """ With this function an image can be shrinked to a certain target size
    with out interpolation. Instead the pixels are extracted at regular intervals
    determined by the ratio of source and target shape.
    """
    step_y = image.shape[0] / float(target_shape[0])
    step_x = image.shape[1] / float(target_shape[1])

    steps_y, steps_x = create_index_array_from_step_sizes(step_y, 
                                                          image.shape[0],
                                                          image.shape[0] - 1,
                                                          step_x, 
                                                          image.shape[1],
                                                          image.shape[1] - 1)

    num_steps_y = len(steps_y)

    # Repeat items to be able to pair every entry in steps_y with every entry in steps_x
    steps_y = np.repeat(steps_y, len(steps_x))
    steps_x = np.tile(steps_x, num_steps_y)

    shrinked_image = np.zeros([target_shape[0], target_shape[1], 3], dtype=image.dtype)
    shrinked_image = image[steps_y, steps_x]
    shrinked_image = shrinked_image.reshape(target_shape)

    return shrinked_image

def resize_and_pad_image(image, shape):
    """
    Resizes an image keeping the aspect ratio and fills the remainder of the
    shape with zeros.
    shape: the dimension to rescale the image to
    Returns:
    image: the resized image
    scale: The scale factor used to resize the image
    padding: The padding that was used to convert the image to the requested shape
    """
    h, w = image.shape[:2]
    max_dim = max(shape[0], shape[1])
    scale = 1
    image_max = max(h, w)
    scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        if image.dtype == np.float32:
            # Scipy can't handle 3-channel float images, but OpenCV hangs during training\
            # and Skimage needs floats to be within [0, 1]
            channel_0 = scipy.misc.imresize(image[:,:,0], (int(scale * h), int(scale * w)), mode="F")
            channel_1 = scipy.misc.imresize(image[:,:,1], (int(scale * h), int(scale * w)), mode="F")
            channel_2 = scipy.misc.imresize(image[:,:,2], (int(scale * h), int(scale * w)), mode="F")
            image = np.stack([channel_0, channel_1, channel_2], axis=2)
        else:
            image = scipy.misc.imresize(image, (int(scale * h), int(scale * w)))

    # Get new height and width
    h, w = image.shape[:2]
    # Images are only padded to bottom and right
    top_pad = 0
    bottom_pad = shape[0] - h
    left_pad = 0
    right_pad = shape[1] - w
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image, scale, (padding[0][1], padding[1][1])

def pad_image(image, shape):
    h, w = image.shape[:2]
    # Images are only padded to bottom and right
    top_pad = 0
    bottom_pad = shape[0] - h
    left_pad = 0
    right_pad = shape[1] - w
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image, (padding[0][1], padding[1][1])