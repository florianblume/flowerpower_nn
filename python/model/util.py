import scipy.misc
import numpy as np

def resize_image(image, shape):
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
    top_pad = (shape[0] - h) // 2
    bottom_pad = shape[0] - h - top_pad
    left_pad = (shape[1] - w) // 2
    right_pad = shape[1] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image, scale, padding