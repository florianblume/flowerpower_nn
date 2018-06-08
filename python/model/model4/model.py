import os
import numpy as np
import datetime
import re
import cv2
import tifffile as tiff
from random import randint

import logging
import tensorflow as tf
from tensorflow.python.client import timeline
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

from .. import model_util

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Custom callback to visualize predictions after each epoch
############################################################

class VisualizePredictionCallback(keras.callbacks.Callback):

    output_path = ""

    dataset = None

    model = None

    shape = None

    images = None

    def __init__(self, model, output_path, dataset, prediction_examples, shape):
        self.output_path = output_path
        self.dataset = dataset
        self.model = model
        self.shape = shape
        self.prediction_examples = prediction_examples

    def on_epoch_end(self, epoch, logs={}):
        output_path = os.path.join(self.output_path, "example_predictions", "epoch_{}".format(epoch))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(
            "Performing sample prediction with model of epoch {}. Results stored at {}.".\
                    format(epoch, output_path))

        for example in self.prediction_examples:

            image = example['image']
            segmentation = example['segmentation']

            image_name = os.path.splitext(os.path.basename(image))[0]

            file_path = os.path.join(output_path, "epoch_{}_image_{}.tiff".format(epoch, image_name))

            loaded_image = cv2.imread(image)
            scaled_padded_image, scale, image_padding = model_util.resize_and_pad_image(
                                                                        loaded_image, 
                                                                        self.shape)
            segmentation_image = cv2.imread(segmentation)
            padded_segmentation_image, seg_and_coord_padding = \
                                        model_util.pad_image(segmentation_image, self.shape)

            # Passing segmentation image as placeholder for the gt object coord image
            # - it is not actually needed for prediction but the model expects the third image
            # as it is build for training

            """
            prediction, loss = self.model.predict([np.array([scaled_padded_image]), 
                                                   np.array([image_padding]),
                                                   np.array([padded_segmentation_image]), 
                                                   np.array([padded_segmentation_image]),
                                                   np.array([seg_and_coord_padding])])
            tiff.imsave(file_path, prediction.astype(np.float16))
            """


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def __init__(self, axis, name, trainable=True):
        super().__init__(axis=axis, name=name)
        self.trainable = trainable

    def call(self, inputs, training=True):
        return super(self.__class__, self).call(inputs, training=self.trainable)

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, batch_norm_trainable=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a', trainable=batch_norm_trainable)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b', trainable=batch_norm_trainable)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c', trainable=batch_norm_trainable)(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, batch_norm_trainable=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a', trainable=batch_norm_trainable)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b', trainable=batch_norm_trainable)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c', trainable=batch_norm_trainable)(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1', trainable=batch_norm_trainable)(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, batch_norm_trainable=True):
    assert architecture in ["resnet35", "resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)

    # All output sizes and receptive field sizes are for 500 as image dim

    # Layer 1 - output: 250, receptive 7
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x) 

    x = BatchNorm(axis=3, name='bn_conv1', trainable=batch_norm_trainable)(x)
    x = KL.Activation('relu')(x)
    # Layer 2 - output: 248, receptive 11
    C1 = x = KL.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    # Stage 2
    # Layer 3 - 5 - output: 248, receptive 15
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), 
                                        batch_norm_trainable=batch_norm_trainable)
    # Layer 6 - 8 - output: 248, receptive 15
    x = identity_block(x, 1, [64, 64, 256], stage=2, block='b', 
                                        batch_norm_trainable=batch_norm_trainable)
    ##############
    # Layer 9 - 11 - output: 248, receptive 19
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', 
                                        batch_norm_trainable=batch_norm_trainable)
    # Stage 3
    # Layer 12 - 14 - output: 124, receptive 19
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', 
                                        batch_norm_trainable=batch_norm_trainable)

    # IMPORTANT: From here different Kernel sizes from the original ResNet, to
    #            keep the receptive field size lower.

    # Layer 15 - 17 - output: 124, receptive 27
    x = identity_block(x, 1, [128, 128, 512], stage=3, block='b', 
                                        batch_norm_trainable=batch_norm_trainable)
    # Layer 18 - 20 - output: 124, receptive 27
    x = identity_block(x, 1, [128, 128, 512], stage=3, block='c', 
                                        batch_norm_trainable=batch_norm_trainable)
    # Layer 21 - 23 - output: 124, receptive 27
    C3 = x = identity_block(x, 1, [128, 128, 512], stage=3, block='d', 
                                        batch_norm_trainable=batch_norm_trainable)
    # Stage 4
    # Layer 24 - 26 - output: 62, receptive 43
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2, 2),
                                        batch_norm_trainable=batch_norm_trainable)

    block_count = {"resnet35" : 0, "resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        # Layer 27 - 41 - output: 62, receptive 43
        x = identity_block(x, 1, [256, 256, 1024], stage=4, block=chr(98 + i), 
                                        batch_norm_trainable=batch_norm_trainable)
    C4 = x

    # Stage 5

    # The followin parameters are only for ResNet35
    # Layer 42 - 44 - output: 62, receptive 59
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1),
                                    batch_norm_trainable=batch_norm_trainable)
    # Layer 45 - 47 - output: 62, receptive 59
    x = identity_block(x, 1, [512, 512, 2048], stage=5, block='b', 
                                    batch_norm_trainable=batch_norm_trainable)
    # Layer 48 - 50- output: 62, receptive 59
    C5 = x = identity_block(x, 1, [512, 512, 2048], stage=5, block='c', 
                                    batch_norm_trainable=batch_norm_trainable)

    return [C1, C2, C3, C4, C5]

def detection_head_graph(feature_map, filters):
    """Builds the computation graph of the final stage of the network that produces
    the object coordinate predictions.

    Inputs:
        feature_map: backbone features [batch, height, width, depth]
        filters: the size of the output channels

    Returns:
        prediction object coordinates: [batch, height, width, 3]
    """
    x = KL.Conv2D(filters, (1, 1), strides=(1, 1),
                  name="detection_head_" + "stage_1", use_bias=True, padding="same")(feature_map)
    x = KL.Activation('relu', name='detection_head_stage_1_activation')(x)
    x = KL.Conv2D(filters, (1, 1), strides=(1, 1),
                  name="detection_head_" + "stage_2", use_bias=True, padding="same")(x)
    x = KL.Activation('relu', name='detection_head_stage_2_activation')(x)
    x = KL.Conv2D(3, (1, 1), strides=(1, 1),
                  name="detection_head_" + "final_stage", use_bias=True, padding="same")(x)
    return x

def compute_indices_graph(original_shape, target_shape):
    """ Computes the indices.

    input_shapes:  [batch, (height, width)]. The un-padded shapes of the input images.
    outupt_shapes: [batch, (height, width)]. The un-padded shapes of the output images.
    """

    # If we use an int step size, the offset error distributes when slicing resulting
    # in a smaller image than the pred_obj_coords
    # That's why we calculate the floating point range and then take ints individually
    step_y = tf.cast(original_shape[0] / target_shape[0], tf.float32)
    step_x = tf.cast(original_shape[1] / target_shape[1], tf.float32)
    start_y = tf.cast(step_y / 2, tf.float32)
    start_x = tf.cast(step_x / 2, tf.float32)

    indices_y = tf.range(start_y, original_shape[0], step_y)
    indices_y_shape = tf.shape(indices_y)
    indices_x = tf.range(start_x, original_shape[1], step_x)
    indices_y = tf.reshape(indices_y, [-1, 1])
    indices_y = tf.tile(indices_y, [1, tf.shape(indices_x)[0]])
    indices_x = tf.reshape(indices_x, [1, -1])
    indices_x = tf.tile(indices_x, [tf.shape(indices_y)[0], 1])
    # Now we have two matrices of equal shape that we can combine to obtain the final index pairs
    indices = tf.stack([indices_y, indices_x], axis=2)
    # Now retrieve the actual indices which reduce the rounding error compared to calculating
    # the step size, e.g. 7.9, and striding this step size along the image
    indices = tf.cast(tf.round(indices), tf.int32)
    return indices

def single_loss_graph(pred_obj_coord_image, image_padding, segmentation_image, 
               target_obj_coord_image, seg_and_coord_padding, color):
    """ Loss for one batch entry.

    pred_obj_coords: [height, width, 3]. The predicted object coordinates.
    image_padding: [(bottom, right)]. The widths of the paddings of the original image.
    segmentation_image: height, width, 3]. The segmentation image.
    target_obj_coords: [height, width, 3]. The ground-truth object coordinates.
    seg_and_coord_padding: [(bottom, right)]. The widths of the paddings of the object
                            coord image and segmentation image. It differs from the paddings of
                            the original input image because both are not reszied.
    color: [r, g, b]. The object's color in the segmentation image.
    """

    # The mask that takes care of downsampling, i.e. that only every i-th pixel is computed
    # in case that the output image is smaller than the input image

    input_shape = tf.shape(target_obj_coord_image)
    batch_size = input_shape[0]
    # The shape of the image that the network output
    output_shape = tf.shape(pred_obj_coord_image)
    # With the downsampling ratio we know how much smaller the padding is in the output image
    ratio_y = output_shape[0] / input_shape[0]
    ratio_x = output_shape[1] / input_shape[1]
    # We multiply top and bottom by the y ratio
    image_padding = tf.cast(image_padding, tf.float64)
    image_padding_y = image_padding[0] * ratio_y
    image_padding_x = image_padding[1] * ratio_x
    image_padding_y = tf.cast(image_padding_y, tf.int32)
    image_padding_x = tf.cast(image_padding_x, tf.int32)
    # Now we have the actual predicted area for each batch image individually
    # [:2] because we only need (height, width) without the channels
    actual_output_shape = output_shape[:2] - (image_padding_y, image_padding_x)
    acutal_output_shape_with_channels = [actual_output_shape[0], actual_output_shape[1], 3]
    # Crop the output prediction to the actual area without padding
    cropped_pred_obj_coord_image = pred_obj_coord_image[:actual_output_shape[0],
                                                        :actual_output_shape[1],
                                                        :]
    # Now we compute the original shape of the ground truth and segmentation
    actual_input_shape = input_shape[:2] - tf.cast(seg_and_coord_padding, tf.int32)
    # And also crop the segmentation and prediction to the actual area
    cropped_segmentation_image = segmentation_image[:actual_input_shape[0],
                                                    :actual_input_shape[1],
                                                    :]
    cropped_target_obj_coord_image = target_obj_coord_image[:actual_input_shape[0],
                                                            :actual_input_shape[1],
                                                            :]

    # Here we compute the relevant indices, e.g. if the output image is 1/8th of the original
    # input image, we use only every 8-th pixel horizontally and vertically. The code takes
    # care of the output image's size not being a divisor of the input image's.
    indices = compute_indices_graph(tf.shape(cropped_segmentation_image)[:2], 
                                             tf.shape(cropped_pred_obj_coord_image)[:2])
    #indices = tf.Print(indices, [indices], "indices", summarize=(63 * 63))

    # Now create and object coord and segmentation image of the size of the output shape
    # that contains only the relevent indices
    final_target_obj_coord_image = tf.gather_nd(cropped_target_obj_coord_image, 
                                                indices)
    final_target_obj_coord_image = tf.reshape(final_target_obj_coord_image, 
                                              acutal_output_shape_with_channels)
    final_segmentation_image = tf.gather_nd(cropped_segmentation_image, indices)
    final_segmentation_image = tf.reshape(final_segmentation_image,
                                          acutal_output_shape_with_channels)

    segmentation_mask = tf.equal(final_segmentation_image, color)
    # We have a matrix of bool values of which indices to use after this step
    segmentation_mask = tf.reduce_all(segmentation_mask, axis=2)
    segmentation_mask = tf.cast(segmentation_mask, tf.bool)

    # L1 loss: sum of squared element-wise differences
    squared_diff = tf.square(final_target_obj_coord_image - cropped_pred_obj_coord_image)
    loss = tf.reduce_mean(squared_diff, axis=2)
    loss = tf.sqrt(loss)
    loss = tf.boolean_mask(loss, segmentation_mask)
    return tf.reduce_mean(loss)

def loss_graph(pred_obj_coord_images, image_paddings, segmentation_images, 
               target_obj_coord_images, seg_and_coord_paddings, color):
    """ Loss for the network.

    pred_obj_coords: [batch, height, width, 3]. The predicted object coordinates.
    image_padding: [batch, (bottom, right)]. The widths of the paddings of the original image.
    segmentation_image: [batch, height, width, 3]. The segmentation image.
    target_obj_coords: [batch, height, width, 3]. The ground-truth object coordinates.
    seg_and_coord_padding: [batch, (bottom, right)]. The widths of the paddings of the object
                            coord image and segmentation image. It differs from the paddings of
                            the original input image because both are not reszied.
    color: [r, g, b]. The object's color in the segmentation image.
    """
    elements = (pred_obj_coord_images, image_paddings, segmentation_images,
                target_obj_coord_images, seg_and_coord_paddings, color)
    # This computes the loss for each batch entry
    loss = tf.map_fn(lambda x: single_loss_graph(*x), elements, name="loss_elem_mapping", dtype=tf.float32)
    return tf.reduce_mean(loss)

def data_generator(dataset, config, shuffle=True, batch_size=1):
    """A generator that returns the images to detect, as well as their segmentation
    masks and, most importantly, their object coordinate ground-truths.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.get_image_ids())
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT object coordinates and segmentation for image.
            image_id = image_ids[image_index]

            image_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
            image = dataset.load_image(image_id)
            segmentation_image = dataset.load_segmentation_image(image_id)
            obj_coord_image = dataset.load_obj_coord_image(image_id)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
                batch_image_pads = np.zeros(
                    (batch_size, 2), dtype=np.uint8)
                batch_segmentation_images = np.zeros(
                    (batch_size, image_shape[0], image_shape[1], 3), dtype=np.uint8)
                batch_obj_coord_images = np.zeros(
                    (batch_size, image_shape[0], image_shape[1], 3), dtype=np.float32)
                batch_seg_and_coord_pads = np.zeros(
                    (batch_size, 2), dtype=np.uint8)

            # The utility function also returns the scale and padding which we dont' need here
            # thus we only retrieve the first return value
            image, scale, image_padding = model_util.resize_and_pad_image(image, image_shape)
            # No need to resize the segmentation images, we only resize the actual image
            # because the network apparently sometimes needs longer during gradient descent
            # if there are a lot of black pixels
            segmentation_image, seg_and_coord_padding = model_util.pad_image(segmentation_image, image_shape)
            # We don't resize the object coordinates, as that would distort the information
            # we only pad it to fit in the array
            # Same padding as for segmentation_image i.e. only store once
            obj_coord_image = model_util.pad_image(obj_coord_image, image_shape)[0]

            batch_images[b] = image
            # bottom, right padding of input images
            batch_image_pads[b] = image_padding
            batch_segmentation_images[b] = segmentation_image
            batch_obj_coord_images[b] = obj_coord_image
            batch_seg_and_coord_pads[b] = seg_and_coord_padding

            b += 1

            # Batch full?
            if b >= batch_size:
                yield [batch_images, batch_image_pads, 
                batch_segmentation_images, batch_obj_coord_images, batch_seg_and_coord_pads], []

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(image_id))
            error_count += 1
            if error_count > 5:
                raise

class FlowerPowerCNN:

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Flower Power architecture.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Inputs of unkown dimensions
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image", dtype=tf.float32)
        input_segmentation_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), 
                                            name="input_segmentation_image", 
                                            dtype=tf.float32)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        batch_norm_trainable = True
        if mode == "training":
            batch_norm_trainable = config.BATCH_NORM_TRAINABLE

        C1, C2, C3, C4, C5 = resnet_graph(input_image, 
                                          "resnet35",
                                          batch_norm_trainable=batch_norm_trainable)

        """
        P3 = KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
        P2 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='resnet_c2p2')(C2)])
        P1 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p2upsampled")(P2),
            KL.Conv2D(256, (1, 1), name='fpn_c1p1')(C1)])
        """

        # We use the layer C3 here, as with the given strides this results in a receptive field of 51
        # (see https://fomoro.com/tools/receptive-fields/ for details) which keeps the network towards
        # a patch-based approach instead of a global one
        #C4 = KL.Conv2D(256, (3, 3), padding="same", name="resnet_p1")(C3)

        obj_coord_image = detection_head_graph(C5, 2048)

        if mode == "training":

            # The color of the object model in the segmentation image
            # We need to tile the color as it gets unpacked into an empty array otherwise
            color = K.constant(np.tile(config.SEGMENTATION_COLOR, (config.BATCH_SIZE, 1)), dtype=tf.float32)
            color = KL.Input(tensor=color)

            # Groundtruth object coordinates
            input_gt_obj_coord_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), 
                                                name="input_obj_coord_image", 
                                                dtype=tf.float32)
            input_image_paddings = KL.Input(shape=[2],
                                      name="input_image_paddings",
                                      dtype=tf.int32)
            input_seg_and_coord_paddings = KL.Input(shape=[2],
                                      name="input_seg_and_coord_paddings",
                                      dtype=tf.int32)

            # Losses
            loss = KL.Lambda(lambda x: loss_graph(*x), name="coord_loss")(
                [obj_coord_image, input_image_paddings, input_segmentation_image, 
                input_gt_obj_coord_image, input_seg_and_coord_paddings, color])

            # Model
            inputs = [input_image, input_image_paddings, 
                    input_segmentation_image, input_gt_obj_coord_image, input_seg_and_coord_paddings,
                    color]

            outputs = [obj_coord_image, loss]
            model = KM.Model(inputs, outputs, name='flowerpower_cnn')
        else:
            # The graph does not need the color directly, we unmold detections after running the graph on an image
            # and only then crop the detection to the segmentation mask, etc.
            model = KM.Model([input_image, input_segmentation_image],
                             [obj_coord_image], name='flowerpower_cnn')

        return model

    def load_weights(self, filepath, by_name=False, exclude=None):
        
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.Adam()
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["coord_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[
                                 None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                                   keep_dims=True))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            # TODO: proper naming of weights
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}_{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, prediction_examples, config, verbose=0):
        
        learning_rates = config.LEARNING_RATE
        epochs = config.EPOCHS
        layers = config.LAYERS_TO_TRAIN

        assert len(learning_rates) == len(epochs) == len(layers), \
                        "Number of epochs, learning rates and layers must match."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(detection\_head\_.*)",
            # From Resnet stage 4 layers and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
            "5+": r"(res5.*)|(bn5.*)",
            # All layers
            "all": ".*"
        }

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=config.HISTOGRAM_FREQ, 
                                        write_graph=True, 
                                        write_images=config.WRITE_IMAGES),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
            VisualizePredictionCallback(self, self.log_dir, 
                                              train_dataset, 
                                              prediction_examples, 
                                              config.IMAGE_SHAPE)
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "max_queue_size": 100,
            "workers": max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": True,
        }

        if val_dataset.size() > 0:
            val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                           batch_size=self.config.BATCH_SIZE)
            fit_kwargs["validation_data"] = next(val_generator)
            fit_kwargs["validation_steps"] = self.config.VALIDATION_STEPS

        for index in range(len(learning_rates)):
            current_learning_rate = learning_rates[index]
            current_epochs = epochs[index]
            current_layers = layers[index]

            # Train
            if index == 0:
                log("\nStarting at epoch {}. LR={}\n".format(self.epoch, current_learning_rate))
                log("Checkpoint Path: {}".format(self.checkpoint_path))
            else:
                log("\nContinuing run {} at epoch {}. LR={}\n".format(index + 1, self.epoch, current_learning_rate))


            if current_layers in layer_regex.keys():
                current_layers = layer_regex[current_layers]

            self.set_trainable(current_layers)
            self.compile(current_learning_rate, self.config.LEARNING_MOMENTUM)

            if verbose > 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                
            self.keras_model.fit_generator(
                train_generator,
                initial_epoch=self.epoch,
                epochs=current_epochs,
                **fit_kwargs
            )

            if verbose > 0:
                timeline = timeline.Timeline(step_stats=run_metadata.step_stats)
                chrome_trace = tl.generate_chrome_trace_format()
                with open(os.path.join(config.OUTPUT_PATH, 'timeline.json'), 'w') as f:
                    f.write(chrome_trace)

            self.epoch = max(self.epoch, current_epochs)


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def predict(self, images, segmentation_images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        obj_coords: [N, (y1, x1, y2, x2)] the predicted object coordinate images
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        assert len(images) == len(segmentation_images), "len(images) must be equal to len(segmentation_images)"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                print(type(image))
                log("image", image)

        prepared_images = []
        _prepared_images = []
        prepared_segmentation_images = []
        _prepared_segmentation_images = []

        for index in range(len(images)):
            # We need to store the scaling and padding as well, to retrieve the
            # actual coordinates after inference
            image, image_scale, image_padding = model_util.resize_and_pad_image(
                                                            images[index], 
                                                            self.config.IMAGE_SHAPE[:2])
            prepared_images.append({"image"     : image,
                                    "original_shape" : images[index].shape[:2],
                                    "scale"     : image_scale,
                                    "padding"   : image_padding})
            _prepared_images.append(image)
            segmentation_image, segmentation_image_scale, segmentation_image_padding = \
                                model_util.resize_and_pad_image(segmentation_images[index], 
                                                                self.config.IMAGE_SHAPE[:2])
            prepared_segmentation_images.append({
                                    "image" : segmentation_image,
                                    "scale" : segmentation_image_scale,
                                    "padding" : segmentation_image_padding
                })
            _prepared_segmentation_images.append(segmentation_image)

        # Run object coordinate prediction
        predictions = self.keras_model.predict([np.array(_prepared_images), 
                                                np.array(_prepared_segmentation_images)], verbose=verbose)
        results = []
        if verbose:
            print("Processing inference results.")
        for i, prediction in enumerate(predictions):
            prepared_image = prepared_images[i]
            scaled_padded_image = prepared_image["image"]

            # The aspect ration to determine the size of the padding in the prediction
            raw_step_y = scaled_padded_image.shape[0] / float(prediction.shape[0])
            raw_step_x = scaled_padded_image.shape[1] / float(prediction.shape[1])

            # Paddings are stored as (padding_bottom, padding_right)
            v_padding, h_padding = prepared_image["padding"]

            # Calculat the size of the padding in the prediction (network downscales image, i.e.
            # padding is smaller in the output)
            v_padding = int(v_padding / raw_step_y)
            h_padding = int(h_padding / raw_step_x)

            # The padding does not store where it begins index-wise but the width of the padding
            # Of course we only need to take care of the right and bottom end, the left and top
            # padding function as offset where the actual image starts
            v_padding = prediction.shape[0] - v_padding
            h_padding = prediction.shape[1] - h_padding

            # Remove the padding that was added and rescale with the scale corresponding
            # to how the input image was resized to fill the requeste image dimensions
            cropped_prediction = prediction[:v_padding, :h_padding]
            final_shape = cropped_prediction.shape

            # Restrict the prediction to the segmentation pixels
            # Use the un-resized segmentation image, we do not need up- and then down-scaling
            # that increases the error
            resized_segmentation_image = model_util.shrink_image_with_step_size(
                                                                segmentation_images[i],
                                                                final_shape)
            segmentation_indices = resized_segmentation_image == self.config.SEGMENTATION_COLOR
            result = np.zeros(cropped_prediction.shape, dtype=np.float32)
            result[segmentation_indices] = cropped_prediction[segmentation_indices]

            original_shape = prepared_image["original_shape"]
            step_y = original_shape[0] / float(final_shape[0])
            step_x = original_shape[1] / float(final_shape[1])

            results.append({
                "obj_coords": result,
                "step_y" : step_y,
                "step_x" : step_x
            })
        return results

    def run_graph(self, images, segmentation_images, obj_coord_images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.
        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.
        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        if self.mode == "training":
            model_in = [images, segmentation_images, obj_coord_images]
        else:
            model_in = [images, segmentation_images]

        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np