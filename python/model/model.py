import os
import numpy as np
import datetime
import re
import cv2

import logging
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

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

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
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
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
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
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet35", "resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    ##############
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet35" : 0, "resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
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

def loss_graph(target_obj_coords, segmentation_image, pred_obj_coords, color):
    """ Loss for the network.

    target_obj_coords: [batch, height, width, 3]. The ground-truth object coordinates.
    pred_obj_coords: [batch, height, width, 3]. The predicted object coordinates.
    """

    # The mask that takes care of downsampling, i.e. that only every i-th pixel is computed
    # in case that the output image is smaller than the input image
    original_image_shape = tf.shape(target_obj_coords)
    conv_image_shape = tf.shape(pred_obj_coords)
    step_y = tf.cast((original_image_shape[1] / conv_image_shape[1]) / 2, tf.int32)
    step_x = tf.cast((original_image_shape[2] / conv_image_shape[2]) / 2, tf.int32)

    # For each batch we gather the same amount of entries
    target_obj_coords = target_obj_coords[:,::step_y,::step_x,:]
    # Reshape to [batch,height of predicted obj coords,width of predicted obj coords,colors]
    target_obj_coords = tf.resize_nearest_neighbor(target_obj_coords, [conv_image_shape[1], conv_image_shape[2]])
    tf.Print(target_obj_coords, [target_obj_coords], "Object coordinates")

    # The segmentation mask where the object is actually visible
    segmentation_mask = tf.equal(segmentation_image, color)
    segmentation_mask = tf.cast(tf.reduce_all(segmentation_mask, axis=3), tf.float32)
    segmentation_mask = segmentation_mask[:,::step_y,::step_x,:]
    segmentation_mask = tf.resize_nearest_neighbor(segmentation_mask, [conv_image_shape[1], conv_image_shape[2]])
    segmentation_mask_indices = segmentation_mask > 0
    tf.Print(segmentation_mask_indices, [segmentation_mask_indices], "Segmentation mask")

    # Three channels in the color images

    squared_diff = tf.abs(target_obj_coords - pred_obj_coords)
    less_than_one = tf.cast(tf.less(squared_diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * squared_diff**2) + (1 - less_than_one) * (squared_diff - 0.5)
    loss = tf.reduce_sum(loss, axis=3)
    loss = loss * segmentation_mask
    loss = loss * index_mask
    loss = tf.reduce_sum(loss, axis=2)
    loss = tf.reduce_sum(loss, axis=1)
    return loss

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
            image = dataset.load_image(image_id)
            segmentation_image = dataset.load_segmentation_image(image_id)
            obj_coord_image = dataset.load_obj_coord_image(image_id)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype=image.dtype)
                batch_segmentation_images = np.zeros(
                    (batch_size, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype=segmentation_image.dtype)
                batch_obj_coord_images = np.zeros(
                    (batch_size, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype=obj_coord_image.dtype)

            new_size = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            segmentation_image = cv2.resize(segmentation_image, new_size, interpolation=cv2.INTER_AREA)
            obj_coord_image = cv2.resize(obj_coord_image, new_size, interpolation=cv2.INTER_AREA)

            batch_images[b] = image
            batch_segmentation_images[b] = segmentation_image
            batch_obj_coord_images[b] = obj_coord_image

            b += 1

            # Batch full?
            if b >= batch_size:
                yield [batch_images, batch_segmentation_images, batch_obj_coord_images], []

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
        input_segmentation_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_segmentation_image", dtype=tf.float32)

        # The color of the object model in the segmentation image
        color = K.constant(config.OBJECT_MODEL_COLOR, dtype=tf.float32)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        C1, C2, C3, C4, C5 = resnet_graph(input_image, "resnet35", stage5=False)

        # We only use C3, as the deeper layers result in a receptive field that is too large
        """
        P3 = KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
        P2 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='resnet_c2p2')(C2)])
        P1 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p2upsampled")(P2),
            KL.Conv2D(256, (1, 1), name='fpn_c1p1')(C1)])
        """

        C3 = KL.Conv2D(256, (3, 3), padding="same", name="resnet_p1")(C3)
        # Rescale to a divisor of the original image size
        #feature_map = KL.Lambda(lambda x : tf.resize_nearest_neighbor(x, [50, 50], align_corners=True))(C3)

        obj_coord_image = detection_head_graph(C3, 1024)

        if mode == "training":

            # Groundtruth object coordinates
            input_obj_coord_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_obj_coord_image", dtype=tf.float32)

            # Losses
            # Color cannot be passed directly as it is a constant
            loss = KL.Lambda(lambda x, color: loss_graph(*x, color), name="coord_loss", arguments={'color': color})(
                [input_obj_coord_image, input_segmentation_image, obj_coord_image])

            # Model
            inputs = [input_image, input_segmentation_image, input_obj_coord_image]

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
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
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
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, config):
        
        learning_rate = config.LEARNING_RATE
        epochs = config.EPOCHS
        layers = config.LAYERS_TO_TRAIN

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
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=config.HISTOGRAM_FREQ, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
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

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
        )
        self.epoch = max(self.epoch, epochs)


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

    def detect(self, images, segmentation_images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        obj_coords: [N, (y1, x1, y2, x2)] the predicted object coordinate images
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        assert len(images) == len(segmentation_images)

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                print(type(image))
                log("image", image)

        zipped = zip(images, segmentation_images)

        # Run object coordinate prediction
        obj_coords = self.keras_model.predict([zipped], verbose=0)

        # TODO: Rescale according to the image size in the config

        # Process detections
        results = []
        for i, image in enumerate(images):
            results.append({
                "obj_coords": obj_coords[i],
            })
        return results

    def unmold_detection(pred_obj_coords, segmentation_image):
        #Depending on the size of the predicted object coordinates we need to either
        #  * set only every i-th pixel
        #  * resize the image to its original ratio
        pass