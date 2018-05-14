# Configuration of the neural network

import config.Config

class TrainingConfig(config.Config):

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # The number of epochs to run the training for
    EPOCHS = 300

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Parameters for TensorBoard
    HISTOGRAM_FREQ = 0

    # All configurations specific to paths etc

    # The path to the images
    IMAGES_PATH = ""

    # The extension of the images
    IMAGE_EXTENSION = "jpg"

    # The object model to train for, i.e. which is used to render groundtruth
    # object coordinates, if specified to re-render them
    OBJECT_MODEL_PATH = ""   

    OBJECT_MODEL_COLOR = [255, 255, 255] 

    # The path to the weights to use for inference
    WEIGHTS_PATH = ""

    # All layers that are to be excluded from weight loading
    LAYERS_TO_EXCLUDE_FROM_WEIGHT_LOADING = []

    # The layers to train
    LAYERS_TO_TRAIN = []

    # The path to the camera info file related to the images
    CAM_INFO_PATH = ""

    # The path to the ground-truth file that is to be used to render ground-truth
    # object coordinate images before training
    GT_PATH = ""

    # The ration of training images to validation images
    TRAIN_VAL_RATIO = 0.7

    # The path where the network can create necessary data
    # This is not the log folder where the network will store
    # its weights, etc, but only the rendered images etc.
    # If data is present there and it is not requested to re-
    # generate data through the REGENERATE_DATA option, the
    # data will be loaded from there.
    DATA_PATH = ""

    # Indicates whether the rendered data is to be updated
    # before training
    REGENERATE_DATA = False

    # The path where the network can create necessary data
    # This is not the log folder where the network will store
    # its weights, etc, but only the rendered images etc
    OUTPUT_PATH = ""

    # Indicates whether images should be written to Tensorboard logs
    WRITE_IMAGES = False