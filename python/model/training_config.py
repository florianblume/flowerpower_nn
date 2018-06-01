# Configuration of the neural network

from . import config

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
    EPOCHS = [300]

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = [0.001]
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Parameters for TensorBoard
    HISTOGRAM_FREQ = 0

    # Batch size is GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1

    IMAGES_PER_GPU = 26

    # All configurations specific to paths etc

    # The path to the images
    IMAGES_PATH = ""

    # The extension of the images
    IMAGE_EXTENSION = "jpg"

    # The object model to train for, i.e. which is used to render groundtruth
    # object coordinates, if specified to re-render them
    OBJECT_MODEL_PATH = ""   

    # We do not use color during training because we render our own segmentation
    # images using white color and the ground truth pose
    SEGMENTATION_COLOR = [255, 255, 255]

    IMAGE_DIM = 500

    # Indicates whether batch normalization layers are trainable
    BATCH_NORM_TRAINABLE = True

    # The path to the weights to use for inference
    WEIGHTS_PATH = ""

    # All layers that are to be excluded from weight loading
    LAYERS_TO_EXCLUDE_FROM_WEIGHT_LOADING = []

    # The layers to train
    LAYERS_TO_TRAIN = ['all']

    # The path to the camera info file related to the images
    CAM_INFO_PATH = ""

    # The path to the ground-truth file that is to be used to render ground-truth
    # object coordinate images before training
    GT_PATH = ""

    # The path where the network can create necessary data
    # This is not the log folder where the network will store
    # its weights, etc, but only the rendered images etc.
    # If data is present there and it is not requested to re-
    # generate data through the REGENERATE_DATA option, the
    # data will be loaded from there.
    DATA_PATH = ""

    # The file which contains the image filenames that will be used for training.
    # The images have to be present in the DATA_PATH/images folder.
    TRAIN_FILE = ""

    # Analogously, this file contains the filenames to use for validation.
    VAL_FILE = ""

    # The path where the network can create necessary data
    # This is not the log folder where the network will store
    # its weights, etc, but only the rendered images etc
    OUTPUT_PATH = ""

    # Indicates whether images should be written to Tensorboard logs
    WRITE_IMAGES = False