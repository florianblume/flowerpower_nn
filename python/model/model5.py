from .model import FlowerPowerCNN as FP

import tensorflow as tf
from tensorflow.python.client import timeline
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE

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

class FlowerPowerCNN(FP):

    def identity_block(self, input_tensor, kernel_size, filters, stage, block,
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


    def conv_block(self, input_tensor, kernel_size, filters, stage, block,
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
        #shortcut = BatchNorm(axis=3, name=bn_name_base + '1', trainable=batch_norm_trainable)(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
        return x


    def resnet_graph(self, input_image, architecture, batch_norm_trainable=True):
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
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), 
                                            batch_norm_trainable=batch_norm_trainable)
        # Layer 6 - 8 - output: 248, receptive 15
        x = self.identity_block(x, 1, [64, 64, 256], stage=2, block='b', 
                                            batch_norm_trainable=batch_norm_trainable)
        ##############
        # Layer 9 - 11 - output: 248, receptive 19
        C2 = x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c', 
                                            batch_norm_trainable=batch_norm_trainable)
        # Stage 3
        # Layer 12 - 14 - output: 124, receptive 19
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a', 
                                            batch_norm_trainable=batch_norm_trainable)

        # IMPORTANT: From here different Kernel sizes from the original ResNet, to
        #            keep the receptive field size lower.

        # Layer 15 - 17 - output: 124, receptive 27
        x = self.identity_block(x, 1, [128, 128, 512], stage=3, block='b', 
                                            batch_norm_trainable=batch_norm_trainable)
        # Layer 18 - 20 - output: 124, receptive 27
        x = self.identity_block(x, 1, [128, 128, 512], stage=3, block='c', 
                                            batch_norm_trainable=batch_norm_trainable)
        # Layer 21 - 23 - output: 124, receptive 27
        C3 = x = self.identity_block(x, 1, [128, 128, 512], stage=3, block='d', 
                                            batch_norm_trainable=batch_norm_trainable)
        # Stage 4
        # Layer 24 - 26 - output: 62, receptive 43
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2, 2),
                                            batch_norm_trainable=batch_norm_trainable)

        block_count = {"resnet35" : 0, "resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            # Layer 27 - 41 - output: 62, receptive 43
            x = self.identity_block(x, 1, [256, 256, 1024], stage=4, block=chr(98 + i), 
                                            batch_norm_trainable=batch_norm_trainable)

        C4 = x

        # Stage 5

        # The followin parameters are only for ResNet35
        # Layer 42 - 44 - output: 62, receptive 59
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1),
                                        batch_norm_trainable=batch_norm_trainable)
        # Layer 45 - 47 - output: 62, receptive 59
        x = self.identity_block(x, 1, [512, 512, 2048], stage=5, block='b', 
                                        batch_norm_trainable=batch_norm_trainable)
        # Layer 48 - 50- output: 62, receptive 59
        C5 = x = self.identity_block(x, 1, [512, 512, 2048], stage=5, block='c', 
                                        batch_norm_trainable=batch_norm_trainable)

        return C5

    def detection_head_graph(self, feature_map, filters):
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

    def construct_detection_graph(self, input_image):
        batch_norm_trainable = True
        if self.mode == "training":
            batch_norm_trainable = self.config.BATCH_NORM_TRAINABLE

        resnet = self.resnet_graph(input_image, 
                                        "resnet50",
                                        batch_norm_trainable=batch_norm_trainable)

        return self.detection_head_graph(resnet, 2048)