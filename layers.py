import tensorflow as tf
from tensorflow.keras.layers import *

class Conv3d_BN(Layer):

    def __init__(self,
                 filters,
                 num_frames,
                 num_row,
                 num_col,
                 padding='same',
                 strides=(1, 1, 1),
                 use_bias=False,
                 use_activation_fn=True,
                 use_bn=True,
                 **kwargs):

        super(Conv3d_BN, self).__init__(**kwargs)

        self.use_activation_fn = use_activation_fn
        self.use_bn = use_bn

        self.conv_3d = x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias)

        # Channels last
        channel_axis = 4
        self.batch_norm = BatchNormalization(axis=channel_axis, scale=False)

        self.activation = Activation('relu')

    def call(self, x):

        out = self.conv_3d(x)

        if self.use_bn:
            out = self.batch_norm(out)

        if self.use_activation_fn:
            out = self.activation(out)

        return out

class Mixed(Layer):

    def __init__(self,
                 layers_units,
                 **kwargs):
        super(Mixed, self).__init__(**kwargs)

        # Mix
        self.branch0_conv1 = Conv3d_BN(layers_units[0], 1, 1, 1, padding='same')

        self.branch1_conv1 = Conv3d_BN(layers_units[1], 1, 1, 1, padding='same')
        self.branch1_conv2 = Conv3d_BN(layers_units[2], 3, 3, 3, padding='same')

        self.branch2_conv1 = Conv3d_BN(layers_units[3], 1, 1, 1, padding='same')
        self.branch2_conv2 = Conv3d_BN(layers_units[4], 3, 3, 3, padding='same')

        self.branch3_maxpool = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')
        self.branch3_conv1 = Conv3d_BN(layers_units[5], 1, 1, 1, padding='same')


    def call(self, x):

        branch0 = self.branch0_conv1(x)

        branch1 = self.branch1_conv1(branch0)
        branch1 = self.branch1_conv2(branch1)

        branch2 = self.branch2_conv1(branch1)
        branch2 = self.branch2_conv2(branch2)

        branch3 = self.branch3_maxpool(branch2)
        branch3 = self.branch3_conv1(branch3)

        # Channels last
        channel_axis = 4
        out = tf.keras.layers.concatenate(
            [branch0, branch1, branch2, branch3],
            axis=channel_axis)

        return out