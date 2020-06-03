import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout

from layers import *

def create_model(model_name):

    if model_name == 'baseline':
        return BaselineModel
    elif model_name == 'i3d':
        return I3D


class BaselineModel(Model):

    def __init__(self, args, **kwargs):
        super(BaselineModel, self).__init__(kwargs)

        self.args = args

        self.xception = Xception(weights='imagenet',
                         input_shape=(3*self.args.frame_height//5,
                                      self.args.frame_width // 2,
                                      self.args.frame_channels),
                         pooling='avg',
                         include_top=False)

        self.xception_output_size = 2048

        self.rnn = LSTM(self.args.hidden_size)

        self.out = Dense(2,
                         input_shape=(self.args.hidden_size,),
                         activation='softmax')

        self.dropout = Dropout(self.args.drop_prob)

    def call(self, x):

        # x.shape =  (batch_size, frames_per_second, width, height, channels)
        batch_size, fps, frame_height, frame_width, frame_channels = x.shape

        x = tf.reshape(x, [-1, frame_height, frame_width, frame_channels])
        # x.shape =  (batch_size * fps, frame_width, frame_height, frame_channels)

        xception_out = self.xception(x)
        # xception_out.shape =  (args.batch_size * args.frames_per_second, xception_output_size)

        xception_out = tf.reshape(xception_out, [-1, fps, self.xception_output_size])
        # xception_out.shape = (batch_size, fps, xception_output_size)

        rnn_out = self.rnn(self.dropout(xception_out))
        # rnn_out.shape = (batch_size, hidden_size)

        out = self.out(self.dropout(rnn_out))
        # out.shape = (batch_size, 2)

        return out

class I3D(Model):

    def __init__(self, args, **kwargs):
        super(I3D, self).__init__(kwargs)

        self.args = args

        self.mixed_downsampling = tf.keras.Sequential([

            # Downsampling via convolution (spatial and temporal)
            Conv3d_BN(64, 7, 7, 7, strides=(2, 2, 2), padding='same'),

            # Downsampling (spatial only)
            MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same'),
            Conv3d_BN(64, 1, 1, 1, strides=(1, 1, 1), padding='same'),
            Conv3d_BN(192, 3, 3, 3, strides=(1, 1, 1), padding='same'),

            # Downsampling (spatial only)
            MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same'),

            # Mixed 3b
            Mixed([64, 96, 128, 16, 32, 32]),

            # Mixed 3c
            Mixed([128, 128, 192, 32, 96, 64]),

            # Downsampling (spatial and temporal)
            MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same'),

            # Mixed 4b
            Mixed([192, 96, 208, 16, 48, 64]),

            # Mixed 4c
            Mixed([160, 112, 224, 24, 64, 64]),

            # Mixed 4d
            Mixed([128, 128, 256, 24, 64, 64]),

            # Mixed 4e
            Mixed([112, 144, 288, 32, 64, 64]),

            # Mixed 4f
            Mixed([256, 160, 320, 32, 128, 128]),

            # Downsampling (spatial and temporal)
            MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'),

            # Mixed 5b
            Mixed([256, 160, 320, 32, 128, 128]),

            # Mixed 5c
            Mixed([384, 192, 384, 48, 128, 128])
        ])

        self.global_average_pool = AveragePooling3D((2, 23, 40), strides=(1, 1, 1), padding='valid')

        self.out = Dense(2, activation='softmax')

    def call(self, x):

        out = self.mixed_downsampling(x)

        out = self.global_average_pool(out)

        out = self.out(tf.reduce_mean(tf.squeeze(out, [2, 3]), axis=1, keepdims=False))

        return out

