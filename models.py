import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout


def create_model(model_name):

    if model_name == 'baseline':
        return BaselineModel


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