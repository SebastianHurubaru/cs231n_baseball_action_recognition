import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout


def create_model(model_name):

    if model_name == 'baseline':
        return BaselineModel


class BaselineModel(Model):

    def __init__(self, args, **kwargs):
        super(BaselineModel, self).__init__(kwargs)

        self.args = args

        self.inception = InceptionV3(weights=None,
                                     input_shape=(self.args.frame_width,
                                                  self.args.frame_height,
                                                  self.args.frame_channels),
                                     include_top=False)

        self.inception_output_size = 2048

        self.rnn = LSTM(self.args.hidden_size)

        self.out = Dense(2,
                         input_shape=(self.args.hidden_size,),
                         activation='softmax')

        self.dropout = Dropout(self.args.drop_prob)

    def call(self, x):

        # x.shape =  (args.batch_size, args.frames_per_second, args.width, args.height, args.channels)

        x = tf.reshape(x, [-1, self.args.frame_height, self.args.frame_width, self.args.frame_channels])
        # x.shape =  (args.batch_size * args.frames_per_second, args.width, args.height, args.channels)

        inception_out = self.inception(x)
        # inception_out.shape =  (args.batch_size * args.frames_per_second, 1, 1, inception_output_size)

        inception_out = tf.reshape(inception_out, [-1, self.args.frames_per_second, self.inception_output_size])
        # inception_out.shape = (args.batch_size, args.frames_per_second, inception_output_size)

        inception_out = self.dropout(inception_out)

        rnn_out = self.rnn(inception_out)
        # rnn_out.shape = (args.batch_size, args.hidden_size)

        rnn_out = self.dropout(rnn_out)

        out = self.out(rnn_out)
        # out.shape = (args.batch_size, 2)

        return out