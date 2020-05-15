"""
Training script

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""
import tensorflow as tf
import tensorflow_datasets as tfds

import datetime

from args import get_train_args
from data_pipeline import VideoDataset

from model import create_model

from util import *

if __name__ == '__main__':

    args = get_train_args()

    # Setup the output dir
    run_dir = args.save_dir + '_' + args.name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=1)

    # Setup the data
    builder = VideoDataset(args=args)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            download_mode=tfds.core.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
            manual_dir=args.input_dir
        )
    )
    ds_train = builder.as_dataset(as_supervised=True)['train'].map(builder._normalize_frames)

    # Create the model
    model = create_model(args.model)(args=args, dynamic=True)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(ds_train,
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              callbacks=[tensorboard_callback])

    tf.saved_model.save(model, run_dir)

