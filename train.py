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


def train_step(inputs):

    frames, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(frames, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)

    return loss


def eval_step(inputs):

    frames, labels = inputs

    predictions = model(frames, training=False)
    t_loss = loss_object(labels, predictions)

    dev_loss.update_state(t_loss)
    dev_accuracy.update_state(labels, predictions)



def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


def distributed_eval_step(dataset_inputs):
    return strategy.run(eval_step, args=(dataset_inputs,))


if __name__ == '__main__':

    args = get_train_args()

    tf.debugging.set_log_device_placement(args.log_device_placement)

    # Get the data over the CPU
    with tf.device('/CPU:0'):

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
        ds_train = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True) \
            .map(builder._normalize_frames,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_dev = builder.as_dataset(split=tfds.Split.VALIDATION, as_supervised=True) \
            .map(builder._normalize_frames, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Copy the model on each GPU and use data parallelism
    # strategy = tf.distribute.MirroredStrategy()

    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")

    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    # ds_train = strategy.experimental_distribute_dataset(dataset=ds_train)
    # ds_dev = strategy.experimental_distribute_dataset(dataset=ds_dev)

    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync

    with strategy.scope():

        # Create model loss
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        # define the metrics
        dev_loss = tf.keras.metrics.Mean(name='dev_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='dev_accuracy')

    with strategy.scope():

        # Create the model, optimizer and checkpoint under 'strategy_scope'
        model = create_model(args.model)(args=args, dynamic=True)
        optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        for epoch in range(args.num_epochs):

            total_loss = 0.0
            num_batches = 0

            # Train
            for (frames, labels) in ds_train:
                for i in range(0, frames.shape[0], args.batch_size):
                    total_loss += distributed_train_step((frames[i:i+args.batch_size], labels[i:i+args.batch_size]))
                    num_batches += 1

            train_loss = total_loss / num_batches

            # Evaluate
            for (frames, labels) in ds_dev:
                for i in range(0, frames.shape[0], args.batch_size):
                    distributed_eval_step((frames[i:i + args.batch_size], labels[i:i+args.batch_size]))

            template = "Epoch {}, Loss: {}, Accuracy: {}, Dev Loss: {}, Dev Accuracy: {}"
            print(template.format(epoch + 1, train_loss,
                                  train_accuracy.result() * 100, dev_loss.result(),
                                  dev_accuracy.result() * 100))

            print('Saving model checkpoint ...')
            checkpoint.save(run_dir)

            dev_loss.reset_states()
            train_accuracy.reset_states()
            dev_accuracy.reset_states()





