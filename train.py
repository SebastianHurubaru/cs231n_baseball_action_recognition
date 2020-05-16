"""
Training script

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""
import tensorflow as tf
import tensorflow_datasets as tfds

import datetime

from args import get_train_args
from data_pipeline import FramesDatasetBuilder

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


def periodically_train_task():

    # Save the checkpoint
    save_path = checkpoint_manager.save()
    print("Saved checkpoint for step {}: {}".format(checkpoint.step.numpy(), save_path))

    # Display metrics on tensorboard
    tf.summary.scalar('train_loss', total_loss / num_batches, step=checkpoint.step.numpy())
    tf.summary.scalar('train_accuracy', train_accuracy.result() * 100, step=checkpoint.step.numpy())

if __name__ == '__main__':

    args = get_train_args()

    tf.debugging.set_log_device_placement(args.log_device_placement)

    configure_gpus()

    # Setup the output dir
    run_dir = args.save_dir + '_' + args.name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup the frames data
    builder = FramesDatasetBuilder(args=args)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            download_mode=tfds.core.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
            manual_dir=args.input_dir
        )
    )

    # Keep all variables on CPU and just do the operations on GPUs
    strategy = tf.distribute.experimental.CentralStorageStrategy()

    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync

    ds_train = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True).batch(GLOBAL_BATCH_SIZE)
    ds_dev = builder.as_dataset(split=tfds.Split.VALIDATION, as_supervised=True).batch(GLOBAL_BATCH_SIZE)

    # distribute the dataset needed by the CentralStorageStrategy strategy
    ds_train = strategy.experimental_distribute_dataset(dataset=ds_train)
    ds_dev = strategy.experimental_distribute_dataset(dataset=ds_dev)

    writer = tf.summary.create_file_writer(run_dir)
    writer.set_as_default()

    with strategy.scope():

        # Create model loss
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=tf.keras.losses.Reduction.NONE)


        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        # define the metrics
        dev_loss = tf.keras.metrics.Mean(name='dev_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='dev_accuracy')

        # Create the model, optimizer and checkpoint under 'strategy_scope'
        model = create_model(args.model)(args=args, dynamic=True)
        optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_dir, max_to_keep=3)

        for epoch in range(args.num_epochs):

            total_loss = 0.0
            num_batches = 0

            # Train
            for (frames, labels) in ds_train:
                total_loss += distributed_train_step((frames, labels)).numpy()
                checkpoint.step.assign_add(1)
                num_batches += 1

                if checkpoint.step.numpy() % 100 == 0:
                    periodically_train_task()

            periodically_train_task()

            # Evaluate
            for (frames, labels) in ds_dev:
                distributed_eval_step((frames, labels))

            tf.summary.scalar('dev_loss', dev_loss.result(), step=checkpoint.step.numpy())
            tf.summary.scalar('dev_accuracy', dev_accuracy.result() * 100, step=checkpoint.step.numpy())

            dev_loss.reset_states()
            train_accuracy.reset_states()
            dev_accuracy.reset_states()

            print(f'Finished epoch{epoch}')

    exit(0)
