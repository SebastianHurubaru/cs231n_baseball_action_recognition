"""
Training script

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
from sklearn.utils import class_weight

import sys
import datetime
from json import dumps

from args import get_train_args
from data_pipeline import FramesDatasetBuilder

from models import create_model

from util import *


def train_step(inputs):
    frames, labels = inputs

    idx = tf.stack([tf.reshape(tf.range(labels.shape[0], dtype=tf.int64), (-1, 1)),
                    tf.reshape(tf.argmax(labels, axis=1), (-1, 1))],
                   axis=-1)
    class_weights = tf.gather_nd(tf.convert_to_tensor([args.class_weights] * labels.shape[0]),
                                 indices=idx)

    with tf.GradientTape() as tape:
        predictions = model(frames, training=True)
        loss = compute_loss(labels, predictions, class_weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    [train_metric.update_state(labels, predictions) for train_metric in train_metrics]

    return loss


def eval_step(inputs):
    frames, labels = inputs

    predictions = model(frames, training=False)

    dev_loss.update_state(labels, predictions)
    [dev_metric.update_state(labels, predictions) for dev_metric in dev_metrics]


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
    [tf.summary.scalar(train_metric.name, train_metric.result() * 100, step=checkpoint.step.numpy()) for train_metric in train_metrics]

    # Print to log
    log_metrics = [f'train_loss - {total_loss / num_batches}'] + [
        f'{train_metric.name} - {train_metric.result().numpy()})' for train_metric in train_metrics]
    log.info(f'Training step - {checkpoint.step.numpy()} ' + ', '.join(log_metrics))


if __name__ == '__main__':

    args = get_train_args()

    tf.debugging.set_log_device_placement(args.log_device_placement)
    configure_gpus()

    # Setup the output dir
    run_dir = os.path.join(args.save_dir, args.name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = tf.summary.create_file_writer(run_dir)
    writer.set_as_default()

    log = get_logger(run_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Setup the frames data
    builder = FramesDatasetBuilder(args=args, log=log)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            download_mode=tfds.core.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
            manual_dir=args.input_dir,
            compute_stats=False
        )
    )

    # Keep all variables on CPU and just do the operations on GPUs
    strategy = tf.distribute.experimental.CentralStorageStrategy()

    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync

    ds_train = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True).map(builder._process_video_timestep_map_fn).batch(GLOBAL_BATCH_SIZE)
    ds_dev = builder.as_dataset(split=tfds.Split.VALIDATION, as_supervised=True).map(builder._process_video_timestep_map_fn).batch(GLOBAL_BATCH_SIZE)

    # # Compute class weights
    # labels = []
    # for _, label in ds_train:
    #     labels += [np.argmax(label[i]) for i in range(label.shape[0])]
    # # length = 24279
    # print(f'Size of the training set is {len(labels)}')
    # labels = np.asarray(labels)
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(labels),
    #                                                   labels)
    #
    # # Print the class weights to add them as default parameters for the train.py
    # print(f'Computed class weights for the training data are: {class_weights}')
    #
    # sys.exit(0)

    # distribute the dataset needed by the CentralStorageStrategy strategy
    ds_train = strategy.experimental_distribute_dataset(dataset=ds_train)
    ds_dev = strategy.experimental_distribute_dataset(dataset=ds_dev)

    with strategy.scope():

        # Create model loss
        loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions, class_weights):
            per_example_loss = loss_object(labels, predictions, class_weights)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        # define the metrics
        dev_loss = tf.keras.metrics.BinaryCrossentropy(name='dev_loss')

        train_metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='train_accuracy'),
            tf.keras.metrics.Recall(name='train_recall_class_0', class_id=0),
            tf.keras.metrics.Recall(name='train_recall_class_1', class_id=1),
            tf.keras.metrics.Precision(name='train_precision_class_0', class_id=0),
            tf.keras.metrics.Precision(name='train_precision_class_1', class_id=1)
        ]

        dev_metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='dev_accuracy'),
            tf.keras.metrics.Recall(name='dev_recall_class_0', class_id=0),
            tf.keras.metrics.Recall(name='dev_recall_class_1', class_id=1),
            tf.keras.metrics.Precision(name='dev_precision_class_0', class_id=0),
            tf.keras.metrics.Precision(name='dev_precision_class_1', class_id=1)
        ]

        # Create the model, optimizer and checkpoint under 'strategy_scope'
        model = create_model(args.model)(args=args, dynamic=True)

        # Create the optimizer dynamically
        if args.use_lr_scheduler is True:
            config = {'learning_rate': tf.keras.optimizers.schedules.serialize(
                    tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=args.learning_rate,
                        decay_rate=args.decay_rate,
                        decay_steps=args.decay_steps
                    )
                )
            }
        else:
            config = {'lr': args.learning_rate}

        config = {'class_name': str(args.optimizer),
                  'config': config }

        optimizer = tf.keras.optimizers.get(config)

        # Create the checkpoint manager
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_dir, max_to_keep=args.max_checkpoints)

        for epoch in range(args.num_epochs):

            total_loss = 0.0
            num_batches = 0

            # Train
            for (frames, labels) in ds_train:
                total_loss += distributed_train_step((frames, labels)).numpy()
                checkpoint.step.assign_add(1)
                num_batches += 1

                if checkpoint.step.numpy() % args.eval_steps == 0:
                    periodically_train_task()

            periodically_train_task()

            # Evaluate
            for (frames, labels) in ds_dev:
                distributed_eval_step((frames, labels))

            tf.summary.scalar('dev_loss', dev_loss.result(), step=checkpoint.step.numpy())
            [tf.summary.scalar(dev_metric.name, dev_metric.result() * 100, step=checkpoint.step.numpy()) for dev_metric
             in dev_metrics]

            # Print to log
            log_metrics = [f'dev_loss - {dev_loss.result().numpy()}'] + [
                f'{dev_metric.name} - {dev_metric.result().numpy()})' for dev_metric in dev_metrics]
            log.info(f'Training step - {checkpoint.step.numpy()} ' + ', '.join(log_metrics))

            dev_loss.reset_states()
            [metric.reset_states() for metric in train_metrics + dev_metrics]

            print(f'Finished epoch {epoch+1} ...')

    sys.exit(0)
