"""
Training script

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
import datetime
from json import dumps

from args import get_test_args
from data_pipeline import FramesDatasetBuilder

from models import create_model

from util import *

def eval_step(inputs):
    frames, labels = inputs

    predictions = model(frames, training=False)

    test_loss.update_state(labels, predictions)
    [test_metric.update_state(labels, predictions) for test_metric in test_metrics]


def distributed_eval_step(dataset_inputs):
    return strategy.run(eval_step, args=(dataset_inputs,))

if __name__ == '__main__':

    args = get_test_args()

    tf.debugging.set_log_device_placement(args.log_device_placement)
    configure_gpus()

    # Setup the output dir
    run_dir = os.path.join(args.save_dir, 'test', args.name + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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

    ds_test = builder.as_dataset(split=tfds.Split.TEST, as_supervised=True).map(builder._process_video_timestep_map_fn).batch(GLOBAL_BATCH_SIZE)

    # distribute the dataset needed by the CentralStorageStrategy strategy
    ds_test = strategy.experimental_distribute_dataset(dataset=ds_test)

    with strategy.scope():

        # define the metrics
        test_loss = tf.keras.metrics.BinaryCrossentropy(name='test_loss')

        test_metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='test_accuracy'),
            tf.keras.metrics.Recall(name='test_recall_class_0', class_id=0),
            tf.keras.metrics.Recall(name='test_recall_class_1', class_id=1),
            tf.keras.metrics.Precision(name='test_precision_class_0', class_id=0),
            tf.keras.metrics.Precision(name='test_precision_class_1', class_id=1)
        ]

        # Create the model, optimizer and checkpoint under 'strategy_scope'
        model = create_model(args.model)(args=args, dynamic=True)

        # Load the pre-trained model weights
        model.load_weights(args.model_checkpoint)

        # Evaluate
        for (frames, labels) in ds_test:
            distributed_eval_step((frames, labels))

        # Print to log
        log_metrics = [f'test_loss - {test_loss.result().numpy()}'] + [
            f'{test_metric.name} - {test_metric.result().numpy()})' for test_metric in test_metrics]
        log.info(f'Test - ' + ', '.join(log_metrics))

    sys.exit(0)