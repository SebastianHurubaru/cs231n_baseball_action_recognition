"""
Command-line arguments for setup.py, train.py, test.py.

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import argparse


def get_train_args():
    """
    Get arguments needed in train.py.
    """
    parser = argparse.ArgumentParser('Train a model on softball data')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs for which to train.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate.')

    parser.add_argument('--decay_steps',
                        type=float,
                        default=10**4,
                        help='Learning rate scheduler step size.')

    parser.add_argument('--decay_rate',
                        type=float,
                        default=0.95,
                        help='Decay rate for exponential moving average of parameters.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='mse_loss',
                        choices=('mse_loss',),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--eval_steps',
                        type=int,
                        default=100,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adadelta',
                        choices=('Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD'),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--use_lr_scheduler',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to use learn rate scheduler.')

    parser.add_argument('--class_weights',
                        type=float,
                        nargs="*",
                        default=[0.57253691, 3.94652146],
                        help='Class weights.')

    args = parser.parse_args()

    return args


def get_test_args():
    """
    Get arguments needed in test.py.
    """
    parser = argparse.ArgumentParser('Test a trained model on Baseball video data')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--model_checkpoint',
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def add_common_args(parser):

    """
    Add arguments common to all 3 scripts: train.py, test.py
    """

    parser.add_argument('--input_dir',
                        type=str,
                        default='./data')

    parser.add_argument('--temp_proc_dir',
                        type=str,
                        default='~/tensorflow_datasets/videos_to_frames')

    parser.add_argument('--dataset_version',
                        type=str,
                        default='0.0.1')

    parser.add_argument('--metadadata_files',
                        type=str,
                        default='*-full-metadata.json')

    parser.add_argument('--frames_per_second',
                        type=int,
                        default='30')

    parser.add_argument('--frame_height',
                        type=int,
                        default='720')

    parser.add_argument('--frame_width',
                        type=int,
                        default='1280')

    parser.add_argument('--frame_channels',
                        type=int,
                        default='3')

    parser.add_argument('--timesteps_to_keep',
                        type=int,
                        default='3')

    parser.add_argument('--model',
                        type=str,
                        choices=['baseline', 'i3d'],
                        default='baseline')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--include_posing',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to include human posing.')

    parser.add_argument('--posing_batch_size',
                        type=int,
                        default='16')

    parser.add_argument('--cfg',
                        type=str,
                        default='./posing/experiments/vgg19_368x368_sgd.yaml',
                        help='experiment configure file name')

    parser.add_argument('--weight',
                        type=str,
                        default='./posing/experiments/pose_model.pth')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--log_device_placement',
                        type=bool,
                        default=False)

def add_train_test_args(parser):
    """
    Add arguments common to train.py and test.py
    """
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify subdir or test run.')

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    parser.add_argument('--seed',
                        type=int,
                        default=231,
                        help='Random seed for reproducibility.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=1024,
                        help='Number of features in the hidden layers.')

    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.4,
                        help='Probability of zeroing an activation in dropout layers.')

    parser.add_argument('--data_shuffle',
                        type=bool,
                        default=False,
                        help='Shuffle the data.')

