"""
Video data pipeline

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import tensorflow as tf
import os
import json

import numpy as np

import tensorflow_datasets.public_api as tfds

from util import get_logger

from util import image_show


class FramesDatasetBuilder(tfds.core.GeneratorBasedBuilder):

    def __init__(self, args, log):

        self.args = args

        self.log = log

        self.VERSION = tfds.core.Version(self.args.dataset_version)
        self.MANUAL_DOWNLOAD_INSTRUCTIONS = "Dataset already downloaded manually"

        super(tfds.core.GeneratorBasedBuilder, self).__init__()

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,

            description=("Softball frame video data."),

            features=tfds.features.FeaturesDict({
                "frame": tfds.features.Sequence(feature=tfds.features.Image()),
                "label": tfds.features.Tensor(shape=(2,), dtype=tf.int32),
            }),

            supervised_keys=("frame", "label"),

            homepage="https://xxx",

            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Hurubaru, Sebastian},"}""",
        )

    def _split_generators(self, dl_manager):

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "split": tfds.Split.TRAIN,
                    "input_dir": os.path.join(self.args.input_dir, 'train')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "split": tfds.Split.VALIDATION,
                    "input_dir": os.path.join(self.args.input_dir, 'dev')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "split": tfds.Split.TEST,
                    "input_dir": os.path.join(self.args.input_dir, 'test')
                },
            )
        ]

    def _generate_examples(self, split, input_dir):

        file_list = tf.data.Dataset.list_files(os.path.join(input_dir, '*', self.args.metadadata_files))

        for file in file_list:

            # Get the content of the meta file
            video_meta = json.loads(tf.io.read_file(file).numpy().decode('utf-8'))

            radar_data_available = video_meta['transcodeConfig'].get('pradar', False)

            radar_velocity = None

            keep_index = None

            if radar_data_available is True:

                radar_velocity = [[1, 0]] * int(video_meta['mpegDurationSeconds'])
                keep_index = np.asarray([False] * int(video_meta['mpegDurationSeconds']))

                # recover new radar event from the radar velocity overlay
                radarOverlays = video_meta['transcodeConfig']['radarOverlaySet']

                for overlay in radarOverlays:
                    assert ('contentHash' in overlay) or (
                            'overlayContentHash' in overlay), "ERR overlay unexpected " + str(overlay)

                    contentHash = overlay.get('contentHash', overlay.get('overlayContentHash', ''))
                    slotStatus = json.loads(contentHash)

                    active = list(filter(lambda x: slotStatus[1][x] is True, range(len(slotStatus[1]))))
                    if (len(active) > 0):
                        # Get the velocity at each time
                        t = overlay['vidTimeInSecs']

                        # Set label to 1 for the found timestep
                        radar_velocity[t] = [0, 1]
                        keep_index[t-self.args.timesteps_to_keep:t] = True
                        keep_index[t:t + self.args.timesteps_to_keep + 1] = True

            # Get the directory where the file is located
            current_dir = os.path.dirname(file.numpy().decode('utf-8'))

            # Call first ffmpeg to save all frames on disk
            _, current_dir_name = os.path.split(current_dir)

            target_dir = os.path.join(
                f"{self.args.temp_proc_dir}_{self.args.frames_per_second}x{self.args.frame_width}x{self.args.frame_height}x{self.args.frame_channels}",
                current_dir_name)

            # Do not run ffmpeg if already run
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

                command = " ".join([
                    'ffmpeg',
                    '-i', os.path.join(current_dir, video_meta['mpegFilename']),
                    '-s', f'{self.args.frame_width}x{self.args.frame_height}',
                    '-vf', f'fps={self.args.frames_per_second}',
                    '-vf hue=s=0' if self.args.frame_channels == 1 else '',
                    os.path.join(target_dir, '%d.jpg')
                ])

                os.system(command)

            self.log.info(f"Processing video {video_meta['mpegFilename']}")

            prev_frame_index = 1
            for t in range(int(video_meta['mpegDurationSeconds'])):

                current_frame_index = (t+1) * self.args.frames_per_second

                if keep_index[t]:
                    frame = [os.path.join(target_dir, f'{frame}.jpg') for frame in range(prev_frame_index, current_frame_index+1)]

                    yield f"{video_meta['mpegFilename']}_{t}", {
                        'frame': frame,
                        'label': radar_velocity[t]
                    }

                prev_frame_index = current_frame_index + 1
