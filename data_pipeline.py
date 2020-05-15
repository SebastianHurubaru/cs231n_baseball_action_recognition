"""
Video data pipeline

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import tensorflow as tf
import os
import json

import tensorflow_datasets.public_api as tfds

from util import image_show

class VideoDataset(tfds.core.GeneratorBasedBuilder):

    def __init__(self, args):

        self.args = args

        self.VERSION = tfds.core.Version(self.args.dataset_version)
        self.MANUAL_DOWNLOAD_INSTRUCTIONS = "Dataset already downloaded manually"

        super(tfds.core.GeneratorBasedBuilder, self).__init__()

    def _info(self):

        ffmpeg_extra_args = ('-s', f'{self.args.frame_width}x{self.args.frame_height}',
                             '-vf', f'fps={self.args.frames_per_second}',
                             '-preset', 'ultrafast')

        return tfds.core.DatasetInfo(
            builder=self,

            description=("Softball video data."),

            features=tfds.features.FeaturesDict({
                "video": tfds.features.Video(
                shape=(None, self.args.frame_height, self.args.frame_width, self.args.frame_channels),
                encoding_format='png',
                    ffmpeg_extra_args=ffmpeg_extra_args),
                # Here, labels can be of 2 distinct values.
                "label": tfds.features.Sequence(feature=tfds.features.ClassLabel(num_classes=2)),
            }),

            supervised_keys=("video", "label"),

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
                    "input_dir": os.path.join(self.args.input_dir, 'train')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "input_dir": os.path.join(self.args.input_dir, 'dev')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "input_dir": os.path.join(self.args.input_dir, 'test')
                },
            )
        ]

    def _generate_examples(self, input_dir):

        file_list = tf.data.Dataset.list_files(os.path.join(input_dir, '*', self.args.metadadata_files))

        for file in file_list:

            # Get the content of the meta file
            video_meta = json.loads(tf.io.read_file(file).numpy().decode('utf-8'))

            radar_data_available = video_meta['transcodeConfig'].get('pradar', False)

            radar_velocity = None

            if radar_data_available is True:

                radar_velocity = [0] * int(video_meta['mpegDurationSeconds'])

                # recover new radar event from the radar velocity overlay
                radarOverlays = video_meta['transcodeConfig']['radarOverlaySet']

                for overlay in radarOverlays:
                    assert ('contentHash' in overlay) or (
                                'overlayContentHash' in overlay), "ERR overlay unexpected " + str(overlay)

                    contentHash = overlay.get('contentHash', overlay.get('overlayContentHash', ''))
                    slotStatus = json.loads(contentHash)

                    active = list(filter(lambda x: slotStatus[1][x] is True, range(len(slotStatus[1]))))
                    if(len(active) > 0):

                        # Get the velocity at each time
                        t = overlay['vidTimeInSecs']

                        # Set label to 1 for the found timestep
                        radar_velocity[t] = 1

            # Get the directory where the file is located
            current_dir = os.path.dirname(file.numpy().decode('utf-8'))

            yield video_meta['mpegFilename'], {
                'video': os.path.join(current_dir, video_meta['mpegFilename']),
                'label': radar_velocity
            }

    def _normalize_frames(self, frames, labels):

        # Keep only the frames corresponding to the integer number of seconds
        frames = tf.reshape(frames[:len(labels) * self.args.frames_per_second],
                            [-1, self.args.frames_per_second, self.args.frame_height, self.args.frame_width, self.args.frame_channels])

        frames = tf.cast(frames, tf.float32) / 255.
        return (frames, labels)
