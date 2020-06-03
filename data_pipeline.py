"""
Video data pipeline

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import tensorflow as tf
import os
import glob
import json

import numpy as np

import tensorflow_datasets.public_api as tfds

from util import torch_get_available_devices

import cv2
from posing.lib.network.rtpose_vgg import get_model
from posing.lib.config import cfg, update_config
from posing.lib.utils.common import draw_humans
from posing.lib.utils.paf_to_pose import paf_to_pose_cpp
from posing.lib.datasets.preprocessing import rtpose_preprocess

import torch
import torch.utils.data as data

# import pydevd

class FramesDatasetBuilder(tfds.core.GeneratorBasedBuilder):

    def __init__(self, args, log):

        self.args = args

        self.log = log

        self.VERSION = tfds.core.Version(self.args.dataset_version)
        self.MANUAL_DOWNLOAD_INSTRUCTIONS = "Dataset already downloaded manually"

        super(tfds.core.GeneratorBasedBuilder, self).__init__()

        # update config file
        update_config(cfg, args)

        model = get_model('vgg19')
        model.load_state_dict(torch.load(args.weight))
        self.pose_model = model

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
                f"{os.path.expanduser(self.args.temp_proc_dir)}_{self.args.frames_per_second}x{self.args.frame_width}x{self.args.frame_height}x{self.args.frame_channels}",
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

            dest_dir = target_dir
            gen_joints = False
            if self.args.include_posing:

                dest_dir = target_dir + '_posing'
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    gen_joints = True

            prev_frame_index = 1
            for t in range(int(video_meta['mpegDurationSeconds'])):

                current_frame_index = (t+1) * self.args.frames_per_second

                if keep_index[t]:

                    if gen_joints is True:
                        self.generate_posing_from_pictures([os.path.join(target_dir, f'{frame}.jpg') for frame in
                                                            range(prev_frame_index, current_frame_index + 1)], dest_dir)

                    frame = [os.path.join(dest_dir, f'{frame}.jpg') for frame in range(prev_frame_index, current_frame_index + 1)]

                    yield f"{video_meta['mpegFilename']}_{t}", {
                        'frame': frame,
                        'label': radar_velocity[t]
                    }

                prev_frame_index = current_frame_index + 1

    def generate_posing_from_pictures(self, images, dest_folder):

        image_file_names = [os.path.split(image)[1] for image in images]

        oriImages = [cv2.imread(image) for image in images]  # B,G,R order

        # Get results of original image
        with torch.no_grad():
            paf, heatmap = self.get_outputs(oriImages, self.pose_model)

        humans = [paf_to_pose_cpp(heatmap[i], paf[i], cfg) for i in range(len(images))]

        out = [draw_humans(oriImages[i], humans[i]) for i in range(len(images))]

        [cv2.imwrite(os.path.join(dest_folder, image_file_names[i]), out[i]) for i in range(len(images))]

    def get_outputs(self, images, model):
        """Computes the averaged heatmap and paf for givens images
        """

        im_data = [rtpose_preprocess(image) for image in images]

        batch_images = np.asarray(im_data)

        # Put model on GPU
        device, gpu_ids = torch_get_available_devices()
        model_gpu = torch.nn.DataParallel(model, gpu_ids)
        model_gpu.float()
        model_gpu.eval()

        # several scales as a batch
        output1, output2 = None, None

        images_loader = data.DataLoader(batch_images,
                                       batch_size=self.args.posing_batch_size,
                                       shuffle=False,
                                       num_workers=0)


        for image_batch in images_loader:

            predicted_outputs, _ = model_gpu(image_batch)
            predicted_outputs = predicted_outputs[0].cpu().data.numpy(), predicted_outputs[1].cpu().data.numpy()
            torch.cuda.empty_cache()
            if output1 is not None and output2 is not None:
                output1 = np.concatenate([output1, predicted_outputs[-2]], axis=0)
                output2 = np.concatenate([output2, predicted_outputs[-1]], axis=0)
            else:
                output1, output2 = predicted_outputs[-2], predicted_outputs[-1]

        heatmap = output2.transpose(0, 2, 3, 1)
        paf = output1.transpose(0, 2, 3, 1)

        return paf, heatmap


    def _process_video_timestep_map_fn(self, timestep, label):

        if self.args.model == 'baseline':

            processed_timestep, label = tf.py_function(self._process_video_timestep,
                                                   inp=[timestep, label],
                                                   Tout=(tf.float32, tf.int32))
        else:
            processed_timestep, label = tf.py_function(self._empty_process_video_timestep,
                                                       inp=[timestep, label],
                                                       Tout=(tf.float32, tf.int32))

        return processed_timestep, label

    def _process_video_timestep(self, timestep, label):

        # Crop the center of the image and take every two frames
        new_timestep = timestep[::3, 2*self.args.frame_height//5:, self.args.frame_width//4:self.args.frame_width - self.args.frame_width//4]

        # To allow debugging in the combined static eager mode
        # pydevd.settrace(suspend=True)
        #
        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # plt.imshow(new_timestep[0].numpy())
        # plt.axis('off')
        # plt.show()
        # plt.close()

        return (tf.image.convert_image_dtype(new_timestep, dtype=tf.float32, saturate=False), label)

    def _empty_process_video_timestep(self, timestep, label):

        return tf.image.convert_image_dtype(timestep, dtype=tf.float32, saturate=False), label