# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.

The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from frechet_video_distance import create_id3_embedding, calculate_fvd, preprocess
import numpy as np
import cv2
import sys
import os
import pandas as pd
import logging
import re
logger = logging.getLogger(__name__)
# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 4
VIDEO_LENGTH = 15

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized = cv2.resize(image, dim, interpolation = inter)
	return resized

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def read_frames(frames_f):
    frames = []

    _files = os.listdir(frames_f)
    for frame in natural_sort(_files):
        if ".jpg" in frame.lower() or ".png" in frame.lower():
            frame_fp = os.path.join(frames_f, frame)
            img = cv2.imread(frame_fp)
            (height, width) = img.shape[:2]
            image_res = image_resize(img, height = 224) if height < width else image_resize(img, width = 224)
            frames.append(image_res)
    return frames

def normalize_sizes(videos_set):
    _len = sys.maxsize

    for _set in videos_set:
        if _len > len(_set):
            _len = len(_set)

    return [ _set[:_len] for _set in videos_set]
    
def get_videos_s(mt_root, movement, baseline = "our", real = False):

    subjects = ["S1", "S2", "S3", "S4"]
    person = "P1" if not real else "P0"
    videos_set = []
    for subj in subjects:
        frames_fp = os.path.join(mt_root, subj, movement, person, baseline)
        logger.info("READING FRAMES FROM {}".format(frames_fp))
        frames_lst = read_frames(frames_fp)
        videos_set.append(frames_lst)
    
    videos_set = normalize_sizes(videos_set)
    return videos_set

def get_videos_m(mt_root, subj, baseline = "our", real = False):
    
    movements = ["box", "cone", "fusion", "hand", "jump", "rotate", "shake_hands", "simple_walk"]
    person = "P1" if not real else "P0"
    videos_set = []

    for movement in movements:
        frames_fp = os.path.join(mt_root, subj, movement, person, baseline)
        logger.info("READING FRAMES FROM {}".format(frames_fp))
        frames_lst = read_frames(frames_fp)
        videos_set.append(frames_lst)
    
    videos_set = normalize_sizes(videos_set)
    return videos_set

def get_videos_iper(data_root, baseline):
    frames_fp = os.path.join(data_root, baseline)
    frames_lst = read_frames(frames_fp)
    return frames_lst

def normalize_size_sets(videos_fake, videos_real):
    len_fake = len(videos_fake[0])
    len_real = len(videos_real[0])

    _min = len_fake if len_fake < len_real else len_real

    videos_fake = [video[:_min] for video in videos_fake]
    videos_real = [video[:_min] for video in videos_real]
    logger.info("VIDEO LEN {}".format(_min))
    return np.asarray(videos_fake), np.asarray(videos_real)

def normalize_size_sets_iper(videos_fake, videos_real):

    len_fake = len(videos_fake)
    len_real = len(videos_real)

    _min = len_fake if len_fake < len_real else len_real

    videos_fake = videos_fake[:_min]
    videos_real = videos_real[:_min]

    return np.asarray(videos_fake), np.asarray(videos_real)

def get_baseline_img_folder(baseline):
    if baseline == "our":
        return "our_ijcv_2"
    elif baseline == "wacv":
        return "our"
    elif baseline == "edn":
        return "edn/final_results/"
    elif baseline == "impersonator":
        return "impersonator/final_results"
    elif baseline == "vid2vid":
        return "vid2vid/final_render"
    elif baseline == "vunet":
        return "vunet/render"
    elif baseline == "our_ijcv_n_ret":
        return "our_ijcv_n_ret"
    elif baseline == "our_ijcv_n_sd-vm":
        return "our_ijcv_n_sd-vm"
    elif baseline == "our_ijcv_n_sem-def":
        return "our_ijcv_n_sem-def"
    elif baseline == "our_ijcv_n_sm":
        return "our_ijcv_n_sm"
    elif baseline == "our_ijcv_n_vmask":
        return "our_ijcv_n_vmask"
    elif baseline == "dd_test_results":
        return "result"

def split_video(video, size = 128):
    _len = len(video)
    _lst = []
    batch_size = int(_len/size)
    for idx in range(0, len(video), batch_size):
        batch = video[idx : idx + batch_size]
        if len(batch) == batch_size:
            _lst.append(batch)
    return np.asarray(_lst)

def main(argv):
    baselines = ["dd_test_results"]
    movements = ["box", "cone", "fusion", "hand", "jump", "rotate", "shake_hands", "simple_walk"]
    subjects = ["S1", "S2", "S3", "S4"]
    mt_root = "/srv/storage/datasets/thiagoluange/mt-dataset/"
    dd_root = "/srv/storage/datasets/thiagoluange/dd_test_results/"
    df = {}

    _type = sys.argv[1]

    if _type == "movement":
        df["movements"] = movements
        for baseline in baselines:
            fvd_values = []
            for movement in movements:

                logger.info("Running for baseline: {}  -- movement {}".format(baseline, movement))

                videos_fake = get_videos_s(dd_root, movement, get_baseline_img_folder(baseline))
                videos_real = get_videos_s(mt_root, movement, "test_img", True)

                videos_fake, videos_real = normalize_size_sets(videos_fake, videos_real)

                #del argv
                with tf.Graph().as_default():

                    #first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
                    #second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]) * 255
                    first_set_of_videos = tf.convert_to_tensor(videos_fake)
                    second_set_of_videos = tf.convert_to_tensor(videos_real)
                    batch_size = first_set_of_videos.shape[0]
                    logger.info("BATCH_SIZE: {}".format(batch_size))
                    result = calculate_fvd(
                        create_id3_embedding(preprocess(first_set_of_videos, (224, 224)), batch_size),
                        create_id3_embedding(preprocess(second_set_of_videos,(224, 224)), batch_size)
                    )

                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.tables_initializer())
                        fvd = sess.run(result)
                        fvd_values.append(fvd)
                        print("FVD is: %.2f." % fvd)
            df[baseline] = fvd_values
        df = pd.DataFrame.from_dict(df)
        df.to_csv("fvd_values_movement.csv", sep = ";")

    elif _type == "subject":
        df["subjects"] = subjects
        for baseline in baselines:
            fvd_values = []
            for subject in subjects:

                logger.info("Running for baseline: {}  -- subject {}".format(baseline, subject))

                videos_fake = get_videos_m(dd_root, subject, get_baseline_img_folder(baseline))
                videos_real = get_videos_m(mt_root, subject, "test_img", True)

                videos_fake, videos_real = normalize_size_sets(videos_fake, videos_real)
            
                #del argv
                with tf.Graph().as_default():

                    #first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
                    #second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]) * 255
                    first_set_of_videos = tf.convert_to_tensor(videos_fake)
                    second_set_of_videos = tf.convert_to_tensor(videos_real)
                    batch_size = first_set_of_videos.shape[0]
                    logger.info("BATCH_SIZE: {}".format(batch_size))
                    result = calculate_fvd(
                        create_id3_embedding(preprocess(first_set_of_videos, (224, 224)), batch_size),
                        create_id3_embedding(preprocess(second_set_of_videos,(224, 224)), batch_size)
                    )

                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.tables_initializer())
                        fvd = sess.run(result)
                        fvd_values.append(fvd)
                        print("FVD is: %.2f." % fvd)
            df[baseline] = fvd_values
        df = pd.DataFrame.from_dict(df)
        df.to_csv("fvd_values_subject.csv", sep = ";")
    elif _type == "iper":
        data_root = "/home/mostqi/rafael/imper_res/result_imper_fid/"
        baselines = ["imper", "our_ijcv"]
        df = {}
        fvd_values = []

        for baseline in baselines:
            _videos_fake = get_videos_iper(data_root, baseline)
            _videos_real = get_videos_iper(data_root, "real")

            _videos_fake = split_video(_videos_fake)
            _videos_real = split_video(_videos_real)
            #_videos_fake, _videos_real = normalize_size_sets_iper(_videos_fake, _videos_real)

            for idx in range(0, len(_videos_real), 4):
                videos_real = _videos_real[idx: idx + 4]
                videos_fake = _videos_fake[idx: idx + 4]
                with tf.Graph().as_default():

                    #first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
                    #second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]) * 255
                    first_set_of_videos = tf.convert_to_tensor(np.asarray(videos_fake))
                    second_set_of_videos = tf.convert_to_tensor(np.asarray(videos_real))

                    batch_size = first_set_of_videos.shape[0]
                    logger.info("BATCH_SIZE: {}".format(batch_size))
                    result = calculate_fvd(
                        create_id3_embedding(preprocess(first_set_of_videos, (224, 224)), batch_size),
                        create_id3_embedding(preprocess(second_set_of_videos,(224, 224)), batch_size)
                    )

                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.tables_initializer())
                        fvd = sess.run(result)
                        fvd_values.append(fvd)
                        print("FVD is: %.2f." % fvd)
            df[baseline] = [np.mean(fvd_values)]
            fvd_values = []
        df = pd.DataFrame.from_dict(df)
        df.to_csv("df_iper.csv", sep = ";")
    

if __name__ == "__main__":
  tf.app.run(main)
