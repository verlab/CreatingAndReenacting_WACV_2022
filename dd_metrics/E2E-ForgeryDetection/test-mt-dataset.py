# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import os
import glob
import numpy as np
import E2E.parameters as parameters
from E2E.detection import detection,preload
from E2E.dataloaders.data_loader import loader
import torch
from sklearn import metrics

persons = ['S1/', 'S2/', 'S3/', 'S4/']
movements = ['box/', 'cone/', 'fusion/', 'hand/', 'jump/', 'rotate/', 'shake_hands/', 'simple_walk/']
instances = ['our_ijcv_n_sd-vm/']

def read_dataset(directory):
    paths = []

    for person in persons:
        for movement in movements:
            for p in ['P1/']:
                for instance in instances:
                    for image_path in sorted(glob.glob(directory + person + movement + p + instance + '*.jpg')):
                        paths.append(image_path)

    return paths

def process_image(img_path,parameters):
    print('Processing image: ',img_path)
    score = np.nan
    try:
        X, RGB, NP, RGN,im_mode = loader(img_path, parameters.mode)
    except Exception as e:
        print("Error in opening image file: ",img_path)
        return score

    if np.min(X.shape[0:2])< (parameters.tile_size+parameters.tile_stride):
        print('Image is too small:'+ img_path)
    else:
        try:
            score = detection(X, RGB, NP, RGN, parameters.mode)
            with open('/home/guilhermetorres/results_ijcv/' + os.path.dirname(img_path).replace('/', '_') + '.csv', 'a+') as f:
                print('{}'.format(score), file=f)
            # Update Output
        except Exception as e:
            print("Error in processing the image: ",img_path)
            raise e

    return score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('-g', '--gpu'   , type=str, default=None)
    parser.add_argument('-m','--mode', type=str, default='FUSION') #RGB | N | RGN | FUSION
    parser.add_argument('-tile_size','--tile_size', type=int, default=200)
    parser.add_argument('-tile_stride','--tile_stride', type=int, default=100)
    parser.add_argument('-train_dataset','--train_dataset', type=str, default='E2E')
    parser.add_argument('-test_dataset','--test_dataset', type=str, default='./test/')


    config, _ = parser.parse_known_args()
    parameters.use_cuda = torch.cuda.is_available() # To run on CPU when GPUs are not available
    parameters.mode = config.mode
    parameters.tile_size = config.tile_size
    parameters.tile_stride = config.tile_stride
    parameters.ds = config.train_dataset

    print('Starting E2E Forgery Detection')
    print('Model: {}-{}'.format(parameters.ds,parameters.mode))

    #preload all models - not strictly necessary
    preload(parameters.mode)

    images = read_dataset(config.test_dataset)
    scores = []
    for img_p in images:
        s = process_image(img_path=img_p,parameters=parameters)
