#!/bin/bash

lpip=LPIPS.txt
ssim=SSMI.txt
mse=MSE.txt
# for s in S1_aux
for action in box cone fusion hand shake_hands;
    do
        # mkdir /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/our
        # mkdir /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/vid2vid
        mkdir /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/vunet
        for s in S1 S2 S3 S4;
            do
                name=`ls /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s | grep $action$lpip`
                cp /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s/$name /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/vunet/$s$name
                name=`ls /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s | grep $action$ssim`
                cp /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s/$name /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/vunet/$s$name
                name=`ls /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s | grep $action$mse`
                cp /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vunet/$s/$name /srv/storage/datasets/thiagoluange/mt-dataset/metrics/actions/$action/vunet/$s$name
            done;
    done;