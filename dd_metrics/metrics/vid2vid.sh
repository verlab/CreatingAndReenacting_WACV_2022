#!/bin/bash


np1=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/our | wc -l`
np0=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img | wc -l`

n=`expr $np1 - $np0`

python lpips.py -i /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/vid2vid/final_render/ -f .jpg -o /srv/storage/datasets/thiagoluange/mt-dataset/metrics/vid2vid/$1/$2 -n $n -c /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img/ -fc .jpg