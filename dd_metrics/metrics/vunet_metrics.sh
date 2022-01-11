#!/bin/bash


np1=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/our | wc -l`
np0=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img | wc -l`

n=`expr $np1 - $np0`

python lpips.py -i /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/vunet/render/ -f .png -o /home/joaoferreira/Desktop/vunet/$1/$2 -n $n -c /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img/ -fc .jpg
