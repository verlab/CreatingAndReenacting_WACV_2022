#!/bin/bash
#first pass our and ground truth

np1=`ls $1 | wc -l`
np0=`ls $2 | wc -l`

n=`expr $np1 - $np0`

mkdir /home/joaoferreira/Desktop/ablation/$3
python lpips.py -i $1 -f out -o /home/joaoferreira/Desktop/ablation/$3/$3 -n $n -c $2 -fc .jpg
