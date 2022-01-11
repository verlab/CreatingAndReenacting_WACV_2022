#!/bin/bash

np1=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/edn/final_results | wc -l`
np0=`ls /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img | wc -l`

n1=$( expr "$np1" - "$np0")
n2=$( expr "$np0" - "$np1")
n1_exp=$( expr "$n1" '*' "$n1")
n2_exp=$( expr "$n2" '*' "$n2")
n1_sqrt=$(echo "sqrt($n1_exp)" | bc)
n2_sqrt=$(echo "sqrt($n2_exp)" | bc)

if (("$n1_sqrt" >= "$n2_sqrt")); then
    n=$n1_sqrt
else
    n=$n2_sqrt
fi

echo "executing script"
mkdir -p /srv/storage/datasets/thiagoluange/mt-dataset/metrics/edn/$1/$2
python lpips.py -i /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P1/edn/final_results/ -f .png -o /srv/storage/datasets/thiagoluange/mt-dataset/metrics/edn/$1/$2/ -n $n -c /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img/ -fc .jpg
echo "executed"