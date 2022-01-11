#!/bin/bash

np1=`ls /srv/storage/datasets/thiagoluange/dd_test_results/$1/$2/P1/result | wc -l`
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
echo $n
mkdir -p /srv/storage/datasets/thiagoluange/dd_dataset/test_results/metrics/$1/$2
python metrics/lpips.py -i /srv/storage/datasets/thiagoluange/dd_test_results/$1/$2/P1/result/ -f .jpg -o /srv/storage/datasets/thiagoluange/dd_dataset/test_results/metrics/$1/$2/ -n $n -c /srv/storage/datasets/thiagoluange/mt-dataset/$1/$2/P0/test_img/ -fc .jpg
echo "executed"
