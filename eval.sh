#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
checkpoint="./resnet_baseline/model.ckpt-99999"
test_dataset="scripts/val.txt"
test_image_root="/data1/common_datasets/imagenet_resized/ILSVRC2012_val/"
output_file="./resnet_baseline/eval-99999.pkl"

python eval.py --checkpoint $checkpoint \
    --test_dataset $test_dataset \
    --test_image_root $test_image_root \
    --output_file $output_file \
    --batch_size 100 \
    --test_iter 500 \
    --ngroups1 1 \
    --ngroups2 1 \
    --gpu_fraction 0.96 \
    --display 10 \
