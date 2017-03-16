#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
train_dir="./resnet_baseline"
train_dataset="scripts/train_shuffle.txt"
train_image_root="/data1/common_datasets/imagenet_resized/"
val_dataset="scripts/val.txt"
val_image_root="/data1/common_datasets/imagenet_resized/ILSVRC2012_val/"

python train.py --train_dir $train_dir \
    --train_dataset $train_dataset \
    --train_image_root $train_image_root \
    --val_dataset $val_dataset \
    --val_image_root $val_image_root \
    --batch_size 64 \
    --num_gpus 4 \
    --val_interval 1000 \
    --val_iter 100 \
    --l2_weight 0.0001 \
    --initial_lr 0.01 \
    --lr_step_epoch 10.0 \
    --lr_decay 0.1 \
    --max_steps 100000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
    --basemodel "./init/model.ckpt" \
    #--finetune True \

# ResNet-18 baseline loaded from torch resnet-18.t7
# Finetune for 10 epochs
