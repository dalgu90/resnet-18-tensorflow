# resnet-18-tensorflow

A TensorFlow implementation of ResNet-18(https://arxiv.org/abs/1512.03385)

<b>Prerequisite</b>

1. TensorFlow 1.0
2. The ImageNet dataset
  1. All image files are required to be valid JPEG files. See [this gist](https://gist.github.com/dalgu90/fc358fdde0a7fe6fbbe0254b901a0de3).
  2. It is highly recommened for every image to be resized so that the shorter side is 256.
3. (Optional) [Torchfile](https://github.com/bshillingford/python-torchfile)(to convert ResNet-18 .t7 checkpoint into tensorflow checkpoint `pip install torchfile`)

<b>How To Run</b>

0. (Optional) Convert torch .t7 into tensorflow ckpt
```
# Download torch
wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7

# Convert .t7 into tensorflow checkpoint
python extract_torch_t7.py
```
1. Modify `train_scratch.sh` or `train.sh` to have valid values of following arguments
  1. `train_dataset`, `train_image_root`, `val_dataset`, `val_image_root`: Path to the list file of train/val dataset and to the root
  2. `num_gpus` and corresponding IDs of GPUs(`CUDA_VISIBLE_DEVICES` at the first line)
2. Run!
  1. `./train_scratch.sh` or `./train.sh`


