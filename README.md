# resnet-18-tensorflow

A TensorFlow implementation of ResNet-18(https://arxiv.org/abs/1512.03385)

<b>Prerequisite</b>

1. TensorFlow
2. ImageNet dataset(All images are required to be valid JPEG files. See [this gist](https://gist.github.com/dalgu90/fc358fdde0a7fe6fbbe0254b901a0de3).)

<b>How To Run</b>

1. Please modify `train_scratch.sh` to have valid values of following arguments
  1. `train_dataset`, `train_image_root`, `val_dataset`, `val_image_root`
  2. `num_gpus` and corresponding IDs of GPUs(`CUDA_VISIBLE_DEVICES` at the first line)
2. run
  1. `$./train_scratch.sh`
