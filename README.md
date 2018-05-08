# resnet-18-tensorflow

A TensorFlow implementation of ResNet-18(https://arxiv.org/abs/1512.03385)

<b>Prerequisite</b>

1. TensorFlow 1.8
2. The ImageNet dataset
  - All image files are required to be valid JPEG files. See [this gist](https://gist.github.com/dalgu90/fc358fdde0a7fe6fbbe0254b901a0de3).
  - It is highly recommened for every image to be resized so that the shorter side is 256.
3. (Optional) [Torchfile](https://github.com/bshillingford/python-torchfile)(to convert ResNet-18 .t7 checkpoint into tensorflow checkpoint. Install with a command `pip install torchfile`)

<b>How To Run</b>

- (Optional) Convert torch .t7 into tensorflow ckpt
```
# Download the ResNet-18 torch checkpoint
wget https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
# Convert into tensorflow checkpoint
python extract_torch_t7.py
```
1. Modify `train_scratch.sh`(training from scratch) or `train.sh`(finetune pretrained weights) to have valid values of following arguments
  - `train_dataset`, `train_image_root`, `val_dataset`, `val_image_root`: Path to the list file of train/val dataset and to the root
  - `num_gpus` and corresponding IDs of GPUs(`CUDA_VISIBLE_DEVICES` at the first line)
2. Run!
  - `./train.sh` if you want to finetune the converted ResNet(NOTE: The model needs to be finetuned for some epochs)
  - `./train_scratch.sh` if you want to train ResNet from scratch
3. Evaluate the trained model
  - `./eval.sh` for evaluating the trained model(change the arguments in `eval.sh` to your preference)

<b>Note</b>

- The extracted weights should be finetuned for several epochs(run `./train.sh`) to get the full performance(If you run the evaluation code without finetuning, the single-crop top-1 validation accuracy is about 60%, which is less than the appeared in [the original](https://github.com/facebook/fb.resnet.torch)). I guess there is some minor issue that I have missed.
