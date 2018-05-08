#
# This code is modified from the TensorFlow tutorial below.
#
# TensorFlow Tutorial - Convolutional Neural Networks
#  (https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html)
#
# ==============================================================================

"""Routine for loading the image file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import cPickle as pickle
import numpy as np

from tensorflow.python.platform import gfile


# Constants used in the model
RESIZE_SIZE = 256
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def parse_input_line(line, dataset_root):
    """
    Parse dataset line and return image and label
    """
    # Parse the line -> subpath, label
    record_default = [[''], [0]]
    parsed_entries = tf.decode_csv(line, record_default, field_delim=' ')
    image_path = dataset_root + parsed_entries[0] # String tensors can be concatenated by add operator
    label = tf.cast(parsed_entries[1], tf.int32)

    # Read image
    raw_jpeg = tf.read_file(image_path)
    image = tf.image.decode_jpeg(raw_jpeg, channels=3)

    return image, label

def resize_image(input_image, random_aspect=False):
    # Resize image so that the shorter side is 256
    height_orig = tf.shape(input_image)[0]
    width_orig = tf.shape(input_image)[1]
    ratio_flag = tf.greater(height_orig, width_orig)  # True if height > width
    if random_aspect:
        aspect_ratio = tf.random_uniform([], minval=0.875, maxval=1.2, dtype=tf.float64)
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig*aspect_ratio, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig*aspect_ratio, tf.int32))
    else:
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig, tf.int32))
    image = tf.image.resize_images(input_image, [height, width])
    return image

def random_sized_crop(input_image):
    # Input image -> crop with random size and random aspect ratio
    height_orig = tf.cast(tf.shape(input_image)[0], tf.float64)
    width_orig = tf.cast(tf.shape(input_image)[1], tf.float64)

    aspect_ratio = tf.random_uniform([], minval=0.75, maxval=1.33, dtype=tf.float64)
    height_max = tf.minimum(height_orig, width_orig*aspect_ratio)
    height_crop = tf.random_uniform([], minval=tf.minimum(height_max, tf.maximum(0.5*height_orig, 0.5*height_max))
                                    , maxval=height_max, dtype=tf.float64)
    width_crop = height_crop / aspect_ratio
    height_crop = tf.cast(height_crop, tf.int32)
    width_crop = tf.cast(width_crop, tf.int32)

    crop = tf.random_crop(input_image, [height_crop, width_crop, 3])

    # Resize to 224x224
    image = tf.image.resize_images(crop, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image

def lighting(input_image):
    # Lighting noise (AlexNet-style PCA-based noise) from torch code
    # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = input_image + tf.reduce_sum(rgb, axis=0)
    return image

def preprocess(image, label, distortion, center_crop):
    # Image augmentation
    if not distortion:
        # Resize_image
        image = resize_image(image)

        # Crop(random/center)
        height = IMAGE_HEIGHT
        width = IMAGE_WIDTH
        if not center_crop:
            image = tf.random_crop(image, [height, width, 3])
        else:
            image_shape = tf.shape(image)
            h_offset = tf.cast((image_shape[0]-height)/2, tf.int32)
            w_offset = tf.cast((image_shape[1]-width)/2, tf.int32)
            image = tf.slice(image, [h_offset, w_offset, 0], [height, width, 3])
    else:
        # Image augmentation for training the network. Note the many random distortions applied to the image.
        image = random_sized_crop(image)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Because these operations are not commutative, consider randomizing the order their operation.
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

        # Lighting noise
        image = lighting(image)

    # Preprocess: imagr normalization per channel
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    image = (image - imagenet_mean) / imagenet_std

    return image, label

def inputs_base(dataset_root, txt_fpath, batch_size, shuffle=False, num_threads=60, num_sets=1, center_crop=False):
    """Construct input for IMAGENET training/evaluation using the Reader ops.

    Args:
        dataset_root: Path to the root of ImageNet datasets
        txt_fpath: Path to the txt file including image subpaths and labels
        batch_size: Number of images per batch(per set).
        num_sets: Number of sets. Note that the images are prefetched to GPUs.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    for f in [dataset_root, txt_fpath]:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with open(txt_fpath, 'r') as fd:
        num_examples_per_epoch = len(fd.readlines())

    print('\tLoad file list from %s: Total %d files' % (txt_fpath, num_examples_per_epoch))
    print('\t\tBatch size: %d, %d sets of batches, %d threads per batch' % (batch_size, num_sets, num_threads))

    # Read txt file containing image filepaths and labels
    dataset = tf.data.TextLineDataset([txt_fpath])
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    else:
        dataset = dataset.repeat()

    # Read txt line and load image and label
    dataset_root_t = tf.constant(dataset_root)
    parse_map = functools.partial(parse_input_line, dataset_root=dataset_root_t)
    dataset = dataset.map(parse_map)
    # dataset = dataset.map(parse_map, num_parallel_calls=num_threads)

    # Preprocess images
    images_list, labels_list = [], []
    for i in range(num_sets):
        preprocess_map = functools.partial(preprocess, distortion=False, center_crop=center_crop)
        dataset_set = dataset.apply(tf.contrib.data.map_and_batch(preprocess_map, batch_size, num_threads))

        # dataset_set = dataset_set.prefetch(10)
        dataset_set = dataset_set.apply(tf.contrib.data.prefetch_to_device('/GPU:%d'%i))
        iterator = dataset_set.make_one_shot_iterator()
        images, labels = iterator.get_next()
        images.set_shape((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        labels.set_shape((batch_size, ))

        images_list.append(images)
        labels_list.append(labels)

    return images_list, labels_list

def distorted_inputs(dataset_root, txt_fpath, batch_size, shuffle=True, num_threads=60, num_sets=1):
    return inputs_base(dataset_root, txt_fpath, batch_size, shuffle, num_threads, num_sets, False)

def inputs(dataset_root, txt_fpath, batch_size, shuffle=False, num_threads=60, num_sets=1, center_crop=False):
    return inputs_base(dataset_root, txt_fpath, batch_size, shuffle, num_threads, num_sets, center_crop)

