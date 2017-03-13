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

import tensorflow as tf
import cPickle as pickle

from tensorflow.python.platform import gfile

RESNET_MEAN_FPATH = 'ResNet_mean_rgb.pkl'
with open(RESNET_MEAN_FPATH, 'rb') as fd:
    resnet_mean = pickle.load(fd).mean(0).mean(0)

# Constants used in the model
RESIZE_SIZE = 256
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def read_input_file(txt_fpath, dataset_root, shuffle=False):
  """Reads and parses examples from AwA data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    list_fpath: Path to a txt file containing subpath of input image and labels
      line-by-line
    dataset_root: Path to the root of the dataset images.

  Returns:
    An object representing a single example, with the following fields:
      path: a scalar string Tensor of the path to the image file.
      labels: an int32 Tensor with the 64 attributes(0/1)
      image: a [height, width, depth(BGR)] float32 Tensor with the image data
  """

  class DataRecord(object):
    pass
  result = DataRecord()

  # Read a line from the file(list_fname)
  filename_queue = tf.train.string_input_producer([txt_fpath], shuffle=shuffle)
  text_reader = tf.TextLineReader()
  _, value = text_reader.read(filename_queue)

  # Parse the line -> subpath, label
  record_default = [[''], [0]]
  parsed_entries = tf.decode_csv(value, record_default, field_delim=' ')
  result.labels = tf.cast(parsed_entries[1], tf.int32)

  # Read image from the filepath
  # image_path = os.path.join(dataset_root, parsed_entries[0])
  dataset_root_t = tf.constant(dataset_root)
  result.image_path = dataset_root_t + parsed_entries[0] # String tensors can be concatenated by add operator
  raw_jpeg = tf.read_file(result.image_path)
  result.image = tf.image.decode_jpeg(raw_jpeg, channels=3)

  return result


def preprocess_image(input_image):
  # Preprocess the image: resize -> mean subtract -> channel swap (-> transpose X -> scale X)
  image = tf.cast(input_image, tf.float32)
  # image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  # image_R, image_G, image_B = tf.split(2, 3, image)

  # 1) Subtract channel mean
  # blue_mean = 103.062624
  # green_mean = 115.902883
  # red_mean = 123.151631
  # image = tf.concat(2, [image_B - blue_mean, image_G - green_mean, image_R - red_mean], name="centered_bgr")

  # 2) Subtract per-pixel mean(the model have to 224 x 224 size input)
  # image = tf.concat(2, [image_B, image_G, image_R]) - resnet_mean
  image = image - resnet_mean

  # image = tf.concat(2, [image_R, image_G, image_B]) # BGR -> RGB
  # imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
  # image = image - imagenet_mean # [224, 224, 3] - [3] (Subtract with broadcasting)
  # image = tf.transpose(image, [2, 0, 1]) # No transpose
  # No scaling

  return image


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle=True, num_threads=60):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of [NUM_ATTRS] of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Attribute labels. 2D tensor of [batch_size, NUM_ATTRS] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.

  if not shuffle:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 10 * batch_size)
  else:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 20 * batch_size,
        min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  #tf.image_summary('images', images)

  return images, label_batch


def distorted_inputs(dataset_root, txt_fpath, batch_size, shuffle=True, num_threads=60, num_sets=1):
  """Construct distorted input for IMAGENET training using the Reader ops.

  Args:
    data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  for f in [dataset_root, txt_fpath]:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with open(txt_fpath, 'r') as fd:
    num_examples_per_epoch = len(fd.readlines())

  print('\tLoad file list from %s: Total %d files' % (txt_fpath, num_examples_per_epoch))
  print('\t\tBatch size: %d, %d sets of batches, %d threads per batch' % (batch_size, num_sets, num_threads))

  # Read examples from files.
  read_input = read_input_file(txt_fpath, dataset_root, shuffle)

  images_list, labels_list = ([], [])

  for i in range(num_sets):
    distorted_image = tf.cast(read_input.image, tf.float32)
    # distorted_image = tf.Print(distorted_image, [read_input.image_path])

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(distorted_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Preprocess the image
    distorted_image = preprocess_image(distorted_image)

    # Generate a batch of images and labels by building up a queue of examples.
    min_queue_examples = batch_size * 10;
    images, labels = _generate_image_and_label_batch(distorted_image, read_input.labels,
                                           min_queue_examples, batch_size, shuffle, num_threads)
    images_list.append(images)
    labels_list.append(labels)

  return images_list, labels_list



def inputs(dataset_root, txt_fpath, batch_size, shuffle=False, num_threads=60, num_sets=1, center_crop=False):
  """Construct input for IMAGENET evaluation using the Reader ops.

  Args:
    data_class: string, indicating if one should use the 'train' or 'eval' or 'test' data set.
    batch_size: Number of images per batch.

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

  # Read examples from files.
  read_input = read_input_file(txt_fpath, dataset_root, shuffle)

  images_list, labels_list = ([], [])

  for i in range(num_sets):
    image = tf.cast(read_input.image, tf.float32)
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    if not center_crop:
      image = tf.random_crop(image, [height, width, 3])
    else:
      image_shape = tf.shape(image)
      h_offset = tf.cast((image_shape[0]-height)/2, tf.int32)
      w_offset = tf.cast((image_shape[1]-width)/2, tf.int32)
      image = tf.slice(image, [h_offset, w_offset, 0], [height, width, 3])

    # Preprocess the image
    image = preprocess_image(image)

    # Generate a batch of images and labels by building up a queue of examples.
    min_queue_examples = batch_size * 10;
    images, labels = _generate_image_and_label_batch(image, read_input.labels,
                                           min_queue_examples, batch_size, shuffle, num_threads)
    images_list.append(images)
    labels_list.append(labels)

  return images_list, labels_list

