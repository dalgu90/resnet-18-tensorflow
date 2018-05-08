#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
from IPython import embed
from tensorflow.python.client import timeline

import imagenet_input as data_input
import resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('train_dataset', 'scripts/train_shuffle.txt', """Path to the ILSVRC2012 the training dataset list file""")
tf.app.flags.DEFINE_string('train_image_root', '/data1/common_datasets/imagenet_resized/', """Path to the root of ILSVRC2012 training images""")
tf.app.flags.DEFINE_string('val_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('val_image_root', '/data1/common_datasets/imagenet_resized/ILSVRC2012_val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 1281167, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_val_instance', 50000, """Number of val images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of GPUs.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "30.0,60.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('val_interval', 1000, """Number of iterations to run a val""")
tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tImageNet training root: %s' % FLAGS.train_image_root)
    print('\tImageNet training list: %s' % FLAGS.train_dataset)
    print('\tImageNet val root: %s' % FLAGS.val_image_root)
    print('\tImageNet val list: %s' % FLAGS.val_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of val images: %d' % FLAGS.num_val_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tNumber of GPUs: %d' % FLAGS.num_gpus)
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per validation: %d' % FLAGS.val_interval)
    print('\tSteps during validation: %d' % FLAGS.val_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        import multiprocessing
        num_threads = multiprocessing.cpu_count() / FLAGS.num_gpus
        print('Load ImageNet dataset(%d threads)' % num_threads)
        with tf.device('/cpu:0'):
            print('\tLoading training data from %s' % FLAGS.train_dataset)
            with tf.variable_scope('train_image'):
                train_images, train_labels = data_input.distorted_inputs(FLAGS.train_image_root, FLAGS.train_dataset
                                               , FLAGS.batch_size, True, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            print('\tLoading validation data from %s' % FLAGS.val_dataset)
            with tf.variable_scope('test_image'):
                val_images, val_labels = data_input.inputs(FLAGS.val_image_root, FLAGS.val_dataset
                                               , FLAGS.batch_size, False, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            tf.summary.image('images', train_images[0][:2])

        # Build model
        lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size/FLAGS.num_gpus for s in lr_decay_steps])
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
        network_val.build_model()
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            print('Load checkpoint %s' % FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)
        elif FLAGS.basemodel:
            # Define a different saver to save model checkpoints
            print('Load parameters from basemodel %s' % FLAGS.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "global_step" in var.name]
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            saver_restore.restore(sess, FLAGS.basemodel)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

        # Training!
        val_best_acc = 0.0
        for step in xrange(init_step, FLAGS.max_steps):
            # val
            if step % FLAGS.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(FLAGS.val_iter):
                    loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                feed_dict={network_val.is_train:False})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= FLAGS.val_iter
                val_acc /= FLAGS.val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                summary_writer.add_summary(val_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
            start_time = time.time()
            # For timeline profiling
            # if step == 153:
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # _, loss_value, acc_value, train_summary_str = \
                        # sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                                 # feed_dict={network_train.is_train:True, network_train.lr:lr_value}
                                 # , options=run_options, run_metadata=run_metadata)
                # # Create the Timeline object, and write it to a json
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('timeline.json', 'w') as f:
                    # f.write(ctf)
                # print('Wrote the timeline profile of %d iter training on %s' %(step, 'timeline.json'))
            # else:
                # _, loss_value, acc_value, train_summary_str = \
                        # sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                                # feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr_value})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            if step % FLAGS.display == 0 or step < 10:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
              char = sys.stdin.read(1)
              if char == 'b':
                embed()


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
