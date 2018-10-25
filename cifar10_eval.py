# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import cifar10_train
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

resultsFile = open("ShadownNet_mnistLast_20th_40k_Noise.csv",'w')
resultsFile.write("Iteration;Original Accuracy; WM Accuracy\n")

def eval_once(saver, summary_writer, cifar_top_k_op, mnist_top_k_op, summary_op,itercount):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      ckpt_path_and_name = ckpt.model_checkpoint_path.split('-')[0]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      NoiseTest = True
      if NoiseTest:
        # saver is put here to always modify the same net, with different amount of noise! 
        # counter is increased to be able to plot the graph!
        saver.save(sess,ckpt_path_and_name,global_step=int(global_step)+1)
        ## adding noise
        print('Adding noise to the loaded net..')
        train_vars = tf.trainable_variables()
        shared_vars = [var for var in train_vars if 'shared_' in var.name]
        cifar_vars = [var for var in train_vars if 'cifar_' in var.name]
        mnist_vars = [var for var in train_vars if 'mnist_' in var.name]

        for v in shared_vars:
          #print(v)
          v1 = sess.graph.get_tensor_by_name(v.name)
          v_shape = tf.shape(v1)
          l = len(v_shape.eval())
          mean, variance = tf.nn.moments(v1,list(range(l)))
          #mean, variance = tf.nn.moments(v1,[0])
          #print(v.name)
          #print('mean : ', mean.eval())
          #print('vari : ', variance.eval())
          # sqrt(variance)
          noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=tf.sqrt(variance)*0.01*itercount, dtype=tf.float32)
          #noise = tf.random_normal(shape=tf.shape(v1), mean=0.0, stddev=0.01, dtype=tf.float32) 
          sess.run(tf.assign(v1,v1+noise))

      # saving noisy one
      #print("###########")
      #print(ckpt.model_checkpoint_path)
      #print(ckpt_path_and_name)
      #saver.save(sess,ckpt_path_and_name,global_step=int(global_step)+10)


      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      cifar_true_count = 0  # Counts the number of correct predictions.
      cifar_total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        cifar_predictions = sess.run([cifar_top_k_op])
        cifar_true_count += np.sum(cifar_predictions)
        step += 1

      # Compute precision @ 1.
      cifar_precision = cifar_true_count / cifar_total_sample_count
      print('%s: CIFAR precision @ %d = %.3f' % (datetime.now(),int(global_step), cifar_precision))

      mnist_true_count = 0  # Counts the number of correct predictions.
      mnist_total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        mnist_predictions = sess.run([mnist_top_k_op])
        mnist_true_count += np.sum(mnist_predictions)
        step += 1

      # Compute precision @ 1.
      mnist_precision = mnist_true_count / mnist_total_sample_count
      print('%s: MNIST precision @ %d = %.3f' % (datetime.now(),int(global_step), mnist_precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='CIFAR Precision @ 1', simple_value=cifar_precision)
      summary.value.add(tag='MNIST Precision @ 1', simple_value=mnist_precision)
      summary_writer.add_summary(summary, global_step)
      resultsFile.write(str(global_step)+";"+str(cifar_precision)+";"+str(mnist_precision)+"\n")
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return global_step, NoiseTest



def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    cifar_images, cifar_labels = cifar10.inputs(eval_data=eval_data)
    mnist_images, mnist_labels = cifar10.mnist_inputs("test")

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # logits = cifar10.inference(images)
    #logits, mnist_logits = cifar10.inference(mnist_images)

    with tf.variable_scope('shared_net') as scope:
      cifar_local4 = cifar10.inference_shared(cifar_images)
      scope.reuse_variables()
      mnist_local4 = cifar10.inference_shared(mnist_images)
      

    mnist_logits = cifar10.inference_mnist(mnist_local4)
    cifar_logits = cifar10.inference_cifar(cifar_local4)

    #mnist_labels = tf.Print(mnist_labels, [mnist_labels],'*.*.*.* MNIST labels:')
    # cifar_labels = tf.Print(cifar_labels, [cifar_labels],'*.*.*.* CIFAR labels:')

    # Calculate predictions.
    cifar_top_k_op = tf.nn.in_top_k(cifar_logits, cifar_labels, 1)

    mnist_top_k_op = tf.nn.in_top_k(mnist_logits, tf.cast(mnist_labels,dtype=tf.int32), 1)



    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    itercount=0

    max_eval_iters = cifar10_train.FLAGS.max_steps
    while True:
      gs,nt = eval_once(saver, summary_writer, cifar_top_k_op, mnist_top_k_op, summary_op,itercount)
      itercount+=1
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
      # must be set according to training

      if nt:
        max_eval_iters = 101 # 101 % of noise is added if noise test is applied
      print('MAX_EVAL_ITERS :',max_eval_iters)
      if int(gs)>=max_eval_iters:
        print('F I N I S H E D ')
        break


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
