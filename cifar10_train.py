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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")
# 30000 iteration for cifar
# 10000 iteration for mnist

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    print('init training')
    global_step = tf.train.get_or_create_global_step()
    print('create global step')
    t1=time.time()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      cifar_images, cifar_labels = cifar10.distorted_inputs()
      mnist_images, mnist_labels = cifar10.mnist_inputs("train")

    # Build a Graph that computes the logits predictions from the
    # inference model.

    with tf.variable_scope('shared_net') as scope:
      cifar_local4 = cifar10.inference_shared(cifar_images)
      scope.reuse_variables()
      mnist_local4 = cifar10.inference_shared(mnist_images)
      

    mnist_logits = cifar10.inference_mnist(mnist_local4)
    cifar_logits = cifar10.inference_cifar(cifar_local4)

    #mnist_labels = tf.Print(mnist_labels, [mnist_labels],'*.*.*.* MNIST labels:')
    #cifar_labels = tf.Print(cifar_labels, [cifar_labels],'*.*.*.* CIFAR labels:')


    #logits, mnist_logits = cifar10.inference(mnist_images)

    #logits, _ = cifar10.inference(images)
    #_, mnist_logits = cifar10.inference(mnist_images)



    # Calculate loss.
    with tf.variable_scope('cifar_losses'):
      cifar_loss = cifar10.loss(cifar_logits, cifar_labels)
    with tf.variable_scope('mnist_losses'):
      mnist_loss = cifar10.loss(mnist_logits,mnist_labels,lossname='mnist_losses')
    ct=time.time()
    print('From start to define losses: ',ct-t1,' sec')



    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.

    # Get variables
    train_vars = tf.trainable_variables()
    shared_vars = [var for var in train_vars if 'shared_' in var.name]
    cifar_vars = [var for var in train_vars if 'cifar_' in var.name]
    mnist_vars = [var for var in train_vars if 'mnist_' in var.name]

    # print('SHAREDSHAREDSHAREDSHARED:')
    # for i in shared_vars:
    #   print(i)
    # print('CIFARCIFARCIFARCIFARCIFARCIFAR:')
    # for i in cifar_vars:
    #   print(i)
    # print('MNISTMNISTMNIST:')
    # for i in mnist_vars:
    #   print(i)


    with tf.name_scope('cifar_train'):
      cifar_train_op = cifar10.train(cifar_loss, global_step,var_list=shared_vars+cifar_vars)

    with tf.name_scope('mnist_train'):
      mnist_train_op = cifar10.train(mnist_loss, global_step,var_list=mnist_vars)
      #mnist_train_op = cifar10.train(mnist_loss, global_step,var_list=shared_vars+mnist_vars)
    ct2=time.time()
    print('From loss to define trainer ops: ',ct2-ct,' sec')

    # # trying to run as simple session...
    # conf = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    # with tf.Session(config=conf) as sess:
    #   sess.run(tf.global_variables_initializer())
    #   print('Ready for training...')
    #   for i in range(FLAGS.max_steps):
    #     print(i)
    #     sess.run(mnist_train_op)
    #     if i% FLAGS.log_frequency:
    #       print('step %d, training loss %g' % (i, mnist_loss))


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        #print('beforeee')
        return tf.train.SessionRunArgs([cifar_loss, mnist_loss])  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          cifar_loss_value = run_values.results[0]
          mnist_loss_value = run_values.results[1]
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, CIFAR loss = %.2f ; MNIST loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, cifar_loss_value, mnist_loss_value,
                               examples_per_sec, sec_per_batch))
    print('MonitoredTrainingSession is about to start')
    ct3=time.time()
    print('From trainer op to MTS strat: ',ct3-ct2,' sec')
    saver = tf.train.Saver()

    #with tf.train.SingularMonitoredSession(
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(cifar_loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement),
        save_checkpoint_secs=5000) as mon_sess:
      StepCount = 1
      dt=time.time()
      print('With MTS as mon_sess.. :',dt-ct, 'sec')
      print('while is coming')
      while not mon_sess.should_stop():
        if StepCount == 1:
          print('First cycle...')
          et=time.time()
          print('From MTS to first cycle :',et-dt,' sec')
        bt=time.time()
        mon_sess.run(cifar_train_op)
        bt2=time.time()
        if StepCount<10:
          print('Single session run :', bt2-bt, 'sec')
        #if (not mon_sess.should_stop()) and ((StepCount%20)==0) and StepCount<8000:
        if (not mon_sess.should_stop()) and ((StepCount%20)==0) and StepCount<=40000:
        # if (not mon_sess.should_stop()) and StepCount<40000:
          mon_sess.run(mnist_train_op)

        #if StepCount==50 or StepCount==100 or StepCount==500 or StepCount==1000 or StepCount==2000 or StepCount==4000 or StepCount==8000 or StepCount==10000 or StepCount==40000:
        #if (StepCount%50)==0:
        #  saver.save(mon_sess._sess._sess._sess._sess,'/tmp/cifar10_train/model.ckpt',global_step=StepCount)
          #time.sleep(4)
        if StepCount==40000:
          saver.save(mon_sess._sess._sess._sess._sess,'/tmp/cifar10_train/model.ckpt',global_step=StepCount)

        StepCount+=1



def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()


## ToDo:

# - fix step count @ hook (when called by other trainer)
# - nyehh.. mnistLats was @ each 20th iteration...
# - proper learning rate?
# - speed up lr decay ?
# - only loss and accuracy @ summary
# - Adamoptimizer
# - optimizers? lr? decay?



## ToDo (NiceToHave) :

# - np.reshape vs tf.reshape
# - maybe_download_mnist data! and save in disc
# - resize mnist data ?
# - mnist resize with cv?
# - - save images and load images as well...



## Done
# - create mnist FC layer
# - import mnist data
# - filter mnist data
# - get mnist data in batches
# - simple lables, not one_hot for current framework
# - get mnist variables
# - define mnist loss
# - define mnist optimizer
# - mnist data size for epoch and other constants!
# - train on mnist only
# - evaluate mnist only
# - why does cifar_loss ruins mnist training?
# - how to call inference with the same weights, but different inputs?
# - mnist complete training
# - cifar complete training
# - original ciafr on single GPU
# - faster init/ startup
# - both losses from hook
# - 10,20 nth iter
# - update evaluation function for both cifar and mnist
# - combine learning
# - eval for both
# - checkpoint freq?
# - eval freq ?
# - combine train and eval
# - train mnist @ each 5-10-20-40 iteration

# - save 8k iter of both shared and mnistLast NETs
# - - save mnist = shared + mnist @8k
# - - save mnist = mnist @8k
# - parameters to Sanyi
# - 100 iteracionkent checkpoint es eval
# - csv @ eval
# - 40k tol tovabbtanitas
# - is noise runs twivce ? Nope

# - save/load net + add diffetent amount of noise + evaluate only!
# - 0) load net
# - 0.1) load noisy net
# - 1) add noise to the net
# - 2) eval net
# - 3) save noisy net-> 0.1)