"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

#from tensorflow.examples.tutorials.mnist import input_data

from attacks import fgm
from data import get_data_set, dataset


logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.seq2seq

#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#adv_test_labels = mnist.test.labels[:80]
raw_train = get_data_set("train")
raw_test = get_data_set("test")
train_data = dataset(raw_train)
test_data = dataset(raw_test)


config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])


image_list = tf.unstack(images_ph, 640)
x_bright = [tf.image.rot90(tf.reshape(im, [config.original_size, config.original_size, config.num_channels])) for im in image_list] #.adjust_brightness(im, 0.9)
x_bright = tf.stack(x_bright)
x_bright = tf.reshape(x_bright, [-1, config.original_size * config.original_size *
                            config.num_channels])

labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)

# adversarial setup
inp_adv = tf.placeholder(tf.float32, [None, config.g_size])
use_adv = tf.placeholder(tf.bool)
init_glimpse = tf.cond(use_adv, lambda: tf.identity(inp_adv), lambda: tf.identity(init_glimpse))

# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)

# adversarial shit ~~~~~~~~~
#x_adv = fgm(init_glimpse, softmax, xent, eps=.3)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

#ret = tf.cond(use_adv, core_net(x_adv), lambda: (logits,softmax)) 

#logits, softmax = ret


pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32) # get r=1 for correct classification, 0 otherwise
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std) # aka log pi(u|s,theta) of stochastic policy
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = train_data.num_examples // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in range(n_steps):
    images, labels = train_data.next_batch(config.batch_size)
    # convert to dense labels
    labels = np.argmax(labels, axis=1)
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1])
    labels = np.tile(labels, [config.M])
    loc_net.samping = True
    baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, _ = sess.run(
            [baselines_mse, xent, logllratio,
             reward, loss, learning_rate, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels,
                inp_adv: np.zeros([1,config.g_size]).astype(float),
                use_adv: False
            })
    #adv_val, //advs, 
    '''
    inp_adv: np.zeros([1,config.g_size]).astype(float),#tf.constant(0.0, dtype=tf.float32, shape=[1,config.g_size]),
    use_adv: False
    '''
    if i and i % 100 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))

    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [train_data, test_data]:
        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        correct_cnt_fgsm = 0
        correct_cnt_brt = 0
        num_samples = steps_per_epoch * config.batch_size
        loc_net.sampling = True
        for test_step in range(steps_per_epoch):
          images, labels = dataset.next_batch(config.batch_size)    
          # convert to dense labels
          labels = np.argmax(labels, axis=1)
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1])
          labels = np.tile(labels, [config.M])
        
          softmax_val = sess.run(softmax,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels,
                                     inp_adv: np.zeros([1,config.g_size]).astype(float),
                                     use_adv: False
                                 })
            #inp_adv: np.zeros([1,config.g_size]).astype(float),
                                     #use_adv: False
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
          
          
        acc = correct_cnt / num_samples
                
        if dataset == train_data:
          logging.info('train accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))
      
      # adversarial testing
      images_fgsm_bb = np.genfromtxt('../adv_exs/cifar10_fgsm_adv3.csv', delimiter=',')
      images_fgsm_bb = images_fgsm_bb.reshape(-1,
                                              config.original_size * config.original_size * config.num_channels)
      n_exs = images_fgsm_bb.shape[0]
      print (images_fgsm_bb.shape)
      images_fgsm_bb = np.tile(images_fgsm_bb, [config.M, 1])
      print ('dos: ', images_fgsm_bb.shape)
      #labels_fgsm_bb = adv_test_labels #mnist.test.labels[:n_exs]
      labels_fgsm_bb = np.genfromtxt('../adv_exs/cifar10_fgsm_adv3_labels.csv', delimiter=',')
      #labels_fgsm_bb = np.tile(labels_fgsm_bb, [config.M])
      '''
      images_fgsm = sess.run(x_adv, feed_dict={
                                 images_ph: images,
                                 labels_ph: labels,
                                 inp_adv: np.zeros([1,config.g_size]).astype(float),
                                 use_adv: False
                             })
                            
      # for brightening
      images_bright = sess.run(x_bright, feed_dict={
                                 images_ph: images,
                                 labels_ph: labels,
                                 inp_adv: np.zeros([1,config.g_size]).astype(float),
                                 use_adv: False
                             })

      # reinsert glimpse into images?
      softmax_val_fgsm = sess.run(softmax, feed_dict={
                                 images_ph: images_fgsm_bb, 
                                 labels_ph: labels_fgsm_bb,
                                 inp_adv: np.zeros([1,config.g_size]).astype(float),
                                 use_adv: False
                             })
      softmax_val_fgsm = np.reshape(softmax_val_fgsm,
                                 [config.M, -1, config.num_classes])
      softmax_val_fgsm = np.mean(softmax_val_fgsm, 0)
      pred_labels_val_fgsm = np.argmax(softmax_val_fgsm, 1)
      #pred_labels_val_fgsm = pred_labels_val_fgsm.flatten()
      print ('predicted: ' + str(pred_labels_val_fgsm[:10]))
      print ('true: ' + str(labels_fgsm_bb[:10]))   
      correct_cnt_fgsm += np.sum(pred_labels_val_fgsm == labels_fgsm_bb)
        
      acc_fgsm = correct_cnt_fgsm / float(n_exs)
          
      logging.info('adv test accuracy = {}'.format(acc_fgsm))
    
      # brightness test
      softmax_val_brt = sess.run(softmax, feed_dict={
                                 images_ph: images_bright, 
                                 labels_ph: labels,
                                 inp_adv: np.zeros([1,config.g_size]).astype(float),
                                 use_adv: False
                             })
      softmax_val_brt = np.reshape(softmax_val_brt,
                                 [config.M, -1, config.num_classes])
      softmax_val_brt = np.mean(softmax_val_brt, 0)
      pred_labels_val_brt = np.argmax(softmax_val_brt, 1)
      #pred_labels_val_brt = pred_labels_val_brt.flatten()
      print ('predicted: ' + str(pred_labels_val_brt[:10]))
      print ('true: ' + str(labels[:10]))   
      correct_cnt_brt += np.sum(pred_labels_val_brt == labels_bak)
        
      acc_brt = correct_cnt_brt / float(n_exs)
     
          
      logging.info('brightness test accuracy = {}'.format(acc_brt))
      '''
