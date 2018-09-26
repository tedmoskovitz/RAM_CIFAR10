from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

distributions = tf.contrib.distributions


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  #logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
  eps = .0001
  x1 = sampled + eps
  x2 = sampled - eps # approx pdf
  logll = tf.log((gaussian.cdf(x1) - gaussian.cdf(x2))/(x1-x2) ) 
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]
  return logll

def conv_layer(x, w, b, scope, stride=1, padding='VALID'):
  """
  A convolutional layer. 
  
  Args: 
      x: previous layer activation (Tensor, float)
      scope: layer scope
      layer_num: position of layer in network (int)
      ksize: size of convolutional kernel (int)
      out_: number of kernels (int)
      stride: convolutional stride length (int)
      padding: input padding (string)
      
  Returns: 
      h: layer activation (Tensor, float)
  """
  u = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding) + b
  h = tf.nn.relu(u, name=scope)
  return h
