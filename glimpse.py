from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable, conv_layer


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth

    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim
    self.k_size = config.kernel_size

    self.images_ph = images_ph
    self.gcnn_depth = config.gnet_cnn_depth
    self.gfc_depth = config.gnet_fc_depth
    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    #self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
    #self.b_g0 = bias_variable((self.hg_size,))
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))
    self.w_l1 = weight_variable((self.hl_size, self.g_size))
    self.b_l1 = weight_variable((self.g_size,))
    
    # cnn params
    self.glimpse_params = {}
    n_channels = self.depth
    for i in range(1,self.gcnn_depth+1):
        self.glimpse_params['w_gconv{}'.format(i)] = weight_variable((self.k_size, self.k_size,
                                                                  n_channels, self.hg_size))
        self.glimpse_params['b_gconv{}'.format(i)] = bias_variable((self.hg_size,))
        n_channels = self.hg_size
    
    in_size = self.hg_size * self.win_size  * self.win_size
    out_size = 100
    for i in range(1,self.gfc_depth+1):
        if i == self.gfc_depth:
            self.glimpse_params['w_gfc{}'.format(i)] = weight_variable((in_size, self.g_size))
            self.glimpse_params['b_gfc{}'.format(i)] = bias_variable((self.g_size,))
        else:
            self.glimpse_params['w_gfc{}'.format(i)] = weight_variable((in_size, out_size)) 
            self.glimpse_params['b_gfc{}'.format(i)] = bias_variable((out_size,))  
        in_size = out_size
 

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,
        self.num_channels
    ])
    glimpse_imgs = tf.image.extract_glimpse(imgs,
                                            [self.win_size, self.win_size], loc)
    glimpse_imgs = tf.reshape(glimpse_imgs, [
        tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
    ])
    return glimpse_imgs

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)
    #glimpse_input = tf.reshape(glimpse_input,
     #                          (tf.shape(loc)[0], self.sensor_size))
    glimpse_input = tf.reshape(glimpse_input,
                               (tf.shape(loc)[0], self.win_size, self.win_size, self.depth))
    
    #g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0)) # replace with a CNN?
    # CNN
    g = glimpse_input
    for i in range(1,self.gcnn_depth+1):
        w = self.glimpse_params['w_gconv{}'.format(i)]
        b = self.glimpse_params['b_gconv{}'.format(i)]
        g = conv_layer(g, w, b, 'glimpse_conv{}'.format(i),
                       stride=1, padding='SAME')
    
    g = tf.reshape(g, (tf.shape(g)[0], -1))
    for i in range(1,self.gfc_depth+1):
        w = self.glimpse_params['w_gfc{}'.format(i)]
        b = self.glimpse_params['b_gfc{}'.format(i)] 
        g = tf.nn.xw_plus_b(g, w, b) 
    
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0)) # what's the point other than transforming the dim?
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g


class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.

  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
