class Config(object):
  win_size = 8
  bandwidth = win_size * win_size
  batch_size = 64
  eval_batch_size = 50
  loc_std = 0.22
  original_size = 32
  num_channels = 3
  depth = 3
  sensor_size = win_size * win_size * depth
  minRadius = 8
  hg_size = hl_size = 100 #128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256
  cell_out_size = cell_size
  num_glimpses = 6
  num_classes = 10
  max_grad_norm = 5.
  kernel_size = 3
  gnet_cnn_depth = 3
  gnet_fc_depth = 2
  step = 200000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10
