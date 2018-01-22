from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs an affine transform, batch norm,
  then followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  b, bf_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, bf_cache, relu_cache)
  return out, cache


def affine_batchnorm_relu_backward(dout, cache):
  """
  Backward pass for the affine-batchnorm-relu convenience layer
  """
  fc_cache, bf_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  db, dgamma, dbeta = batchnorm_backward(da, bf_cache)
  dx, dw, db = affine_backward(db, fc_cache)
  return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_batchnorm_relu_forward(x, w, b, beta, gamma, bn_param, conv_param):
  """
  A convenience layer that performs a convolution and batch norm, followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - beta, gamma: batch normalization parameters
  - bn_param: dictionary of key:value pairs needed to compute batch norm
  - conv_param: dictionary of stride and pad values to compute convolution

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  out_cf, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out_bf, batch_cache = spatial_batchnorm_forward(out_cf, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out_bf)
  cache = (conv_cache, batch_cache, relu_cache)

  return out, cache


def conv_batchnorm_relu_backward(dout, cache):
  """
  Backward pass for the conv-batchnorm-relu convenience layer.
  """
  conv_cache, batch_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  db, dgamma, dbeta = spatial_batchnorm_backward(da, batch_cache)
  dx, dw, db = conv_backward_fast(db, conv_cache)
  return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_batchnorm_relu_pool_forward(x, w, b, beta, gamma, bn_param, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, batch norm, ReLU, and a max pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - beta, gamma: batch normalization parameters
  - bn_param: dictionary of key:value pairs needed to compute batch norm
  - conv_param: dictionary of stride and pad values to compute convolution
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  out_cff, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out_sbf, batch_cache = spatial_batchnorm_forward(out_cff, gamma, beta, bn_param)
  out_rf, relu_cache = relu_forward(out_sbf)
  out, pool_cache = max_pool_forward_fast(out_rf, pool_param)
  cache = (conv_cache, batch_cache, relu_cache, pool_cache)
  return out, cache


def conv_batchnorm_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-batchnorm-relu-pool convenience layer
  """
  dgamma = None
  dbeta = None

  conv_cache, batch_cache, relu_cache, pool_cache = cache
  dmp = max_pool_backward_fast(dout, pool_cache)
  drl = relu_backward(dmp, relu_cache)
  dsb, dgamma, dbeta = spatial_batchnorm_backward(drl, batch_cache)
  dx, dw, db = conv_backward_fast(dsb, conv_cache)
  return dx, dw, db, dgamma, dbeta
