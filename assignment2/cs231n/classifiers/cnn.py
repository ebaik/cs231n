import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - [batchnorm] - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, use_batchnorm=False, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.bn_params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    Sc = 1  # stride for conv
    pad = (filter_size - 1) / 2
    Hc = 1 + (H - filter_size + 2 * pad) / Sc
    Wc = 1 + (W - filter_size + 2 * pad) / Sc
    pool_height = 2
    pool_width = 2
    Sp = 2  # stride for max pool
    Hp = 1 + (Hc - pool_height) / 2
    Wp = 1 + (Wc - pool_width) / 2
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    self.bn_params['mode'] = 'train'
    # conv + relu layer
    # input: (N,C,H,W)
    # output: (N,num_filters,Hc,Wc)
    self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['beta'] = np.zeros(num_filters)
    self.params['gamma'] = np.ones(num_filters)
    # max pool + affine layer
    # input: (N,num_filters*Hp*Wp)
    # output: (N,hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*Hp*Wp,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    # affine layer
    # input: (N,hidden_dim)
    # output: (N,num_classes)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
    scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
    names to gradients of the loss with respect to those parameters.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    beta, gamma = self.params['beta'], self.params['gamma']
    reg = self.reg
    N = X.shape[0]
    use_batchnorm = self.use_batchnorm

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing
    self.bn_params['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, beta, gamma, self.bn_params, conv_param, pool_param, use_batchnorm)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores_input = out3 - np.amax(out3,axis=1,keepdims=True) # (N,C)
    exp_scores = np.exp(scores_input)
    scores = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)  # (N,C)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    correct_prob = -np.log(scores[range(N),y])
    data_loss = np.sum(correct_prob) / N  # average over N samples
    # data_loss, dscores = softmax_loss(scores,y)
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
    loss = data_loss + reg_loss

    # compute output gradient of softmax for backpropagation
    dscores = scores
    dscores[range(N),y] -= 1
    dscores = dscores / N

    dout3, dW3, db3 = affine_backward(dscores,cache3)
    dout2, dW2, db2 = affine_relu_backward(dout3,cache2)
    dout1, dW1, db1, dgamma, dbeta = conv_relu_pool_backward(dout2, cache1, use_batchnorm)

    grads['W3'] = dW3 + reg*W3   # don't forget regularization gradient
    grads['b3'] = db3
    grads['W2'] = dW2 + reg*W2   # don't forget regularization gradient
    grads['b2'] = db2
    grads['W1'] = dW1 + reg*W1   # don't forget regularization gradient
    grads['b1'] = db1
    grads['gamma'] = dgamma
    grads['beta'] = dbeta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class ConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:

  {conv - [spatial batch norm] - relu - 2x2 max pool} X L - conv - relu - {affine - [batch norm]} X M - softmax

  where spatial batch norm and vanilla batch norm are optional, and the {...} blocks are
  repeated L and M times respectively

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.

  Learnable parameters are stored in the self.params dictionary and will
  be learned using the Solver class.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    C, H, W = input_dim
    Sc = 1  # stride for conv
    pad = (filter_size - 1) / 2
    Hc = 1 + (H - filter_size + 2 * pad) / Sc
    Wc = 1 + (W - filter_size + 2 * pad) / Sc
    pool_height = 2
    pool_width = 2
    Sp = 2  # stride for max pool
    Hp = 1 + (Hc - pool_height) / 2
    Wp = 1 + (Wc - pool_width) / 2
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    # conv + relu layer
    # input: (N,C,H,W)
    # output: (N,num_filters,Hc,Wc)
    self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    # max pool + affine layer
    # input: (N,num_filters*Hp*Wp)
    # output: (N,hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*Hp*Wp,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    # affine layer
    # input: (N,hidden_dim)
    # output: (N,num_classes)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
    scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
    names to gradients of the loss with respect to those parameters.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    reg = self.reg
    N = X.shape[0]

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores_input = out3 - np.amax(out3,axis=1,keepdims=True) # (N,C)
    exp_scores = np.exp(scores_input)
    scores = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)  # (N,C)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    correct_prob = -np.log(scores[range(N),y])
    data_loss = np.sum(correct_prob) / N  # average over N samples
    # data_loss, dscores = softmax_loss(scores,y)
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
    loss = data_loss + reg_loss

    # compute output gradient of softmax for backpropagation
    dscores = scores
    dscores[range(N),y] -= 1
    dscores = dscores / N

    dout3, dW3, db3 = affine_backward(dscores,cache3)
    dout2, dW2, db2 = affine_relu_backward(dout3,cache2)
    dout1, dW1, db1 = conv_relu_pool_backward(dout2, cache1)

    grads['W3'] = dW3 + reg*W3   # don't forget regularization gradient
    grads['b3'] = db3
    grads['W2'] = dW2 + reg*W2   # don't forget regularization gradient
    grads['b2'] = db2
    grads['W1'] = dW1 + reg*W1   # don't forget regularization gradient
    grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
