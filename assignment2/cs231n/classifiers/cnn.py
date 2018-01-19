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

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer: pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(3)]

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
    self.params['beta'] = np.zeros(num_filters)
    self.params['gamma'] = np.ones(num_filters)
    # max pool + affine layer
    # input: (N,num_filters*Hp*Wp)
    # output: (N,hidden_dim)
    self.params['W2'] = weight_scale*np.random.randn(num_filters*Hp*Wp,hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)
    self.params['gamma2'] = np.ones(hidden_dim)
    # affine layer
    # input: (N,hidden_dim)
    # output: (N,num_classes)
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes)
    self.params['beta3'] = np.zeros(num_classes)
    self.params['gamma3'] = np.ones(num_classes)
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
    beta2, gamma2 = self.params['beta2'], self.params['gamma2']
    beta3, gamma3 = self.params['beta3'], self.params['gamma3']
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
    if use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    # dictionary with batch norm parameters needed for each separate layer
    bn_param1 = self.bn_params[0]
    bn_param2 = self.bn_params[1]
    bn_param3 = self.bn_params[2]

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if use_batchnorm:
        out_crpf, cache_crpf = conv_batchnorm_relu_pool_forward(X, W1, b1, beta, gamma, bn_param1, conv_param, pool_param)
        out_af, cache_af = affine_forward(out_crpf, W2, b2)
        out_bf, cache_bf = batchnorm_forward(out_af, gamma2, beta2, bn_param2)
        out_rf, cache_rf = relu_forward(out_bf)
        out_af2, cache_af2 = affine_forward(out_rf, W3, b3)
        scores, cache_final = batchnorm_forward(out_af2, gamma3, beta3, bn_param3)
    else:
        out_crpf, cache_crpf = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_arf, cache_arf = affine_relu_forward(out_crpf, W2, b2)
        scores, cache_final = affine_forward(out_arf, W3, b3)

    scores_norm = scores - np.amax(scores,axis=1,keepdims=True) # (N,C)
    exp_scores = np.exp(scores_norm)
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)  # (N,C)
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
    correct_log_prob = -np.log(probs[range(N),y])
    data_loss = np.sum(correct_log_prob) / N  # average over N samples
    # data_loss, dscores = softmax_loss(scores,y)
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
    loss = data_loss + reg_loss

    # compute output gradient of softmax for backpropagation
    dscores = probs
    dscores[range(N),y] -= 1
    dscores = dscores / N

    if use_batchnorm:
        dout_final, dgamma_final, dbeta_final = batchnorm_backward_alt(dscores, cache_final)
        dout_ab, dW3, db3 = affine_backward(dout_final,cache_af2)
        dout_rb = relu_backward(dout_ab,cache_rf)
        dout_bb, dgamma_bb, dbeta_bb = batchnorm_backward_alt(dout_rb, cache_bf)
        dout_ab, dW2, db2 = affine_backward(dout_bb,cache_af)
        dout1, dW1, db1, dgamma, dbeta = conv_batchnorm_relu_pool_backward(dout_ab, cache_crpf, use_batchnorm)
        # save batchnorm gradients from backpropagation
        grads['gamma'] = dgamma
        grads['gamma2'] = dgamma_bb
        grads['gamma3'] = dgamma_final
        grads['beta'] = dbeta
        grads['beta2'] = dbeta_bb
        grads['beta3'] = dbeta_final
    else:
        dout_final, dW3, db3 = affine_backward(dscores, cache_final)
        dout_ab, dW2, db2 = affine_relu_backward(dout_final, cache_arf)
        dout1, dW1, db1, dgamma, dbeta = conv_relu_pool_backward(dout_ab, cache_crpf)

    # save gradients from backpropagation
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


class ConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:

  {conv - [spatial batch norm] - relu - 2x2 max pool} X L -
  conv - [spatial batch norm] - relu -
  {affine - [batch norm] - relu} X M
  - affine - softmax

  where spatial batch norm and vanilla batch norm are optional, and the {...} blocks are
  repeated L and M times respectively

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.

  Learnable parameters are stored in the self.params dictionary and will
  be learned using the Solver class.
  """

  def __init__(self, input_dim=(3, 32, 32), L=1, num_filters=32, filter_size=7,
               hidden_dims=[100], num_classes=10, weight_scale=1e-3, use_batchnorm=False, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dims: List with number of units to use in the fully-connected hidden affine layers; default is list of one item
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.L = L
    self.M = len(hidden_dims)
    self.num_layers = L + 1 + M + 1
    self.conv_params = {}  # to keep track of input data dimension changes by max pooling through conv layers
    self.pool_params = {}  # to keep track of and pass max pool parameters

    # initialize output sizes after 1st conv and max pool layers
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

    # for passing conv filter parameters to the forward pass
    self.conv_params['stride'] = Sc, self.conv_params['pad'] = pad
    # for passing pool_param to the forward pass for the max-pooling layer
    self.pool_params = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': Sp}

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer: pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(L+1+M)]

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the first convolutional layer using the     #
    # keys 'W1' and 'b1'; for the second layer use keys 'W2' and 'b2' for the  #
    # weights and biases, etc.                                                 #
    # Continue this pattern through the affine layers using keys 'WX' and 'bX' #
    # where X = the layer number for the weights and biases                    #
    # of the output affine layer.                                              #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    # initialize the conv layers
    for idx in xrange(L+1):
        w_str = "W%s" % (idx+1)
        b_str = "b%s" % (idx+1)
        H_str = "H%s" % (idx+1)  # height of input to conv layer
        W_str = "W%s" % (idx+1)  # width of input to conv layer
        Hc_str = "Hc%s" % (idx+1)
        Wc_str = "Wc%s" % (idx+1)
        Hp_str = "Hp%s" % (idx+1)
        Wp_str = "Wp%s" % (idx+1)
        self.params[w_str] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
        self.params[b_str] = np.zeros(num_filters)
        # keep track of changing conv layer input/output dimensions due to max pooling
        if idx == 0:
            self.conv_params[H_str] = H
            self.conv_params[W_str] = W
            self.conv_params[Hc_str] = Hc
            self.conv_params[Wc_str] = Wc
            self.conv_params[Hp_str] = Hp
            self.conv_params[Wp_str] = Wp
        else:
            self.conv_params[H_str] = self.conv_params["Hp%s" % idx]
            self.conv_params[W_str] = self.conv_params["Wp%s" % idx]
            self.conv_params[Hc_str] = 1 + (self.conv_params[H_str] - filter_size + 2 * pad) / Sc
            self.conv_params[Wc_str] = 1 + (self.conv_params[W_str] - filter_size + 2 * pad) / Sc
            self.conv_params[Hp_str] = 1 + (self.conv_params[Hc_str] - pool_height) / 2
            self.conv_params[Wp_str] = 1 + (self.conv_params[Wc_str] - pool_width) / 2
        if use_batchnorm:
            gamma_str = "gamma%s" % (idx+1)
            beta_str = "beta%s" % (idx+1)
            self.params[gamma_str] = np.ones(num_filters)
            self.params[beta_str] = np.zeros(num_filters)

    # initialize the affine layers
    for idx_params, hidden_dim in enumerate(hidden_dims+1):  # hidden_dims = M so +1 for last affine layer
        idx = L + 1 + idx_params + 1
        w_str = "W%s" % (idx+1)
        b_str = "b%s" % (idx+1)
        if self.use_batchnorm:
            gamma_str = "gamma%s" % (idx+1)
            beta_str = "beta%s" % (idx+1)
        if idx_params == 0:  # first affine layer
            self.params[w_str] = weight_scale*np.random.randn(num_filters*Hp_str*Wp_str,hidden_dim)
            self.params[b_str] = np.zeros(hidden_dim)
            if self.use_batchnorm:
                self.params[gamma_str] = np.ones(hidden_dim)
                self.params[beta_str] = np.zeros(hidden_dim)
        elif idx_params == len(hidden_dims)-1:   # affine layer M
            prev_layer = hidden_dims[idx_params-1]
            self.params[w_str] = weight_scale*np.random.randn(prev_layer,hidden_dim)
            self.params[b_str] = np.zeros(hidden_dim)
            if self.use_batchnorm:
                self.params[gamma_str] = np.ones(hidden_dim)
                self.params[beta_str] = np.zeros(hidden_dim)
        else:  # last affine layer M+1
            prev_layer = hidden_dims[idx_params-1]
            self.params[w_str] = weight_scale*np.random.randn(prev_layer,num_classes)
            self.params[b_str] = np.zeros(num_classes)
            if self.use_batchnorm:
                self.params[gamma_str] = np.ones(num_classes)
                self.params[beta_str] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the multi-layer convolutional network:

    {conv - [spatial batch norm] - relu - 2x2 max pool} X L -
    conv - [spatial batch norm] - relu -
    {affine - [batch norm] - relu} X M-1 - affine - softmax

    where spatial batch norm and vanilla batch norm are optional, and the {...} blocks are
    repeated L and M times respectively

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
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    reg = self.reg
    N = X.shape[0]
    use_batchnorm = self.use_batchnorm
    cache_affine_history = []
    cache_relu_history = []
    cache_batchnorm_history = []
    cache_conv_history = []

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing
    if use_batchnorm:
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

    ############################################################################
    # TODO: Implement the forward pass for the L+1 layers of convolutional     #
    # net and M affine layers, computing the class scores for X and storing    #
    # them in the scores variable.                                             #
    ############################################################################

    # loop over each layer, setup initialization and compute the forward pass
    for layer_idx in xrange(self.num_layers):
        w_str, b_str = "W%s" % (layer_idx + 1), "b%s" % (layer_idx + 1)
        wght, b = self.params['w_str'], self.params['b_str']

        # add the regularization loss
        reg_loss += 0.5*reg*np.sum(wght*wght)

        if use_batchnorm and layer_idx < self.num_layers-1:
            gamma_str = "gamma%s" % (layer_idx + 1)
            beta_str = "beta%s" % (layer_idx + 1)
            gamma, beta = self.params[gamma_str], self.params[beta_str]
            bn_param = self.bn_params[layer_idx]

        if layer_idx == 0:  # first conv layer
            layer_input = X
            if use_batchnorm:
                out, cache = conv_batchnorm_relu_pool_forward(layer_input, wght, b, beta, gamma, bn_param, self.conv_params, pool_param)
            else:
                out, cache = conv_relu_pool_forward(layer_input, wght, b, self.conv_params, pool_param)
            cache_conv_history.append(cache)
        elif layer_idx > 0 and layer_idx < self.L:  # next L-1 conv layers
            layer_input = out
            if use_batchnorm:
                out, cache = conv_batchnorm_relu_pool_forward(layer_input, wght, b, beta, gamma, bn_param, self.conv_params, pool_param)
            else:
                out, cache = conv_relu_pool_forward(layer_input, wght, b, self.conv_params, pool_param)
            cache_conv_history.append(cache)
        elif layer_idx == self.L:  # last "L+1" conv layer without max pooling
            layer_input = out
            if use_batchnorm:
                out, cache = conv_batchnorm_relu_forward(layer_input, wght, b, beta, gamma, bn_param, self.conv_params)
            else:
                out, cache = conv_relu_forward(layer_input, wght, b, self.conv_params)
            cache_conv_history.append(cache)
        elif layer_idx > self.L and layer_idx < self.num_layers-1:  # first M-1 affine layers
            layer_input = out
            if use_batchnorm:
                out, cache = affine_batchnorm_relu_forward(layer_input, wght, b, gamma, beta, bn_param)
                cache_affine_history.append(cache)
            else:
                out, cache = affine_relu_forward(layer_input, wght, b)
                cache_affine_history.append(cache)
        else:  # last affine layer M; no batchnorm
            layer_input = out
            scores, cache = affine_forward(layer_input, wght, b)
            cache_affine_history.append(cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the multi-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # after last layer in for loop, compute softmax loss
    data_loss, dsoftmax = softmax_loss(scores, y)
    loss = data_loss + reg_loss

    for backprop_idx in xrange(self.num_layers-1,-1,-1):
        # set strings for keys for grads dictionary values
        w_str, b_str = "W%s" % (backprop_idx+1), "b%s" % (backprop_idx+1)
        wght = self.params['w_str']
        if use_batchnorm:
            gamma_str, beta_str = "gamma%s" % (backprop_idx+1), "beta%s" % (backprop_idx+1)

        # for each loop, set grads dictionary values based on layer of the network
        if backprop_idx == self.num_layers-1:  # last affine layer M+1; no batchnorm
            dx, dw, db = affine_backward(dsoftmax, cache_affine_history[backprop_idx])
            grads[w_str], grads[b_str] = dw + reg * wght, db
        elif backprop_idx > self.L and backprop_idx < self.num_layers-1:  # first M affine layers
            if use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dx, cache_affine_history[backprop_idx])
                grads[w_str], grads[b_str], grads[gamma_str], grads[beta_str] = dw + reg * wght, db, dgamma, dbeta
            else:
                dx, dw, db = affine_relu_backward(dx, cache_affine_history[backprop_idx])
                grads[w_str], grads[b_str] = dw + reg * wght, db
        elif backprop_idx == self.L:   # last "L+1" conv layer without max pooling
            if use_batchnorm:
                dx, dw, db, dgamma, dbeta = conv_batchnorm_relu_backward(dx, cache_conv_history[backprop_idx])
                grads[w_str], grads[b_str], grads[gamma_str], grads[beta_str] = dw + reg * wght, db, dgamma, dbeta
            else:
                dx, dw, db = conv_relu_backward(dx, cache_conv_history[backprop_idx])
                grads[w_str], grads[b_str] = dw + reg * wght, db
        elif backprop_idx >= 0 and backprop_idx < self.L:  # next L conv layers
            if use_batchnorm:
                dx, dw, db, dgamma, dbeta = conv_batchnorm_relu_pool_backward(dx, cache_conv_history[backprop_idx])
                grads[w_str], grads[b_str], grads[gamma_str], grads[beta_str] = dw + reg * wght, db, dgamma, dbeta
            else:
                dx, dw, db = conv_relu_pool_backward(dx, cache_conv_history[backprop_idx])
                grads[w_str], grads[b_str] = dw + reg * wght, db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
