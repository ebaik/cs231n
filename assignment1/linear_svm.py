import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimensions = W.shape[0]
  dW_temp = np.zeros((num_train,num_dimensions,num_classes))
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    grad_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        grad_count += 1
        loss += margin
        dW_temp[i,:,j] = X[i] + 2 * W[:,j]
    dW_temp[i,:,y[i]] = -1 * grad_count * X[i] + 2 * W[:,y[i]]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW[:,:] = np.sum(dW_temp,axis=0) / num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero, shape (D, C)
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  # A numpy array of shape (N, C)
  correct_class_score = scores[np.arange(num_train),y]  # A numpy array of shape (N,) picking the correct scores using y
  correct_class_score = np.reshape(correct_class_score, (num_train,1))  # Reshape to (N,1) for broadcasting in next step
  margins = np.maximum(0,scores - correct_class_score + 1)  # shape (N, C)
  # on y-th position scores[:,y] - correct_class_score canceled and gave 1. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[np.arange(num_train),y] = 0

  loss = np.sum(margins,axis=1)
  loss = np.mean(loss,axis=0)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  idx_gradient = margins > 0
  grad_count = np.copy(margins)  # shape (N,C)
  grad_count[idx_gradient] = 1
  grad_count_y = np.sum(grad_count,axis=1)  # shape (N,)
  grad_count[np.arange(num_train),y] = -1 * grad_count_y
  dW_X = grad_count[:,np.newaxis,:] * X[:,:,np.newaxis]

  dW_temp = dW_X + 2 * W
  dW = np.mean(dW_temp,axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
