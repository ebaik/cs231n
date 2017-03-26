import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimensions = W.shape[0]
  dW_temp = np.zeros((num_train,num_dimensions,num_classes)) # use this 3d array to store gradient computations

  for i in xrange(num_train):
    scores = X[i].dot(W)  # shape (C,)
    scores -= np.max(scores)  # shift values fj so that the highest number is 0 for computational stability
    f_yi = scores[y[i]]
    loss += -1*f_yi + np.log(np.sum(np.exp(scores)))
    sum_of_exp = np.sum(np.exp(scores))
    for j in xrange(num_classes):
        if j == y[i]:
            dW_temp[i,:,j] = (np.exp(f_yi) / sum_of_exp - 1) * X[i]
        else:
            dW_temp[i,:,j] = (np.exp(scores[j]) / sum_of_exp) * X[i]
   # analytic gradient for softmax loss: take derivative of L with respect to Wyi and Wj where Wyi, Wj of shape (N,)
   # Li = -fyi + log(sum(exp(fj)))
   # If W = Wyi, then Li = -Wyi^T * Xi + log(sumoverj(exp(Wj^T*Xi)))
   # and D(Li)/D(Wyi) = -Xi + [ exp(Wyi^T*Xi) / sumoverj(exp(Wj^T*Xi)) ] * Xi because D(log(u(x)))/D(x) = [1 / u(x)] * D(u(x)) where log is base e
   # which simplifies to D(Li)/D(Wyi) = [ exp(Wyi^T*Xi)/sumoverj(exp(Wj^T*Xi)) - 1 ] * Xi
   # which is D(Li)/D(Wyi) = [ exp(fyi)/sumoverj(exp(fj)) - 1 ] * Xi
   # If W = Wj, then
   # D(Li)/D(Wj) = 0 + exp(Wj^T*Xi)/sumoverjprime(exp(Wjprime^T*Xi) * Xi
   # which is D(Li)/D(Wj) = 0 + exp(fj)/sumoverjprime(exp(fjprime)) * Xi

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW[:,:] = np.sum(dW_temp,axis=0) / num_train  # also, need to average the regularization over all training samples

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dimensions = W.shape[0]
  #dW_3d = np.zeros((num_train,num_dimensions,num_classes)) # use this array to store gradient computations
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)   # shape (N,C)

  # First, compute Softmax Cross-Entropy Loss
  # shift values fj=scores so that the highest number is 0 for computational stability
  # below: scores shape (N,C)
  # max function results in shape (N,) => requires reshape second array to (N,1) for proper broadcasting in scores computation step
  scores = scores - np.reshape(np.max(scores,axis=1),(num_train,1))  # shape (N,C)
  f_yi = scores[np.arange(num_train),y]  # shape (N,)
  loss = np.mean(-1*f_yi + np.log(np.sum(np.exp(scores),axis=1)))
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Next, compute the gradient
  # first, compute for all classes j; then in next step we'll update correct values for class = yi
  sum_of_exp = np.sum(np.exp(scores),axis=1) # shape (N,) => reshape to (N,1) for proper broadcasting in next step computing dW_temp
  dW_temp = np.exp(scores) / np.reshape(sum_of_exp,(num_train,1))  # shape (N,C)
  dW_3d = dW_temp[:,np.newaxis,:] * X[:,:,np.newaxis]  # shape (N,D,C)
  # now, correct for the right gradient values for class = yi; (N,) * (N,D) to broadcast to (N,D)
  # first, compute the gradient for the correct classes yi for each training sample
  # and store it in the dW gradient matrix properly by replacing values for location yi in C dimension
  dW_temp_yi = (np.exp(f_yi) / sum_of_exp - 1)  # shape (N,)
  dW_3d[np.arange(num_train),np.arange(num_dimensions),y] = np.reshape(dW_temp_yi,(num_train,1)) * X  # shape (N,1) * (N,D) = (N,D)

  dW = np.mean(dW_3d,axis=0)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
