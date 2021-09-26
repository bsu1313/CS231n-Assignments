from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_data = X.shape[0]
    num_class = W.shape[1]
    

    loss_sum =0
    for i in range(num_data):
      s = X[i].dot(W)
      s = s - np.max(s)
      exp_s = np.exp(s)
      exp_s_sum = exp_s.sum()  # scalar
      softmax = exp_s / exp_s_sum
      loss_sum += -np.log(softmax[y[i]])

      for j in range(num_class):
        if j == y[i]:
          continue

        dW[:, j] += (X[i]*softmax[j])
      
      dW[:, y[i]] += (X[i]*(softmax[y[i]]-1))


  
    loss = loss_sum / num_data
    dW /= num_data

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****
    
    num_data = X.shape[0]
    score = X.dot(W)
    score -= np.max(score, axis=1).reshape(-1,1)
    exp_s = np.exp(score)
    exp_sum = np.sum(exp_s, axis = 1).reshape(-1,1)
    softmax = exp_s / exp_sum
    loss_sum = -np.log(softmax[np.arange(num_data),y]).sum()
    loss = loss_sum/num_data
    loss += reg * np.sum(W * W)

    softmax[np.arange(num_data), y] -= 1

    dW = (X.T.dot(softmax)) / num_data
    dW += reg * 2 * W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
