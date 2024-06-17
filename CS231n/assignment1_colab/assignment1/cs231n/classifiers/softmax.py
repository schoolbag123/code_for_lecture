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

    for i in range(0,X.shape[0]):
      scores = X[i].dot(W)
      scores = np.exp(scores)
      totalsum = np.sum(scores)
      sofmaxscores = scores / totalsum
      halfloss  = np.log(sofmaxscores)
      loss += np.sum(-halfloss[y[i]])
      halfdw = sofmaxscores
      halfdw[y[i]] -=1
      ones  = np.ones((X.shape[1],10))
      dW += X[i,:].reshape((X.shape[1],1)) * ones * halfdw


    
    loss = loss / X.shape[0]
    loss += reg * np.sum(W * W)

    dW /= X.shape[0]
    dW += 2 * reg * W


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    scores = np.exp(scores)
    sofmaxscores = scores / np.sum(scores,axis=1,keepdims=True)
    halfloss  = np.log(sofmaxscores)
    loss = np.sum(-halfloss[np.arange(X.shape[0]),y]) / X.shape[0]
    loss += reg * np.sum(W * W)

    halfdw = sofmaxscores
    halfdw[np.arange(X.shape[0]),y] -=1
    # print(halfdw.shape)
    ones  = np.ones((X.shape[0],10,X.shape[1]))
    dW += np.sum(np.swapaxes(np.swapaxes(X.reshape((X.shape[0],X.shape[1],1)), 1 , 2 )* ones * halfdw.reshape((X.shape[0],W.shape[1],1)),1,2),axis=0).reshape(W.shape)
    
    dW /= X.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
