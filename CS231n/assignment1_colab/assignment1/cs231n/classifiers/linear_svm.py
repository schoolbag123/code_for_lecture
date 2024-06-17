from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                #print(dW.shape,X[i].T.shape)
                dW[:,j] += X[i].T
                dW[:,y[i]] -= X[i].T
    
    dW = dW / num_train

    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero



    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X.dot(W)
    correct_class_score = scores[range(0,num_train),y]
    #print(scores.shape,correct_class_score.shape)
    #margin = np.max(np.zeros(scores.shape), scores - correct_class_score.reshape((scores.shape[0],1)) + 1)  # note delta = 1
    margin = scores - correct_class_score.reshape((scores.shape[0],1)) + 1
    margin = np.where(margin > 0, margin, 0)
    margin[np.arange(num_train),y] = 0
    #print(margin[:10,:])
    loss =np.sum(margin) / num_train
    loss += reg * np.sum(W * W)


    dW_ = np.zeros((X.shape[0],X.shape[1],W.shape[1]))
    dW_ += X.reshape((dW_.shape[0],dW_.shape[1],1))
    # print(dW_.shape)
    noinfluence = (margin != 0)
    # print(y)
    # print(noinfluence.shape)
    # print(noinfluence[:10,:6])
    # print(np.swapaxes(dW_, 1, 2).shape)
    dW_ = np.swapaxes(np.swapaxes(dW_, 1, 2) * noinfluence.reshape(dW_.shape[0],dW_.shape[2],1) , 1 , 2)
    # print(dW_.shape)
    dW_ = dW_.sum(axis=0)
    # print(dW_.shape)
    dW += dW_

    noinfluencesum = np.sum(noinfluence,axis=1)
    dW_2 = np.zeros((X.shape[0],X.shape[1],W.shape[1]))
    dW_2 += X.reshape((dW_2.shape[0],dW_2.shape[1],1))
    YY = np.zeros((X.shape[0],W.shape[1]))
    YY[np.arange(num_train),y] = 1
    YY = YY * noinfluencesum.reshape((YY.shape[0],1))
    # print(YY[:10,:])

    dW_2 = np.swapaxes(np.swapaxes(dW_2, 1, 2) * YY.reshape((dW_2.shape[0],dW_2.shape[2],1)) , 1 , 2)
    dW_2 = dW_2.sum(axis=0)
    dW -= dW_2

    dW = dW / num_train

    dW += 2 * reg * W


    return loss, dW.reshape((dW.shape[0],dW.shape[1]))
