from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

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
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def mixedforwordliner(x, w, b, gamma=None, beta=None, bn_param=None, dropout_param=None, last=False):
    """Convenience layer that performs an affine transform, a batch/layer normalization
    if needed, a ReLU, and dropout if needed.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Scale and shift params for the batch normalization
    - bn_param: Dictionary of required BN parameters
    - dropout_param: Dictionary of required Dropout parameters
    - last: Indicates wether to perform just affine forward

    Returns a tuple of:
    - out: Output from the ReLU or Dropout
    - cache: Object to give to the backward pass
    """
    # Initialize optional caches to None
    bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None

    # Affine forward is a must
    out, fc_cache = affine_forward(x, w, b)

    # If the the layer is not last
    if not last:
        # If it has normalization layer we normalize outputs: if it bn_param
        # has mode (train | test), it's batchnorm, otherwise, it's layernorm
        if bn_param is not None:
            if 'mode' in bn_param:
                out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
            else:
                out, ln_cache = layernorm_forward(out, gamma, beta, bn_param)

        # Pass the outputs through activation
        out, relu_cache = relu_forward(out) # perform relu

        # Use dropout if we are given its parameters
        if dropout_param is not None:
            out, dropout_cache = dropout_forward(out, dropout_param)
    
    # Prepare cache for backward pass
    cache = (fc_cache, bn_cache, ln_cache, relu_cache, dropout_cache)

    return out, cache

def mixedbackwordliner(dout,cache):
    # Initialize optional caches to None
    fc_cache,bn_cache, ln_cache, relu_cache, dropout_cache = cache[0],cache[1],cache[2],cache[3],cache[4]
    #print(fc_cache)
    dx, dgamma, dbeta, dw, db = None,None,None,None,None
    dx = dout
    if dropout_cache is not None:
        dx = dropout_backward(dout,dropout_cache)
    
    if relu_cache is not None:
        dx = relu_backward(dx,relu_cache)
    
    if ln_cache is not None:
        dx, dgamma, dbeta = layernorm_backward(dx,ln_cache)
    
    if bn_cache is not None:
        dx, dgamma, dbeta = batchnorm_backward(dx,bn_cache)

    dx, dw, db = affine_backward(dx,fc_cache)
    
    
    # Prepare cache for backward pass
    grad = (dx, dgamma, dbeta, dw, db)

    return grad

    




# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

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
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

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
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
