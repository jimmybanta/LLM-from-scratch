''' Utility functions to use in implementing the transformer. '''

import numpy as np


def softmax(x, axis=2):
    '''
    Compute the softmax values.
    Shifts the array to avoid overflow.
    '''

    # get the max value - to shift everything by
    x_max = np.amax(x, axis=axis, keepdims=True)

    # exponentiate all shifted values
    exp_shifted = np.exp(x - x_max)

    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

def relu(x):
    '''
    Calculate rectified linear unit (ReLU) of a vector.
    '''

    # set all negative values to zero
    x[x < 0] = 0

    return x




