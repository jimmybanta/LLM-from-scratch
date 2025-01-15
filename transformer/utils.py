''' Utility functions to use in implementing the transformer. '''

import numpy as np


def softmax(x, axis=2, temperature=1.0):
    '''
    Compute the softmax values.
    Shifts the array to avoid overflow.
    '''

    # if temperature is 0, set it to a very small value
    if temperature == 0:
        temperature = 0.00000000000001

    # get the max value - to shift everything by
    x_max = np.amax(x, axis=axis, keepdims=True)

    # exponentiate all shifted values, divided by the temperature
    exp_shifted = np.exp((x - x_max) / temperature)

    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

def relu(x):
    '''
    Calculate rectified linear unit (ReLU) of a vector.
    '''

    # set all negative values to zero
    x[x < 0] = 0

    return x




