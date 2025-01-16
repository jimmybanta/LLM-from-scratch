''' Contains code for loss functions. '''


import numpy as np

def cross_entropy_loss(preds, targets, padding_mask=None):
    ''' Compute the cross entropy loss. 
    
    Parameters
    ----------
    preds : np.ndarray
        The predicted probabilities.
    targets : np.ndarray
        The true labels.
    padding_mask : np.ndarray, optional
        The mask specifying which tokens are padding tokens.

    Returns
    -------
    float
        The cross entropy loss.
    '''

    # calculate the inner values
    loss_values = targets * np.log(preds)

    # mask out padding tokens
    ## set loss values to 0 for every position with a padding token
    ## we aren't predicting off the padding tokens
    if padding_mask is not None:
        loss_values[padding_mask] = 0

    return -np.sum(loss_values)