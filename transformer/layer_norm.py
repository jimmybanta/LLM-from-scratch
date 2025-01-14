

import numpy as np

class LayerNorm:
    '''
    Layer Normalization layer.
    '''

    def __init__(self, d_model, 
                 eps=1e-5, 
                 scale_shift=True,
                 gamma=None,
                 beta=None):
        '''
        Initializes the layer.

        Parameters
        ----------
        d_model: int
            The size of word embeddings of the model
        eps: float, optional
            A small value to avoid division by zero
        scale_shift: bool, optional
            Whether to scale and shift the normalized input
        gamma: array, optional
            The gamma parameter, of shape (d_model,)
        beta: array, optional
            The beta parameter, of shape (d_model,)
        '''

        # if scale_shift is True, then we need to initialize gamma and beta
        if scale_shift:

            # if gamma and beta are not provided, initialize them
            if gamma and beta:
                self.gamma = gamma
                self.beta = beta
            else:
                self.gamma = np.ones(d_model)
                self.beta = np.zeros(d_model)
        else:
            self.gamma = None
            self.beta = None
    
        self.eps = eps

    def forward(self, x):
        '''
        Forward pass through the layer norm.
        '''

        # calculate the mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # normalize the input
        x = (x - mean) / np.sqrt(variance + 1e-5)

        # scale and shift
        if self.gamma is not None:
            x = self.gamma * x + self.beta
        
        return x
        



