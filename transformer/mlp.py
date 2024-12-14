''' Functions for a MultiLayer Perceptron '''

import numpy as np


class Linear:
    '''
    A single linear layer.
    '''

    def __init__(self, in_features, out_features, bias=True, w=None):
        '''
        Initializes the linear layer.

        Parameters
        ----------
        in_features: int
            The number of input features
        out_features: int
            The number of output features
        bias: bool, optional
            Whether to include bias in the linear layer
        w: array, optional
            The weights of the linear layer, of shape (in_features, out_features)
        '''

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # weights of the linear layer, of shape (in_features, out_features)
        self.w = np.random.randn(in_features, out_features) if type(w) == type(None) else w
        # bias of the linear layer, of shape (out_features)
        self.b = np.random.randn(out_features) if bias else None

    def forward(self, x):
        '''
        Forward pass through the linear layer.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size (optional), seq_len (optional), in_features)

        Returns
        -------
        array
            Output array - of shape (batch_size (optional), seq_len (optional), out_features)
        '''

        output = x @ self.w

        if self.bias:
            output += self.b

        return output