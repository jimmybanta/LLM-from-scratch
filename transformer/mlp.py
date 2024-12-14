''' Functions for a MultiLayer Perceptron '''

import numpy as np
from transformer.utils import relu


class Linear:
    '''
    A single linear layer.
    '''

    def __init__(self, in_features, out_features, bias=True, 
                 w=None, b=None):
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
        b: array, optional
            The biases of the linear layer, of shape (out_features,)
        '''

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # weights of the linear layer, of shape (in_features, out_features)
        self.w = np.random.randn(in_features, out_features) if type(w) == type(None) else w
        # bias of the linear layer, of shape (out_features)
        if bias:
            if type(b) == type(None):
                self.b = np.random.randn(out_features)
            else:
                self.b = b

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
    

class TwoLayerMLP:
    '''
    A Two-Layer Multi-Layer perceptron, with a ReLU in between.
    '''

    def __init__(self, 
                    input_dim=512,
                    hidden_dim=2048,
                    bias=True,
                    layers_w=None,
                    layers_b=None
                 ):
        '''
        Initialize the MLP.
        '''

        self.layer_one = Linear(in_features=input_dim,
                                out_features=hidden_dim,
                                bias=bias,
                                w=layers_w[0] if layers_w else None,
                                b=layers_b[0] if layers_b else None)
        
        self.layer_two = Linear(in_features=hidden_dim, 
                                out_features=input_dim,
                                bias=bias,
                                w=layers_w[1] if layers_w else None,
                                b=layers_b[1] if layers_b else None)


    def forward(self, x):
        '''
        Runs an input through the MLP.
        '''

        
        x = self.layer_one.forward(x)
        x = relu(x)
        x = self.layer_two.forward(x)
        
        return x
