''' Contains the code for a transformer block. '''


import numpy as np
from transformer.attention import MultiHeadAttention
from transformer.mlp import TwoLayerMLP
from transformer.layer_norm import LayerNorm
from transformer.utils import dropout



class TransformerBlock:
    '''
    A single transformer block.
    '''

    def __init__(self, d_model, 
                    dropout=0.1,
                    # attention
                    num_heads=8, 
                    w_q=None,
                    w_k=None,
                    w_v=None,
                    w_o=None,
                    # feedforward
                    hidden_dim=2048,
                    bias=True,
                    layers_w=None,
                    layers_b=None,
                    # layer norm
                    eps=1e-5,
                    scale_shift=True,
                    gamma=None,
                    beta=None
        ):
        '''
        Initializes the transformer block.

        Parameters
        ----------
        d_model: int
            The size of word embeddings of the model
        dropout: float, optional
            The dropout rate to apply to the outputs of the sublayers.
        num_heads: int, optional
            The number of attention heads to use in the attention block.
        w_q: array, optional
            The query matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_k: array, optional
            The key matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_v: array, optional
            The value matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_o: array, optional
            The output matrix, of shape (d_model, d_model)
        hidden_dim: int, optional
            The number of hidden units in the feedforward block
        bias: bool, optional
            Whether to include bias in the feedforward block
        layers_w: list, optional
            List of weights for the linear layers of the feedforward block,
            of shape [(d_model, hidden_dim), (hidden_dim, d_model)]
        layers_b: list, optional
            List of biases for the linear layers of the feedforward block,
            of shape [(hidden_dim,), (d_model,)]
        eps: float, optional
            A small value to avoid division by zero in the layer norm
        scale_shift: bool, optional
            Whether to scale and shift the normalized input in the layer norm
        gamma: array, optional
            The gamma parameter for the layer norm, of shape (d_model,)
        beta: array, optional
            The beta parameter for the layer norm, of shape (d_model,)
        '''

        # initialize the multi-headed attention block
        self.attention = MultiHeadAttention(d_model, num_heads=num_heads, 
                                            w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o)
        
        # initialize the feedforward block
        self.feedforward = TwoLayerMLP(input_dim=d_model, hidden_dim=hidden_dim, bias=bias,
                                        layers_w=layers_w, layers_b=layers_b)
        
        # initialize the layer norm
        self.layer_norm = LayerNorm(d_model, eps=eps, 
                                    scale_shift=scale_shift, 
                                    gamma=gamma, beta=beta)
        
        # store the dropout rate
        self.dropout = dropout


    def forward(self, x, attention_mask=None, padding_mask=None):
        '''
        Forward pass through the transformer block.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)
        attention_mask: array, optional
            The attention mask to apply to the attention block - of shape (batch_size, seq_len, d_model)
        padding_mask: array, optional
            A mask showing all the padding tokens 

        Returns
        -------
        array
            Output array - of shape (batch_size, seq_len, d_model)
        '''

        # pass through the attention block
        att_output = self.attention.forward(x, attention_mask=attention_mask)

        # apply dropout
        att_output = dropout(att_output, self.dropout)

        # add the residual connection
        x += att_output

        # normalize the output
        x = self.layer_norm.forward(x)

        # pass through the feedforward block
        ff_output = self.feedforward.forward(x)

        # apply dropout
        ff_output = dropout(ff_output, self.dropout)
        
        # add the residual connection
        x += ff_output

        # normalize the output
        x = self.layer_norm.forward(x)

        # use the padding mask to zero out the padding tokens
        if padding_mask is not None:
            x[padding_mask] = 0.0
        
        return x