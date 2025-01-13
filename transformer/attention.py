
import numpy as np
from transformer.utils import softmax

class AttentionHead:
    '''
    A single attention head.
    '''

    def __init__(self, d_model, d_k, d_v, seq_len,
                    w_q=None, w_k=None, w_v=None):
        '''
        Initializes the attention head.

        Parameters
        ----------
        d_model: int
            The size of word embeddings of the model
        d_k: int
            The size of query and key vectors
        d_v: int
            The size of value vectors
        seq_len: int
            The length of input sequences
        w_q: array, optional
            The query weights, of shape (d_model, d_k)
        w_k: array, optional
            The key weights, of shape (d_model, d_k)
        w_v: array, optional
            The value weights, of shape (d_model, d_v)
        '''


        self.d_k = d_k
        self.d_v = d_v

        # query weights, of shape (d_model, d_k)
        self.w_q = np.random.randn(d_model, d_k) if not w_q else w_q 
        # key weights, of shape (d_model, d_k)
        self.w_k = np.random.randn(d_model, d_k) if not w_k else w_k
        # value weights, of shape (d_model, d_k)
        self.w_v = np.random.randn(d_model, d_v) if not w_v else w_v

        # generate the mask - that masks out later tokens
        ## returns True whenever a token is after the current token
        mesh = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
        self.position_mask = (mesh[1] < mesh[0])
        
        

    def forward(self, x):
        '''
        Forward pass through the attention head.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)

        Returns 
        -------
        values: array
            Array of calculated values - of shape (batch_size, seq_len, d_v)
        '''

        # calculate query, key, and value vectors
        query = x @ self.w_q
        key = x @ self.w_k
        value = x @ self.w_v

        # calculate the attention scores
        ## multiply query by transposed key matrices
        scores = query @ key.transpose(0, 2, 1)
        # divide element-wise by square root of d_k
        scores /= np.sqrt(self.d_k)

        # masks
        ## expand the position mask to the batch size
        batch_size = x.shape[0]
        position_mask = np.repeat(np.expand_dims(self.position_mask, 0), batch_size, axis=0)

        ## generate the padding mask
        ### find where the embeddings are equal to zero
        ### these are the padding tokens
        padding_mask = np.all(x == 0, axis=2)
        ### expand the padding mask
        padding_mask = np.repeat(padding_mask[:, None, :], padding_mask.shape[1], axis=1)

        # mask out the values
        scores[padding_mask] = -np.inf
        scores[position_mask] = -np.inf

        # take softmax of these attention scores
        scores = softmax(scores, axis=2)

        # replace nan with 0 - for the padding tokens
        scores = np.nan_to_num(scores, nan=0.0)

        # multiply by value vectors
        values = scores @ value

        return values
    

class MultiHeadAttention:
    '''
    An attention layer with multiple attention heads.
    '''

    def __init__(self, d_model, seq_len, 
                 num_heads=8,
                 w_o=None):
        '''
        Initializes the layer.

        Parameters
        ----------
        d_model: int
            The size of word embeddings of the model
        seq_len: int
            The length of input sequences
        num_heads: int, optional
            The number of attention heads to use.
        w_o: array, optional
            The output matrix, of shape (d_model, d_model)
        '''

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # instantiate the attention heads
        self.heads = [AttentionHead(d_model, self.d_k, self.d_v, seq_len) for _ in range(num_heads)]

        self.w_o = np.random.randn(d_model, d_model) if not w_o else w_o


    def forward(self, x):
        '''
        Forward pass through the attention layer.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)

        Returns 
        -------
        values: array
            Array of calculated values - of shape (batch_size, seq_len, d_model)
        '''

        # get the outputs from each head
        outputs = [head.forward(x) for head in self.heads]

        # concatenate the outputs to form one array
        stacked_output = np.concatenate(outputs, axis=2)

        # multiply the outputs by the output matrix to get the output of the layer
        return stacked_output @ self.w_o
