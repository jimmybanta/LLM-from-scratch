
import numpy as np
from transformer.utils import softmax

class AttentionHead:
    '''
    A single attention head.
    '''

    def __init__(self, d_model, d_k, d_v,
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
        

    def forward(self, x, attention_mask=None):
        '''
        Forward pass through the attention head.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)
        attention_mask: array, optional
            Attention mask to apply to the attention scores

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

        # apply the attention mask, if provided
        if attention_mask is not None:
            scores[attention_mask] = -np.inf


        """ 
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
        """

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

    def __init__(self, d_model, 
                 num_heads=8,
                 w_q=None, 
                 w_k=None, 
                 w_v=None,
                 w_o=None):
        '''
        Initializes the layer.

        Parameters
        ----------
        d_model: int
            The size of word embeddings of the model
        num_heads: int, optional
            The number of attention heads to use.
        w_q: array, optional
            The query matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_k: array, optional
            The key matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_v: array, optional
            The value matrix, of shape (num_heads, d_model, d_model // num_heads)
        w_o: array, optional
            The output matrix, of shape (d_model, d_model)
        '''

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # instantiate the attention heads
        if w_q and w_k and w_v:
            self.heads = [AttentionHead(d_model, self.d_k, self.d_v, w_q=w_q[i], w_k=w_k[i], w_v=w_v[i]) for i in range(num_heads)]
        else:
            self.heads = [AttentionHead(d_model, self.d_k, self.d_v) for _ in range(num_heads)]

        self.w_o = np.random.randn(d_model, d_model) if not w_o else w_o


    def forward(self, x, attention_mask=None):
        '''
        Forward pass through the attention layer.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)
        attention_mask: array, optional
            Attention mask to apply to the attention scores

        Returns 
        -------
        values: array
            Array of calculated values - of shape (batch_size, seq_len, d_model)
        '''

        # get the outputs from each head
        outputs = [head.forward(x, attention_mask=attention_mask) for head in self.heads]

        # concatenate the outputs to form one array
        stacked_output = np.concatenate(outputs, axis=2)

        # multiply the outputs by the output matrix to get the output of the layer
        return stacked_output @ self.w_o
