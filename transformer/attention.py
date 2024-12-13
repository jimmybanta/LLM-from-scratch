
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
        # take softmax of these attention scores
        scores = softmax(scores, axis=2)

        # multiply by value vectors
        values = scores @ value

        return values
    

if __name__ == '__main__':

    batch_size = 32
    seq_len = 2048
    d_model = 512


    i = np.random.randn(batch_size, seq_len, d_model)

    head = AttentionHead(d_model, 64, 64)

    temp = head.forward(i)




