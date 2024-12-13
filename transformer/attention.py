
import numpy as np

class AttentionHead:
    '''
    A single attention head.
    '''

    def __init__(self, d_model, d_k, d_v, 
                    w_q=None, w_k=None, w_v=None):

        # input shape - (batch_size, seq_len, d_model)

        self.d_k = d_k


        # query weights, of shape (d_model, d_k)
        self.w_q = np.random.randn(d_model, d_k) if not w_q else w_q 
        # key weights, of shape (d_model, d_k)
        self.w_k = np.random.randn(d_model, d_k) if not w_k else w_k
        # value weights, of shape (d_model, d_k)
        self.w_v = np.random.randn(d_model, d_v) if not w_v else w_v

        # vector of length [512] -> [64]
        # [1, 512] * [512, 64] = [1, 64]
        

    def forward(self, x):

        # first, we need to calculate query, key, and value vectors
        query = x @ self.w_q
        key = x @ self.w_k
        value = x @ self.w_v

        # then, we need to calculate the attention scores
        ## first, multiply query by transposed key matrices
        scores = query @ key.transpose(0, 2, 1)
        # then, divide element-wise by square root of d_k
        scores /= np.sqrt(self.d_k)

        # then, we need to take softmax
        ## will implement softmax later 
        #scores = softmax(scores)

        # then, we need to multiply by value vectors
        scores = scores @ value

        
        return scores
    

if __name__ == '__main__':

    batch_size = 32
    seq_len = 2048
    d_model = 512


    i = np.random.randn(batch_size, seq_len, d_model)

    head = AttentionHead(d_model, 64, 64)

    temp = head.forward(i)

    print(temp.shape)




