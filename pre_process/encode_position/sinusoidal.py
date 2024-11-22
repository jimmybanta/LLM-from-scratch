
import numpy as np


class SinusoidalPE:
    '''
    Sinusoidal Positional Encoding
    '''


    def __init__(self, 
                 pe_file=None,
                 context_size=256,
                 d_embedding=512):
        '''
        Initialize the Sunusoidal Positional Encoder

        Parameters
        ----------
        context_size : int
            The maximum number of tokens in a sequence
        d_embedding : int
            The dimension of the word embedding vectors
        '''

        self.context_size = context_size
        self.d_embedding = d_embedding

        # either load the positional encodings from a file or calculate them
        if pe_file is not None:
            self.positional_encodings = self.load_pe(pe_file)
        else:
            self.positional_encodings = self.calculate_pe()

    def calculate_pe(self) -> np.ndarray:
        '''
        Calculate the positional encodings.
        Using sinusoidal, as described in the paper "Attention is All You Need".

        Returns
        -------
        np.ndarray
            An array of shape (context_size, d_embedding) containing the positional encodings.
        '''

        # get the sequence length and embedding dimensions
        seq_len = self.context_size
        emb_dim = self.d_embedding

        # create an array to store the positional encodings
        pe = np.zeros((seq_len, emb_dim))

        # position values
        pos = np.arange(seq_len).reshape(seq_len, 1)

        # dimension values
        emb_dim_indices = np.arange(0, emb_dim / 2)
        dim = 10000 ** ((emb_dim_indices * 2) / emb_dim)

        # calculate the encodings for even and odd embedding dimensions
        evens = np.sin(pos / dim)
        odds = np.cos(pos / dim)

        # apply the encodings to the positional encoding array
        pe[:, 0::2] = evens
        pe[:, 1::2] = odds

        return pe
    
    def save_pe(self, pe: np.ndarray, path: str):
        '''
        Save the positional encodings to a file.

        Parameters
        ----------
        pe : np.ndarray
            An array of shape (context_size, d_embedding) containing the positional encodings.
        path : str
            The path to save the positional encodings.
        '''

        if not isinstance(pe, np.ndarray):
            raise TypeError('pe must be a numpy array.')

        if not isinstance(path, str):
            raise TypeError('path must be a string.')

        np.save(path, pe)

    def load_pe(self, path: str) -> np.ndarray:
        '''
        Load the positional encodings from a file.

        Parameters
        ----------
        path : str
            The path to load the positional encodings.

        Returns
        -------
        np.ndarray
            An array of shape (context_size, d_embedding) containing the positional encodings.
        '''

        if not isinstance(path, str):
            raise TypeError('path must be a string.')

        return np.load(path)

    def encode(self, batch_embeddings: np.ndarray) -> np.ndarray:
        '''
        Encode the positional information of the word embeddings.

        Parameters
        ----------
        batch_embeddings : np.ndarray
            An array of shape (batch_size, context_size, d_embedding) containing the embeddings of the tokens in the batch.

        Returns
        -------
        np.ndarray
            An array of shape (batch_size, context_size, d_embedding) containing the 
            embeddings plus positional encodings of the tokens in the batch.
        '''

        if not isinstance(batch_embeddings, np.ndarray):
            raise TypeError('batch_embeddings must be a numpy array.')

        # Find the first non-zero index for each sequence in the batch
        # Returns array of shape (batch_size,)
        first_indices = np.argmax(np.any(batch_embeddings != 0, axis=2), axis=1)

        # Create a mask for non-padding positions
        # Shape: (batch_size, context_size)
        mask = np.arange(batch_embeddings.shape[1])[None, :] >= first_indices[:, None]

        # Expand dimensions to match the embedding dimension
        # Shape: (batch_size, context_size, 1)
        mask = mask[..., None]

        # Create position indices relative to the first token
        # Shape: (batch_size, context_size)
        positions = np.arange(batch_embeddings.shape[1])[None, :] - first_indices[:, None]
        positions = np.clip(positions, 0, self.positional_encodings.shape[0] - 1)

        # Get the positional encodings for each position
        # Shape: (batch_size, context_size, d_embedding)
        encodings = self.positional_encodings[positions]

        # Apply mask to zero out encodings for padding tokens
        encodings = encodings * mask

        # Add positional encodings to the embeddings
        return batch_embeddings + encodings            


        
