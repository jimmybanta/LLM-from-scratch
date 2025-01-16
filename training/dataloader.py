
import numpy as np


class DataLoader:
    '''
    A DataLoader that, given pre-processed text data, forms batches of sentence-label pairs.
    '''

    def __init__(self, embeddings, integers, 
                 vocab_size=10000,
                 batch_size=32, shuffle=True):
        '''
        Initialize the DataLoader.

        Parameters
        ----------
        embeddings: np.ndarray
            The word embeddings.
        integers: np.ndarray
            The integer representations of the tokens.
        vocab_size: int, optional
            The size of the vocabulary.
        batch_size: int, optional
            The batch size.
        shuffle: bool, optional
            Whether to shuffle the data.
        '''

        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.embeddings = np.array(embeddings)

        # remove the first integer from each sentence
        ## this way, at a given position, you have the embedding, 
        ## and the token value that comes next
        ## i.e. the token it's trying to predict
        integers = np.array([sentence[1:] for sentence in integers])
                
        # now, form one-hot vectors from the integers
        ## this is the target
        ## the model will predict the next token
        ## given the current token
        one_hot = np.zeros((integers.shape[0], integers.shape[1], vocab_size))
        one_hot[np.arange(integers.shape[0])[:, None], np.arange(integers.shape[1]), integers] = 1
        
        self.targets = one_hot

        self.shuffle = shuffle

        self.batches = []

        self.form_batches()
        

    
    def form_batches(self):
        '''
        Form batches of sentence-label pairs.
        '''
        
        # shuffle the data
        if self.shuffle:
            indices = np.random.permutation(len(self.embeddings))
            self.embeddings = self.embeddings[indices]
            self.targets = self.targets[indices]
        
    
        # form the batches
        for i in range(0, len(self.embeddings), self.batch_size):
            self.batches.append((self.embeddings[i: i + self.batch_size], self.targets[i: i + self.batch_size]))
        
    def __iter__(self):
        '''
        Iterate through the DataLoader.
        '''
        
        for batch in self.batches:
            yield batch


    