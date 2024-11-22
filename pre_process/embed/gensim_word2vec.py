''' Uses gensim to train a word2vec model on a text corpus. '''

import os
from typing import List
import logging
import logging.config

import gensim
import numpy as np

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


class GensimWord2Vec:
    '''
    Class to train a word2vec model using gensim.
    '''

    def __init__(self, 
                 tokenized_sentences=None, 
                 model_file=None,
                 size=512, 
                 window=5, 
                 min_count=1, 
                 workers=3,
                 context_size=250,
                 padding_token='<pad>'):
        '''
        Initialize the class with the given parameters.

        Parameters
        ----------
        tokenized_sentences : list
            A list of sentences, where each sentence is a list of tokens.
        model_file : str
            The path to a file to load a pre-trained model/embeddings from.
        size : int
            The size of the word vectors.
        window : int
            The size of the context window.
        min_count : int
            The minimum number of times a word must appear in the corpus to be included in the vocabulary.
        workers : int
            The number of worker threads to use for training
        context_size : int
            The size of the context window to use when training the LLM.
        '''
        self.sentences = tokenized_sentences
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.context_size = context_size

        if model_file:
            if os.path.splitext(model_file)[1] == '.bin':
                self.load(model_file, file_type='binary')
            elif os.path.splitext(model_file)[1] == '.txt':
                self.load(model_file, file_type='table')
            
            # set the vector size

            if self.model.vector_size != self.size:
                raise ValueError(f'The size of the model ({self.model.vector_size}) does not match the size parameter ({self.size}).')

            self.size = self.model.vector_size

        
        self.padding_token = padding_token
        
        self.unknown_embedding = np.random.rand(self.size)
        self.padding_embedding = np.zeros(self.size)
    
    def train(self):
        '''
        Train the word2vec model on the given corpus.
        '''

        if not self.sentences:
            raise ValueError('No sentences provided for training.')

        trained_w2v = gensim.models.Word2Vec(sentences=self.sentences, 
                                            vector_size=self.size,
                                            window=self.window,
                                            min_count=self.min_count, 
                                            workers=self.workers)
        
        # set the model to the trained KeyedVectors object
        self.model = trained_w2v.wv
   
    def save(self, path, file_type='binary'):
        '''
        Save the model/embeddings to a file.
        Either saves the full model, the word vectors as a binary file, or as a text file.

        Parameters
        ----------
        path : str
            The path to save the file to.
        file_type : str | 'binary'
            The type of file to save. Can be 'binary' or 'table'.
        '''
    
        if file_type == 'binary':
            self.model.save_word2vec_format(path, binary=True)
        elif file_type == 'table':
            self.model.save_word2vec_format(path, binary=False)

    def load(self, path, file_type='binary'):
        '''
        Load a model/embeddings from a file.

        Parameters
        ----------
        path : str
            The path to the file to load.
        file_type : str | 'binary'
            The type of file to load. Can be 'binary' or 'table'.
        '''
        if file_type == 'binary':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        elif file_type == 'table':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    
    def embed_batch(self, text: List[List]) -> np.ndarray:
        '''
        Embed a batch (list) of lists of tokens using the trained word2vec model.

        Parameters
        ----------
        text : List[List[str/int]]
            A list of tokens to embed.

        Returns
        -------
        ndarray
            An array of arrays of word vectors.
        '''

        batch_embeddings = []

        # iterate through each sentence
        for sentence in text:

            sentence_embeddings = []

            for token in sentence:
                if token not in self.model:
                    if token == self.padding_token:
                        sentence_embeddings.append(self.padding_embedding)
                    else:
                        logger.info(f'Unknown token: {token}')
                        sentence_embeddings.append(self.unknown_token)
                
                else:
                    sentence_embeddings.append(self.model[token])
            
            # if the sentence is shorter than the context size, pad it
            while len(sentence_embeddings) < self.context_size:
                sentence_embeddings.insert(0, self.padding_embedding)
            
            batch_embeddings.append(sentence_embeddings)

        
        return np.array(batch_embeddings)
                

            

    
