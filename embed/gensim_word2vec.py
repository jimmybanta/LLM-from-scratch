''' Uses gensim to train a word2vec model on a text corpus. '''

import gensim


class GensimWord2Vec:
    '''
    Class to train a word2vec model using gensim.
    '''

    def __init__(self, tokenized_sentences, size=100, window=5, min_count=1, workers=3):
        '''
        Initialize the class with the given parameters.

        Parameters
        ----------
        tokenized_sentences : list
            A list of sentences, where each sentence is a list of tokens.
        size : int
            The size of the word vectors.
        window : int
            The size of the context window.
        min_count : int
            The minimum number of times a word must appear in the corpus to be included in the vocabulary.
        workers : int
            The number of worker threads to use for training
        '''
        self.sentences = tokenized_sentences
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
    
    def train(self):
        '''
        Train the word2vec model on the given corpus.
        '''
        self.model = gensim.models.Word2Vec(sentences=self.sentences, 
                                            vector_size=self.size,
                                            window=self.window,
                                            min_count=self.min_count, 
                                            workers=self.workers)
        
    def save(self, model_path):
        '''
        Save the trained model to the given path.

        Parameters
        ----------
        model_path : str
            The path to save the model to.
        '''
        self.model.save(model_path)