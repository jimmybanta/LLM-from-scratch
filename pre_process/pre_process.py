
import os
import logging.config
import logging
from time import perf_counter as pc


from pre_process.tokenize.bpe import NaiveBPETokenizer
from pre_process.embed.gensim_word2vec import GensimWord2Vec
from pre_process.encode_position.sinusoidal import SinusoidalPE

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

class PreProcessor:
    '''
    Combines all data preprocessing steps
    '''


    def __init__(self,
                    corpus=None,
                    train=False,
                    save_path=None,
                    files={
                        'vocab': None,
                        'lookup_table': None,
                        'word_embeddings': None,
                        'positional_encodings': None
                    },
                    vocab_size=4096,
                    context_size=2048,
                    embedding_dim=512,
                    embedding_params={
                        'window': 5,
                        'min_count': 1,
                        'workers': 3
                    },
                    padding_token='<pad>'
                 ):
        '''
        Initialize the preprocessor.

        Parameters
        ----------
        corpus : List[str] | None
            The corpus to train on. 
            A list of sentences, where each sentence is a string.
        train : bool | False
            Whether to train a tokenizer and word embedder.
        save_path : str | None
            If training, the path to save the training files.
        files : Dict[str, str] | None
            The paths to the files to load.
        vocab_size : int | 4096
            The size of the vocabulary.
        context_size : int | 2048
            The size of the context window.
        embedding_dim : int | 512
            The size of the word vectors.
        embedding_params : Dict[str, int] | {'window': 5, 'min_count': 1, 'workers': 3}
            The parameters to use when training the word embeddings.
        padding_token : str | '<pad>'
            The token to use for padding.
        '''

        # if we're training, check that we have what we need
        if train:

            if not corpus:
                raise ValueError('If training, a corpus must be provided.')
            if not save_path:
                raise ValueError('If training, a save path must be provided.')
        
        else:

            if not files['vocab']:
                raise ValueError('A vocabulary file must be provided.')
            if not files['lookup_table']:
                raise ValueError('A lookup table file must be provided.')
            if not files['word_embeddings']:
                raise ValueError('A word embeddings file must be provided.')
        
        # initialize the tokenizer
        self.tokenizer = NaiveBPETokenizer(
            vocab_file=files['vocab'],
            lookup_table_file=files['lookup_table'],
            corpus=corpus,
            vocab_size=vocab_size,
            context_size=context_size,
            padding_token=padding_token
        )

        # initialize the word embedder
        self.word_embedder = GensimWord2Vec(
                embeddings_file=files['word_embeddings'],
                size=embedding_dim,
                context_size=context_size,
                padding_token=padding_token,
                **embedding_params
        )
        
        # initialize the positional encoder
        self.pe = SinusoidalPE(
            pe_file=files['positional_encodings'],
            context_size=context_size, 
            d_embedding=embedding_dim)
        
        # train the tokenizer and word embedder if needed
        if train:
            logger.info('Training tokenizer and word embedder.')
            # train the tokenizer
            self.tokenizer.train()
            logger.info('Tokenizer trained.')
            # save the vocab
            self.tokenizer.save_to_file(self.tokenizer.vocab, os.path.join(save_path, 'vocab.json'))
            # save the lookup table
            self.tokenizer.save_to_file(self.tokenizer.lookup_table, os.path.join(save_path, 'lookup_table.json'))
            logger.info('Tokenizer saved.')

            # tokenize the corpus
            tokenized_corpus = self.tokenizer.encode(corpus, return_integers=False)
            logger.info('Corpus tokenized.')

            # train the word embedder
            self.word_embedder.sentences = tokenized_corpus
            self.word_embedder.train()
            logger.info('Word embedder trained.')

            # save the word embeddings, in both binary and table format
            self.word_embedder.save(os.path.join(save_path, 'word_embeddings.bin'), file_type='binary')
            self.word_embedder.save(os.path.join(save_path, 'word_embeddings.txt'), file_type='table')
            logger.info('Word embeddings saved.')

            
    def pre_process(self, batch):
        '''
        Pre-process a batch of text.

        Parameters
        ----------
        batch : List[str]
            A list of sentences to pre-process.
        '''

        # tokenize the batch
        tokenized_batch = self.tokenizer.encode(batch, return_integers=False)

        # embed the batch
        embedded_batch = self.word_embedder.embed_batch(tokenized_batch)

        # encode the positional information
        encoded_batch = self.pe.encode(embedded_batch)

        return encoded_batch
        


        