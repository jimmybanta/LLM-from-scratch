''' File used for training various tokenizers. '''

import os
from dotenv import load_dotenv

from data.import_tiny import import_tiny
from data.import_small import import_full_list
from pre_process.tokenize.bpe import NaiveBPETokenizer

load_dotenv()

def train_bpe_and_save(corpus, 
                       vocab_save_path,
                       lookup_table_save_path,
                       vocab_size=10000):
    '''
    Given a corpus, trains and saves a tokenizer.
    '''

    # initialize the tokenizer
    tokenizer = NaiveBPETokenizer(corpus=corpus, vocab_size=vocab_size)

    # train the tokenizer
    tokenizer.train()

    # save the vocab and lookup table
    tokenizer.write_to_file(tokenizer.vocab, vocab_save_path)
    tokenizer.write_to_file(tokenizer.lookup_table, lookup_table_save_path)


def train_bpe_tiny():
    '''
    Train a BPE tokenizer on a tiny corpus.
    '''

    corpus = import_tiny()
    
    vocab_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'tiny_vocab.json')
    lookup_table_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'tiny_lookup_table.json')

    train_bpe_and_save(corpus, vocab_save_path, lookup_table_save_path)

def train_bpe_small():
    '''
    Train a BPE on a larger corpus, with a relatively small vocab (1024)
    '''

    corpus = import_full_list()

    vocab_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'small_vocab.json')
    lookup_table_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'small_lookup_table.json')

    train_bpe_and_save(corpus, vocab_save_path, lookup_table_save_path, vocab_size=1024)

def train_bpe_medium():
    '''
    Train a BPE on a larger corpus, with a larger vocab (as big as it can get)
    '''

    corpus = import_full_list()

    vocab_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_vocab.json')
    lookup_table_save_path = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_lookup_table.json')

    train_bpe_and_save(corpus, vocab_save_path, lookup_table_save_path)


if __name__ == '__main__':

    # train the BPE tokenizers
    train_bpe_tiny()
    train_bpe_small()
    train_bpe_medium()