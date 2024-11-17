from unidecode import unidecode
from typing import List
import string
import json
import os
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv

from data.import_small import import_full_list

load_dotenv()


class BPETokenizer:
    '''
    A tokenizer that uses Byte Pair Encoding (BPE) to tokenize text.
    '''
    
    def __init__(self, model_dir, 
                 vocab_file=None,
                 corpus=None, vocab_size=1024):
        '''
        Initialize the BPETokenizer.
        
        Parameters
        ----------
        corpus : str | None
            The corpus used to train the tokenizer. Default is None.
        '''

        # the directory to save the model
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # the corpus used to train the tokenizer
        self.corpus = corpus

        # if a vocab file is given, then read in the vocab
        if vocab_file:
            self.vocab = self.read_vocab_from_file(vocab_file)
        else:
            self.vocab = []

        self.vocab_size = vocab_size


    def normalize(self, input: list, 
                  lowercase=False) -> List[str]:
        '''
        Normalize the input text.

        Parameters
        ----------
        input : list
            The input text, as a list of strings.
        lowercase : bool | False
            Whether to lowercase the text. Default is False.

        Returns
        -------
        list[str]
            The normalized text, as a list of strings.
        '''

        if type(input) != list:
            raise TypeError('Input must be a list of strings.')
        
        normalized_text = []
        
        for text in input:

            # convert to ASCII
            text = unidecode(text)

            # lowercase the text, if required
            if lowercase:
                text = text.lower()
            
            normalized_text.append(text)
        
        return normalized_text
    
    def get_vocab_word_counts(self, corpus_split) -> tuple:
        '''
        Given a split corpus that has been pre-tokenized, 
        gets the initial vocabulary
        and the word counts dictionary.
        
        Parameters
        ----------
        corpus_split : list
            The pre-tokenized corpus.
            
        Returns
        -------
        list, dict
            The initial vocabulary and the word counts dictionary.
        '''

        word_counts = {}
        vocab = []

        # create the initial vocabulary from the corpus
        # also create the word_counts -- 
        ## contains the words in the corpus, their individual tokens, and the frequency of the word
        for word in corpus_split:

            # skip our special characters
            if word in self.special_characters:
                continue

            # if the word is not in the word_counts, add it
            if word not in word_counts:
                word_counts[word] = {
                    'count': 1,
                    'tokens': [char for char in word]
                }
            # otherwise, increment the count
            else:
                word_counts[word]['count'] += 1

            # add characters to the vocab
            for char in word:
                if char not in vocab:
                    vocab.append(char)

        return vocab, word_counts
    
    def update_vocab(self, word_counts) -> list: 
        '''
        Given the word counts dictionary, 
        updates the vocab with the most frequent character combos using BPE.

        Returns
        -------
        list
            The updated vocabulary.
        '''

        # copy the vocab
        vocab = self.vocab.copy()

        # loop until the vocab size is reached
        while len(vocab) < self.vocab_size:

            print(f'Working on vocab # {len(vocab)}')

            # combos will store the frequency of the character combos
            combos = {}

            # create the combos
            for word, word_info in word_counts.items():
                
                for i in range(len(word_info['tokens'])):

                    if i == 0:
                        continue

                    combo = f'{word_info["tokens"][i - 1]}{word_info["tokens"][i]}'

                    if combo not in combos:
                        combos[combo] = word_info['count']
                    else:
                        combos[combo] += word_info['count']

            # sort the combos by frequency
            combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)

            # if there are no combos, break
            if len(combos) == 0:
                print('No combos found.')
                break

            # get the most frequent combo
            most_frequent = combos[0][0]

            # update the vocab with the most frequent combo
            vocab.append(most_frequent)

            # update the word_counts to include this combo
            for word, word_info in word_counts.items():

                if most_frequent in word:
                    
                    # update the tokens
                    new_tokens = []

                    for i, token in enumerate(word_info['tokens']):

                        # look at this token and the next token
                        try:
                            if f'{token}{word_info["tokens"][i + 1]}' == most_frequent:
                                new_tokens.append(most_frequent)
                                continue
                        except IndexError:
                            pass
                        
                        # look at this token and the previous token
                        ## we don't want to add the most frequent token twice
                        if f'{word_info["tokens"][i - 1]}{token}' == most_frequent:
                            continue

                        # if the token isn't part of the combo, then just add it
                        new_tokens.append(token)

                    word_info['tokens'] = new_tokens
        
        return vocab

    def train(self):
        '''
        Train the tokenizer using the corpus.

        Updates self.vocab with the vocabulary it is learning through training.
        '''

        # add the special characters to the vocab
        self.vocab = self.special_characters

        # normalize the corpus
        corpus = self.normalize(self.corpus)
        print('done normalizing')

        # pre-tokenize the corpus
        corpus_split = self.pre_tokenize(corpus)
        print('done pre-tokenizing')

        initial_vocab, word_counts = self.get_vocab_word_counts(corpus_split)

        print(word_counts)

        # add the initial vocab to the vocab
        self.vocab = self.vocab + initial_vocab
        
        print(self.vocab)

        # update the vocab using BPE algorithm
        self.vocab = self.update_vocab(word_counts)

        # sort the vocab
        self.vocab = sorted(self.vocab)
        
    def write_vocab_to_file(self, filename):
        '''
        Write the vocabulary to a json file.

        Parameters
        ----------
        filename : str
            The filename to write the vocabulary to.
        '''

        with open(filename, 'w') as f:
            json.dump(self.vocab, f, indent=4)
    
    def read_vocab_from_file(self, filename):
        '''
        Read the vocabulary from a json file.

        Parameters
        ----------
        filename : str
            The filename to read the vocabulary from.
        '''

        with open(filename, 'r') as f:
            vocab = json.load(f)

        return vocab

    def lookup_brute_search(self, token: str) -> int:
        '''
        Given a token, does a brute force search to find the index of the token.

        Parameters
        ----------
        token : str
            The token to find the index of.

        Returns
        -------
        int
            The index of the token.
        '''

        for i, vocab_token in enumerate(self.vocab):
            if vocab_token == token:
                return i
        
        return -1
        
    def encode(self, text: str) -> List[int]:
        '''
        Given text, encodes it using the vocabulary.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        List[int]
            A list of the encoded tokens.
        '''

        # TO DO: implement this method

        pass




        

class NaiveBPETokenizer(BPETokenizer):
    '''
    A BPE tokenizer that uses a naive approach - having whitespace and punctuation as separate tokens, 
    that don't get included into the merges.
    '''

    def __init__(self, model_dir, 
                 vocab_file=None,
                 corpus=None):
        '''
        Initialize the NaiveBPETokenizer.
        
        Parameters
        ----------
        corpus : list | None
            The corpus used to train the tokenizer. Default is None.
        '''

        super().__init__(model_dir, vocab_file=vocab_file, corpus=corpus)

        self.special_characters = ['<space>', '<newline>', '<tab>', '<endoftext>', '<unknown>']
        self.special_characters = self.special_characters + [mark for mark in string.punctuation]

    def pre_tokenize(self, input: list) -> List[str]:
        '''
        Pre-tokenize the input text.
        Splits along whitespace (space, newline, tab) and punctuation/special characters.

        Parameters
        ----------
        input : list
            The input text, as a list of strings.

        Returns
        -------
        List[str]
            The pre-tokenized text.
        '''

        if type(input) != list:
            raise TypeError('Input must be a list of strings.')

        space_dict = {
            ' ': '<space>',
            '\n': '<newline>',
            '\t': '<tab>'
        }

        # text-split will store the pre-tokenized text
        text_split = []

        # iterate through all the text in the input
        for text in input:

            # remove leading and trailing white space
            text = text.strip()

            # traverse through the characters of the text string
            current_word = ''
            for char in text:

                # space, newline, tab
                if char in [' ', '\n', '\t']:
                        
                    # if current_word is not empty, add it to words_split
                    if current_word != '':
                        text_split.append(current_word)

                    # add the special character to words_split
                    text_split.append(space_dict[char])

                    # reset current_word
                    current_word = ''
                
                # check for punctuation
                elif bool(re.match(r'[^\w\s]', char)):

                    if current_word != '':
                        text_split.append(current_word)

                    text_split.append(char)
                    current_word = ''
                
                # otherwise, add the character to current_word
                else:
                    current_word += char
            
            # add the last word to words_split
            if current_word != '':
                text_split.append(current_word)
            
            # add end of text token
            text_split.append('<endoftext>')

        return text_split
    
