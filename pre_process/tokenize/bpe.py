import json
import logging
import logging.config
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from unidecode import unidecode

load_dotenv()

# configure the logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


class BPETokenizer:
    '''
    A tokenizer that uses Byte Pair Encoding (BPE) to tokenize text.
    '''
    
    def __init__(self, 
                    model_dir=None, 
                    vocab_file=None,
                    lookup_table_file=None,
                    corpus=None, 
                    vocab_size=1024):
        '''
        Initialize the BPETokenizer.
        
        Parameters
        ----------
        model_dir : str | None
            The directory to save the model. Default is None.
        vocab_file : str | None
            The file to read the vocabulary from. Default is None.
        lookup_table_file : str | None
            The file to read the lookup table from. Default is None.
        corpus : str | None
            The corpus used to train the tokenizer. Default is None.
        vocab_size : int | 1024
            The size of the vocabulary. Default is 1024.
        '''

        # the directory to save the model (optional)
        self.model_dir = model_dir
        if model_dir:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # the corpus used to train the tokenizer
        self.corpus = corpus

        # if a vocab file is given, then read in the vocab
        if vocab_file:
            self.vocab = self.read_from_file(vocab_file)
        else:
            self.vocab = []

        # if a lookup table file is given, then read in the lookup table
        if lookup_table_file:
            self.lookup_table = self.read_from_file(lookup_table_file)
        else:
            self.lookup_table = {}

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

        last_combo = ''

        # loop until the vocab size is reached
        ## taking into account the special characters
        while len(vocab) < self.vocab_size - len(self.special_characters):

            # combos will store the frequency of the character combos
            combos = {}

            # create the combos

            # go through each word in the corpus
            for word, word_info in word_counts.items():
                
                # go through each token in the word
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
                logger.info('No combos found.')
                break
                
            # get the most frequent combo
            most_frequent = combos[0][0]

            # if the most frequent combo is the same as the last combo, break
            if most_frequent == last_combo:
                break

            last_combo = most_frequent

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

            logger.info(f'Added {most_frequent} to vocab. Vocabulary size: {len(vocab)}')

            
                
        return vocab

    def train(self):
        '''
        Train the tokenizer using the corpus.

        Updates self.vocab with the vocabulary it is learning through training.
        '''

        # normalize the corpus
        corpus = self.normalize(self.corpus)
        print('done normalizing')

        # pre-tokenize the corpus
        corpus_split = self.pre_tokenize(corpus)
        print('done pre-tokenizing')

        initial_vocab, word_counts = self.get_vocab_word_counts(corpus_split)

        # add the initial vocab to the vocab
        self.vocab = self.vocab + initial_vocab
        
        # update the vocab using BPE
        self.vocab = self.update_vocab(word_counts)

        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add the special characters to the front
        self.vocab = self.special_characters + self.vocab

        # generate the lookup table
        self.generate_lookup_table()
        
    def write_to_file(self, filename):
        '''
        Writes an object to a json file.

        Parameters
        ----------
        filename : str
            The filename to write to.
        '''

        with open(filename, 'w') as f:
            json.dump(self.vocab, f, indent=4)
    
    def read_from_file(self, filename):
        '''
        Reads an object from a json file.

        Parameters
        ----------
        filename : str
            The filename to read from.
        '''

        with open(filename, 'r') as f:
            data = json.load(f)

        return data
    
    def read_special_char_from_file(self, filename):
        '''
        Read the special characters from a json file.

        Parameters
        ----------
        filename : str
            The filename to read the special characters from.
        '''

        with open(filename, 'r') as f:
            special_characters = json.load(f)

        return special_characters

    def generate_lookup_table(self):
        '''
        From the vocab, generates a lookup table to allow for easy lookup of tokens.
        The lookup table is structured as a nested dictionary - so given a token,
        you iteratively go through the characters until you arrive at the index of the token.

        Returns
        -------
        dict
            The lookup table.
        '''

        lookup_table = {'special': {}}

        for i, token in enumerate(self.vocab):

            
            if token in self.special_characters:
                
                lookup_table['special'][token] = i

            else:
                current_dict = lookup_table

                # go through each character in the token
                for char in token:

                    # if the character is not in the current dictionary, add it
                    if char not in current_dict:
                        current_dict[char] = {}
                    
                    # move to the next dictionary
                    current_dict = current_dict[char]


                # add the token to the current dictionary
                current_dict['complete_token'] = i
                


        self.lookup_table = lookup_table

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
        
    def lookup_binary_search(self, token: str) -> int:
        '''
        Given a token, does a binary search to find the index of the token.

        Parameters
        ----------
        token : str
            The token to find the index of.

        Returns
        -------
        int
            The index of the token.
        '''

        # if token is in special characters, then we can brute force search
        # as all the special characters are front-loaded in the vocab
        if token in self.special_characters:
            return self.lookup_brute_search(token)


        low = len(self.special_characters)
        high = len(self.vocab) - 1

        while low <= high:
                
            mid = (low + high) // 2

            if self.vocab[mid] == token:
                return mid
            elif self.vocab[mid] < token:
                low = mid + 1
            else:
                high = mid - 1
        
        return -1

    def lookup_table_search(self, token: str) -> int:
        '''
        Given a token, uses the lookup table to find the index of the token.

        Parameters
        ----------
        token : str
            The token to find the index of.

        Returns
        -------
        int
            The index of the token.
        '''

        if not self.lookup_table:
            raise ValueError('Lookup table not generated.')

        current_dict = self.lookup_table

        if token in current_dict['special']:
            return current_dict['special'][token]

        for char in token:

            if char not in current_dict:
                return -1

            current_dict = current_dict[char]
        
        if 'complete_token' not in current_dict:
            return -1

        return current_dict['complete_token']

    def encode(self, text: list, return_integers=True) -> List[int]:
        '''
        Given a list of strings, encodes them using the vocabulary.
        Either encodes them as their integer values, or as the tokens themselves.

        Parameters
        ----------
        text : str
            The text to encode.
        return_integers : bool | True
            Whether to return the tokens as integers. Default is True.

        Returns
        -------
        List[int]
            A list of the tokens, encoded either as integers or as the tokens themselves.
        '''

        # first, normalize and pre-tokenize the text
        normalized_text = self.normalize(text)
        pre_tokenized_text = self.pre_tokenize(normalized_text)

        values = []

        for word in pre_tokenized_text:
            
            if word in self.special_characters:
                if return_integers:
                    values.append(self.lookup_table_search(word))
                else:
                    values.append(word)
                continue

            # start with the full word
            current_word = word

            while current_word:

                # iterate backwards through the word
                i = len(current_word)

                while i > 0:

                    # see if the token is in the vocab
                    token_value = self.lookup_table_search(current_word[:i])
                
                    # if it's not in the vocab, then move back a character
                    if token_value == -1:

                        # if we've reached the end of the word, then no token exists
                        ## add the unknown token
                        if i == 1:
                            if return_integers:
                                values.append(0)
                            else:
                                values.append('<unknown>')
                            current_word = current_word[i:]

                        i -= 1
                        

                    # if it is in the vocab, then add it to token_values
                    # and update current_word to be the remaining characters after the token
                    else:
                        if return_integers:
                            values.append(token_value)
                        else:
                            values.append(current_word[:i])
                        current_word = current_word[i:]
                        break


        return values


    def decode(self, token_values: List[int], integers=True) -> str:
        '''
        Given encoded tokens, decodes them using the vocabulary.

        Parameters
        ----------
        token_values : List[int]
            The encoded tokens.
        integers : bool | True
            Whether the tokens are encoded as integers. Default is True.

        Returns
        -------
        str
            The decoded text.
        '''

        special = {
            "<endoftext>": '',
            "<newline>": "\n",
            "<space>": " ",
            "<tab>": "\t"
        }

        decoded_text = ''

        # iterate through the token values
        for token_value in token_values:

            if integers:
                decoded_token = self.vocab[token_value]
            else:
                decoded_token = token_value

            if decoded_token in special:
                decoded_text += special[decoded_token]
            else:
                decoded_text += decoded_token

        return decoded_text

        

class NaiveBPETokenizer(BPETokenizer):
    '''
    A BPE tokenizer that uses a naive approach - having whitespace and punctuation as separate tokens, 
    that don't get included into the merges.
    '''

    def __init__(self, 
                 model_dir=None,
                 vocab_file=None,
                 lookup_table_file=None,
                 corpus=None,
                 vocab_size=1024):
        '''
        Initialize the NaiveBPETokenizer.
        
        Parameters
        ----------
        model_dir : str | None
            The directory to save the model. Default is None.
        vocab_file : str | None
            The file to read the vocabulary from. Default is None.
        lookup_table_file : str | None
            The file to read the lookup table from. Default is None.
        corpus : list | None
            The corpus used to train the tokenizer. Default is None.
        vocab_size : int | 1024
            The size of the vocabulary. Default is 1024.
        '''

        super().__init__(model_dir=model_dir, 
                         vocab_file=vocab_file, 
                         lookup_table_file=lookup_table_file,
                         corpus=corpus, 
                         vocab_size=vocab_size)
        
        # special characters
        special_char_filepath = os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'naive_bpe_special_char.json')
        self.special_characters = self.read_special_char_from_file(special_char_filepath)


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
    
