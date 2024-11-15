from unidecode import unidecode
from typing import List
import re

class BPETokenizer:
    '''
    A tokenizer that uses Byte Pair Encoding (BPE) to tokenize text.
    '''
    
    def __init__(self, corpus=None):
        '''
        Initialize the BPETokenizer.
        
        Parameters
        ----------
        corpus : str | None
            The corpus used to train the tokenizer. Default is None.
        '''

        # the corpus used to train the tokenizer
        self.corpus = corpus


    def normalize(self, text: str, 
                  lowercase=False) -> str:
        '''
        Normalize the input text.

        Parameters
        ----------
        text : str
            The input text.
        lowercase : bool | False
            Whether to lowercase the text. Default is False.

        Returns
        -------
        str
            The normalized text.
        '''

        # convert to ASCII
        text = unidecode(text)

        # lowercase the text, if required
        if lowercase:
            return text.lower()
        
        return text
    
    
    

class NaiveBPETokenizer(BPETokenizer):
    '''
    A BPE tokenizer that uses a naive approach - having whitespace and punctuation as separate tokens, 
    that don't get included into the merges.
    '''

    def __init__(self, corpus=None):
        '''
        Initialize the NaiveBPETokenizer.
        
        Parameters
        ----------
        corpus : str | None
            The corpus used to train the tokenizer. Default is None.
        '''

        super().__init__(corpus)

    def pre_tokenize(self, text: str) -> List[str]:
        '''
        Pre-tokenize the input text.
        Splits along whitespace (space, newline, tab) and punctuation/special characters.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        List[str]
            The pre-tokenized text.
        '''

        space_dict = {
            ' ': '<space>',
            '\n': '<newline>',
            '\t': '<tab>'
        }

        # remove leading and trailing white space
        text = text.strip()

        # words_split will store the pre-tokenized text
        words_split = []

        # traverse through the characters of the text string
        current_word = ''
        for char in text:

            # space, newline, tab
            if char in [' ', '\n', '\t']:
                    
                # if current_word is not empty, add it to words_split
                if current_word != '':
                    words_split.append(current_word)

                # add the special character to words_split
                words_split.append(space_dict[char])

                # reset current_word
                current_word = ''
            
            # check for punctuation
            elif bool(re.match(r'[^\w\s]', char)):

                if current_word != '':
                    words_split.append(current_word)

                words_split.append(char)
                current_word = ''
            
            # otherwise, add the character to current_word
            else:
                current_word += char
        
        # add the last word to words_split
        if current_word != '':
            words_split.append(current_word)
        
        # add end of text token
        words_split.append('<endoftext>')

        return words_split