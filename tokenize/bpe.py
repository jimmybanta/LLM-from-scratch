from unidecode import unidecode
from typing import List
import re

class BPETokenizer:
    '''
    A tokenizer that uses Byte Pair Encoding (BPE) to tokenize text.
    '''
    
    def __init__(self):

        pass


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
    
    