import pytest
import os
import json
from dotenv import load_dotenv

from pre_process.tokenize.bpe import BPETokenizer, NaiveBPETokenizer

load_dotenv()

# tokens to use to test lookup methods
TEST_TOKENS = [
        "<pad>",
        "<unknown>",
        "<endoftext>",
        "<newline>",
        "<space>",
        "<tab>",
        "*",
        "0",
        "A",
        "H",
        "items", 
        "nature",
        "spite",
        "uses",
        "zing",
        "zipper",
        "zy",
        "zz",
        "zzle",
        "thisisatokenthatdoesntexist"
    ]

EXPECTED_INDICES = {
    'tiny': [0, 1, 2, 3, 4, 5, 15, -1, -1, 37, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    'small': [0, 1, 2, 3, 4, 5, 15, 37, 47, 62, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    'medium': [0, 1, 2, 3, 4, 5, 15, 37, 50, 289, 3048, 3579, 4757, 5357, 5662, 5663, 5664, 5665, 5666, -1]
}

FILEPATHS = {
    'tiny_vocab': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'tiny_vocab.json'),
    'tiny_lookup_table': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'tiny_lookup_table.json'),

    'small_lookup_table': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'small_lookup_table.json'),
    'small_vocab': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'small_vocab.json'),

    'medium_vocab': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_vocab.json'),
    'medium_lookup_table': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_lookup_table.json'),

    'naive_special_char': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'naive_bpe_special_char.json'),

    'encode_text': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_text.txt'),
    'encode_token_values': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_token_values.json'),
    'encode_integer_values': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_integer_values.json')
}

@pytest.fixture
def bpe():
    return BPETokenizer()

@pytest.fixture
def naive_bpe():
    return NaiveBPETokenizer()

@pytest.fixture
def naive_bpe_tiny_vocab():
    return NaiveBPETokenizer(
        vocab_file=FILEPATHS['tiny_vocab'],
        lookup_table_file=FILEPATHS['tiny_lookup_table'],
        vocab_size=175
    )

@pytest.fixture
def naive_bpe_small_vocab():
    return NaiveBPETokenizer(
        vocab_file=FILEPATHS['small_vocab'],
        lookup_table_file=FILEPATHS['small_lookup_table'],
        vocab_size=1024
    )

@pytest.fixture
def naive_bpe_medium_vocab():
    return NaiveBPETokenizer(
        vocab_file=FILEPATHS['medium_vocab'],
        lookup_table_file=FILEPATHS['medium_lookup_table'],
        vocab_size=5667,
        context_size=100
    )


def test_bpe_init():

    # Test case 1: Initialize with default parameters
    bpe = BPETokenizer()
    assert bpe.corpus is None
    assert bpe.vocab == []
    assert bpe.vocab_size == 1024

    # Test case 2: Initialize with custom parameters
    corpus = ["This is a test corpus."]
    vocab_file = FILEPATHS['small_vocab']
    bpe = BPETokenizer(corpus=corpus, vocab_file=vocab_file, vocab_size=500)
    assert bpe.corpus == corpus
    assert bpe.vocab_size == 500
    assert bpe.vocab == bpe.load_from_file(vocab_file)

    # Test case 3: Initialize with only vocab_file
    bpe = BPETokenizer(vocab_file=vocab_file)
    assert bpe.vocab == bpe.load_from_file(vocab_file)

    # Test case 4: Initialize with only corpus
    bpe = BPETokenizer(corpus=corpus)
    assert bpe.corpus == corpus
    assert bpe.vocab == []
 
def test_naive_bpe_init():

    # Test case 1: Initialize with default parameters
    bpe = NaiveBPETokenizer()
    assert bpe.corpus is None
    assert bpe.vocab == []
    assert bpe.vocab_size == 1024
    assert bpe.special_characters == bpe.load_special_char_from_file(FILEPATHS['naive_special_char'])

    # Test case 2: Initialize with custom parameters
    corpus = ["This is a test corpus."]
    vocab_file = FILEPATHS['small_vocab']
    bpe = NaiveBPETokenizer(corpus=corpus, vocab_file=vocab_file, vocab_size=500)
    assert bpe.corpus == corpus
    assert bpe.vocab_size == 500
    assert bpe.vocab == bpe.load_from_file(vocab_file)
    assert bpe.special_characters == bpe.load_special_char_from_file(FILEPATHS['naive_special_char'])

    # Test case 3: Initialize with only vocab_file
    bpe = NaiveBPETokenizer(vocab_file=vocab_file)
    assert bpe.vocab == bpe.load_from_file(vocab_file)
    assert bpe.special_characters == bpe.load_special_char_from_file(FILEPATHS['naive_special_char'])

    # Test case 4: Initialize with only corpus
    bpe = NaiveBPETokenizer(corpus=corpus)
    assert bpe.corpus == corpus
    assert bpe.vocab == []
    assert bpe.special_characters == bpe.load_special_char_from_file(FILEPATHS['naive_special_char'])


def test_normalize(bpe):

    # Test case 1: Normal text without lowercase
    text = ["Café"]
    expected = ["Cafe"]
    assert bpe.normalize(text) == expected

    # Test case 2: Normal text with lowercase
    text = ["Café"]
    expected = ["cafe"]
    assert bpe.normalize(text, lowercase=True) == expected

    # Test case 3: Text with special characters
    text = ["naïve"]
    expected = ["naive"]
    assert bpe.normalize(text) == expected

    # Test case 4: Text with special characters and lowercase
    text = ["naïve"]
    expected = ["naive"]
    assert bpe.normalize(text, lowercase=True) == expected

    # Test case 5: Text with uppercase letters
    text = ["HELLO"]
    expected = ["HELLO"]
    assert bpe.normalize(text) == expected

    # Test case 6: Text with uppercase letters and lowercase
    text = ["HELLO"]
    expected = ["hello"]
    assert bpe.normalize(text, lowercase=True) == expected

def test_naive_pre_tokenize(naive_bpe):

    # Test case 1: Simple sentence
    text = ["Hello, world!"]
    expected = [["Hello", ",", "<space>", "world", "!", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 2: Sentence with punctuation
    text = ["How's it going?"]
    expected = [["How", "'", "s", "<space>", "it", "<space>", "going", "?", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 3: Sentence with multiple punctuation marks
    text = ["Wait... What?!"]
    expected = [["Wait", ".", ".", ".", "<space>", "What", "?", "!", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 4: Sentence with numbers
    text = ["The price is $5.99."]
    expected = [["The", "<space>", "price", "<space>", "is", "<space>", "$", "5", ".", "99", ".", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 5: Sentence with mixed characters
    text = ["Café-naïve"]
    expected = [["Café", "-", "naïve", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 6: Sentence with special characters and whitespace
    text = ["Hello\tworld\nHow are you?"]
    expected = [["Hello", "<tab>", "world", "<newline>", "How", "<space>", "are", "<space>", "you", "?", "<endoftext>"]]
    assert naive_bpe.pre_tokenize(text) == expected

    # Test case 7: complex sentence
    text = ['''     
         This is a sample text.. \n\t\n 
            I want to includ0e special characters like !@#$%^&*()_+{}|:"<>?[]\\;',./`~ and numbers like 1234567890. 
to make sure they're all pre-tokenized correctly.

        bla bla bla 10$10
    ''']
    expected = [['This', '<space>', 'is', '<space>', 'a', '<space>', 'sample', '<space>', 'text', 
                '.', '.', '<space>', '<newline>', '<tab>', '<newline>', '<space>', '<newline>', '<space>', 
                '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
                '<space>', '<space>', '<space>', 'I', '<space>', 'want', '<space>', 'to', '<space>', 
                'includ0e', '<space>', 'special', '<space>', 'characters', '<space>', 'like', '<space>', 
                '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', 
                '<', '>', '?', '[', ']', '\\', ';', "'", ',', '.', '/', '`', '~', '<space>', 'and', 
                '<space>', 'numbers', '<space>', 'like', '<space>', '1234567890', '.', '<space>', 
                '<newline>', 'to', '<space>', 'make', '<space>', 'sure', '<space>', 'they', "'", 're', 
                '<space>', 'all', '<space>', 'pre', '-', 'tokenized', '<space>', 'correctly', '.', 
                '<newline>', '<newline>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
                '<space>', '<space>', 'bla', '<space>', 'bla', '<space>', 'bla', '<space>', 
                '10', '$', '10', '<endoftext>']]
    
    # Test case 8: batch of sentences
    text = [
        'This is a sample text.. \n\t\n',
        "I want to includ0e special characters like !@#$%^&*()_+{}|:\"<>?[]\\;',./`~ and numbers like 1234567890. to make sure they\'re all pre-tokenized correctly.",
        'bla bla bla 10$10'
    ]
    expected = [['This', '<space>', 'is', '<space>', 'a', '<space>', 'sample', '<space>', 'text', '.', '.', '<endoftext>'], 
                ['I', '<space>', 'want', '<space>', 'to', '<space>', 'includ0e', '<space>', 
                 'special', '<space>', 'characters', '<space>', 'like', '<space>', '!', '@',
                   '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', 
                   '<', '>', '?', '[', ']', '\\', ';', "'", ',', '.', '/', '`', '~', '<space>',
                     'and', '<space>', 'numbers', '<space>', 'like', '<space>', '1234567890', 
                     '.', '<space>', 'to', '<space>', 'make', '<space>', 'sure', '<space>', 'they',
                       "'", 're', '<space>', 'all', '<space>', 'pre', '-', 'tokenized', '<space>',
                         'correctly', '.', '<endoftext>'], 
                ['bla', '<space>', 'bla', '<space>', 'bla', '<space>', '10', '$', '10', '<endoftext>']]


    assert naive_bpe.pre_tokenize(text) == expected
    
def test_get_vocab_word_counts(naive_bpe):

    # Test case 1: complex batch of 2 sentences
    text = [['This', '<space>', 'is', '<space>', 'a', '<space>', 'sample', '<space>', 
             'text', '.', '.', '<space>', '<newline>', '<tab>', '<newline>', '<space>',
               '<newline>', '<space>', '<space>', '<space>', '<space>', '<space>', 
               '<space>', '<space>', '<space>', '<space>', '<space>', '<space>',
                 '<space>', 'I', '<space>', 'want', '<space>', 'to', '<space>', 
                 'includ0e', '<space>', 'special', '<space>', 'characters', 
                 '<space>', 'like', '<space>', '!', '@', '#', '$', '%', '^', '&', '*',
                   '(', ')', '_', '+', '{', '}', '|', ':', '"', '<', '>', '?', '[', ']', 
                   ';', "'", ',', '.', '/', '`', '~', '<space>', 'and', '<space>', 
                   'numbers', '<space>', 'like', '<space>', '1234567890', '.', '<newline>',
                     '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
                     '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
                     'to', '<space>', 'make', '<space>', 'sure', '<space>', 'they', "'", 
                     're', '<space>', 'all', '<space>', 'pre', '-', 'tokenized', '<space>', 
                     'correctly', '.', '<endoftext>'], 
                     ['bla', '<space>', 'bla', '<space>',
                        'bla', '<space>', '10', '$', '10', '<newline>',
                          '<newline>', '<space>', '<space>', '<space>', '<space>',
                            '<space>', '<space>', '<space>', '<space>', '<space>', 
                            '<space>', '<space>', '<space>', 'the', '<space>', 'quick',
                            '<space>', 'brown', '<space>', 'fox', '<space>', 'jumps',
                            '<space>', 'over', '<space>', 'the', '<space>', 'lazy',
                            '<space>', 'dog', '.', '<newline>', '<space>', '<space>',
                            '<space>', '<space>', '<space>', '<space>', '<space>',
                            '<space>', '<space>', '<space>', '<space>', '<space>', 
                            'the', '<space>', 'hen', '<space>', 'is', '<space>', 
                            'in', '<space>', 'the', '<space>', 'pen', '.', '<endoftext>']]

    expected = (
        ['T', 'h', 'i', 's', 'a', 'm', 'p', 'l', 'e', 't', 'x', 'I', 'w', 'n', 'o', 'c', 'u',
          'd', '0', 'r', 'k', 'b', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'y', 'z', 'q',
            'f', 'j', 'v', 'g'],
        {
        'This': {'count': 1, 'tokens': ['T', 'h', 'i', 's']},
        'is': {'count': 2, 'tokens': ['i', 's']},
        'a': {'count': 1, 'tokens': ['a']},
        'sample': {'count': 1, 'tokens': ['s', 'a', 'm', 'p', 'l', 'e']},
        'text': {'count': 1, 'tokens': ['t', 'e', 'x', 't']},
        'I': {'count': 1, 'tokens': ['I']},
        'want': {'count': 1, 'tokens': ['w', 'a', 'n', 't']},
        'to': {'count': 2, 'tokens': ['t', 'o']},
        'includ0e': {'count': 1, 'tokens': ['i', 'n', 'c', 'l', 'u', 'd', '0', 'e']},
        'special': {'count': 1, 'tokens': ['s', 'p', 'e', 'c', 'i', 'a', 'l']},
        'characters': {'count': 1, 'tokens': ['c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', 's']},
        'like': {'count': 2, 'tokens': ['l', 'i', 'k', 'e']},
        'and': {'count': 1, 'tokens': ['a', 'n', 'd']},
        'numbers': {'count': 1, 'tokens': ['n', 'u', 'm', 'b', 'e', 'r', 's']},
        '1234567890': {'count': 1, 'tokens': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']},
        'make': {'count': 1, 'tokens': ['m', 'a', 'k', 'e']},
        'sure': {'count': 1, 'tokens': ['s', 'u', 'r', 'e']},
        'they': {'count': 1, 'tokens': ['t', 'h', 'e', 'y']},
        're': {'count': 1, 'tokens': ['r', 'e']},
        'all': {'count': 1, 'tokens': ['a', 'l', 'l']},
        'pre': {'count': 1, 'tokens': ['p', 'r', 'e']},
        'tokenized': {'count': 1, 'tokens': ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'e', 'd']},
        'correctly': {'count': 1, 'tokens': ['c', 'o', 'r', 'r', 'e', 'c', 't', 'l', 'y']},
        'bla': {'count': 3, 'tokens': ['b', 'l', 'a']},
        '10': {'count': 2, 'tokens': ['1', '0']},
        'the': {'count': 4, 'tokens': ['t', 'h', 'e']},
        'quick': {'count': 1, 'tokens': ['q', 'u', 'i', 'c', 'k']},
        'brown': {'count': 1, 'tokens': ['b', 'r', 'o', 'w', 'n']},
        'fox': {'count': 1, 'tokens': ['f', 'o', 'x']},
        'jumps': {'count': 1, 'tokens': ['j', 'u', 'm', 'p', 's']},
        'over': {'count': 1, 'tokens': ['o', 'v', 'e', 'r']},
        'lazy': {'count': 1, 'tokens': ['l', 'a', 'z', 'y']},
        'dog': {'count': 1, 'tokens': ['d', 'o', 'g']},
        'hen': {'count': 1, 'tokens': ['h', 'e', 'n']},
        'in': {'count': 1, 'tokens': ['i', 'n']},
        'pen': {'count': 1, 'tokens': ['p', 'e', 'n']}
    }
    )

    assert naive_bpe.get_vocab_word_counts(text) == expected

def test_update_vocab(naive_bpe):

    # Test case 1: Update vocab with new words
    
    text = {
        'This': {'count': 1, 'tokens': ['T', 'h', 'i', 's']},
        'is': {'count': 2, 'tokens': ['i', 's']},
        'a': {'count': 1, 'tokens': ['a']},
        'sample': {'count': 1, 'tokens': ['s', 'a', 'm', 'p', 'l', 'e']},
        'text': {'count': 1, 'tokens': ['t', 'e', 'x', 't']},
        'I': {'count': 1, 'tokens': ['I']},
        'want': {'count': 1, 'tokens': ['w', 'a', 'n', 't']},
        'to': {'count': 2, 'tokens': ['t', 'o']},
        'includ0e': {'count': 1, 'tokens': ['i', 'n', 'c', 'l', 'u', 'd', '0', 'e']},
        'special': {'count': 1, 'tokens': ['s', 'p', 'e', 'c', 'i', 'a', 'l']},
        'characters': {'count': 1, 'tokens': ['c', 'h', 'a', 'r', 'a', 'c', 't', 'e', 'r', 's']},
        'like': {'count': 2, 'tokens': ['l', 'i', 'k', 'e']},
        'and': {'count': 1, 'tokens': ['a', 'n', 'd']},
        'numbers': {'count': 1, 'tokens': ['n', 'u', 'm', 'b', 'e', 'r', 's']},
        '1234567890': {'count': 1, 'tokens': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']},
        'make': {'count': 1, 'tokens': ['m', 'a', 'k', 'e']},
        'sure': {'count': 1, 'tokens': ['s', 'u', 'r', 'e']},
        'they': {'count': 1, 'tokens': ['t', 'h', 'e', 'y']},
        're': {'count': 1, 'tokens': ['r', 'e']},
        'all': {'count': 1, 'tokens': ['a', 'l', 'l']},
        'pre': {'count': 1, 'tokens': ['p', 'r', 'e']},
        'tokenized': {'count': 1, 'tokens': ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'e', 'd']},
        'correctly': {'count': 1, 'tokens': ['c', 'o', 'r', 'r', 'e', 'c', 't', 'l', 'y']},
        'bla': {'count': 3, 'tokens': ['b', 'l', 'a']},
        '10': {'count': 2, 'tokens': ['1', '0']},
        'the': {'count': 4, 'tokens': ['t', 'h', 'e']},
        'quick': {'count': 1, 'tokens': ['q', 'u', 'i', 'c', 'k']},
        'brown': {'count': 1, 'tokens': ['b', 'r', 'o', 'w', 'n']},
        'fox': {'count': 1, 'tokens': ['f', 'o', 'x']},
        'jumps': {'count': 1, 'tokens': ['j', 'u', 'm', 'p', 's']},
        'over': {'count': 1, 'tokens': ['o', 'v', 'e', 'r']},
        'lazy': {'count': 1, 'tokens': ['l', 'a', 'z', 'y']},
        'dog': {'count': 1, 'tokens': ['d', 'o', 'g']},
        'hen': {'count': 1, 'tokens': ['h', 'e', 'n']},
        'in': {'count': 1, 'tokens': ['i', 'n']},
        'pen': {'count': 1, 'tokens': ['p', 'e', 'n']}
    }
    expected = [
        'he', 'the', 'ke', 're', 'la', 'is', 'to', 'er', 'bla', 'mp', 'an', 'in', 'pe', 'al', 'ct', 'ers', 
        'li', 'like', '10', 'Th', 'This', 'sa', 'samp', 'sampl', 'sample', 'te', 'tex', 'text', 'wan', 
        'want', 'inc', 'incl', 'inclu', 'includ', 'includ0', 'includ0e', 'spe', 'spec', 'speci', 'special', 
        'ch', 'cha', 'char', 'chara', 'charact', 'characters', 'and', 'nu', 'num', 'numb', 'numbers', '12', 
        '123', '1234', '12345', '123456', '1234567', '12345678', '123456789', '1234567890', 'ma', 'make', 
        'su', 'sure', 'they', 'all', 'pre', 'toke', 'token', 'tokeni', 'tokeniz', 'tokenize', 'tokenized', 
        'co', 'cor', 'corre', 'correct', 'correctl', 'correctly', 'qu', 'qui', 'quic', 'quick', 'br', 'bro', 
        'brow', 'brown', 'fo', 'fox', 'ju', 'jump', 'jumps', 'ov', 'over', 'laz', 'lazy', 'do', 'dog', 'hen', 
        'pen'
    ]

    assert naive_bpe.update_vocab(text) == expected

def test_generate_lookup_table(naive_bpe_tiny_vocab, naive_bpe_small_vocab, naive_bpe_medium_vocab):

    # Test case 1: tiny vocab
    naive_bpe_tiny_vocab.generate_lookup_table()
    expected = naive_bpe_tiny_vocab.load_from_file(FILEPATHS['tiny_lookup_table'])
    assert naive_bpe_tiny_vocab.lookup_table == expected

    # Test case 2: small vocab
    naive_bpe_small_vocab.generate_lookup_table()
    expected = naive_bpe_small_vocab.load_from_file(FILEPATHS['small_lookup_table'])
    assert naive_bpe_small_vocab.lookup_table == expected

    # Test case 3: medium vocab
    naive_bpe_medium_vocab.generate_lookup_table()
    expected = naive_bpe_medium_vocab.load_from_file(FILEPATHS['medium_lookup_table'])
    assert naive_bpe_medium_vocab.lookup_table == expected


def test_lookup_brute_search(naive_bpe_tiny_vocab, naive_bpe_small_vocab, naive_bpe_medium_vocab):
    
    # Test case 1: tiny vocab
    expected = EXPECTED_INDICES['tiny']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_tiny_vocab.lookup_brute_search(token)
        assert idx == expected[i]

    # Test case 2: small vocab
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_small_vocab.lookup_brute_search(token)
        assert idx == expected[i]

    # Test case 3: medium vocab
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_medium_vocab.lookup_brute_search(token)
        assert idx == expected[i]


def test_lookup_binary_search(naive_bpe_tiny_vocab, naive_bpe_small_vocab, naive_bpe_medium_vocab):
    
    # Test case 1: tiny vocab
    expected = EXPECTED_INDICES['tiny']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_tiny_vocab.lookup_binary_search(token)
        assert idx == expected[i]

    # Test case 2: small vocab
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_small_vocab.lookup_binary_search(token)
        assert idx == expected[i]

    # Test case 3: medium vocab
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_medium_vocab.lookup_binary_search(token)
        assert idx == expected[i]

def test_lookup_table_search(naive_bpe_tiny_vocab, naive_bpe_small_vocab, naive_bpe_medium_vocab):
    
    # Test case 1: tiny vocab
    expected = EXPECTED_INDICES['tiny']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_tiny_vocab.lookup_table_search(token)
        assert idx == expected[i]

    # Test case 2: small vocab
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_small_vocab.lookup_table_search(token)
        assert idx == expected[i]

    # Test case 3: medium vocab
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = naive_bpe_medium_vocab.lookup_table_search(token)
        assert idx == expected[i]
    

def test_encode(naive_bpe_medium_vocab):
    
    # Test case 1: simple sentence, encoded as integers
    text = ['The mitochondria is the powerhouse of the cell!']
    expected = [[637, 4, 3488, 3680, 2711, 1942, 729, 4, 3015, 4, 5054, 4, 4011, 2735, 4, 3690, 4, 5054, 4, 1448, 6, 2]]
    assert naive_bpe_medium_vocab.encode(text, return_value='integers') == expected

    # Test case 2: paragraph, encoded as integers
    text = []
    with open(FILEPATHS['encode_text'], 'r') as file:
        for line in file.readlines():
            text.append(line)
    
    with open(FILEPATHS['encode_integer_values'], 'r') as file:
        expected = json.load(file)

    assert naive_bpe_medium_vocab.encode(text, return_value='integers') == expected


    # Test case 3: simple sentence, encoded as tokens
    text = ['The mitochondria is the powerhouse of the cell!']
    expected = [['The', '<space>', 'mit', 'oc', 'hon', 'dri', 'a', '<space>', 
                'is', '<space>', 'the', '<space>', 'power', 'house', '<space>', 
                'of', '<space>', 'the', '<space>', 'cell', '!', '<endoftext>']]
    assert naive_bpe_medium_vocab.encode(text, return_value='words') == expected

    # Test case 4: paragraph, encoded as tokens
    text = []
    with open(FILEPATHS['encode_text'], 'r') as file:
        for line in file.readlines():
            text.append(line)

    with open(FILEPATHS['encode_token_values'], 'r') as file:
        expected = json.load(file)
    
    assert naive_bpe_medium_vocab.encode(text, return_value='words') == expected

    # Test case 3: simple sentence, encoded as both
    text = ['The mitochondria is the powerhouse of the cell!']
    expected = [
        [('The', 637), ('<space>', 4), ('mit', 3488), ('oc', 3680), ('hon', 2711), ('dri', 1942), ('a', 729), 
         ('<space>', 4), ('is', 3015), ('<space>', 4), ('the', 5054), ('<space>', 4), ('power', 4011), 
         ('house', 2735), ('<space>', 4), ('of', 3690), ('<space>', 4), ('the', 5054), ('<space>', 4), 
         ('cell', 1448), ('!', 6), ('<endoftext>', 2)]
    ]
    assert naive_bpe_medium_vocab.encode(text, return_value='both') == expected

def test_decode(naive_bpe_medium_vocab):

    # Test case 1: simple sentence
    text = [[637, 4, 3488, 3680, 2711, 1942, 729, 4, 3015, 4, 5054, 4, 4011, 2735, 4, 3690, 4, 5054, 4, 1448, 6, 2]]
    expected = ['The mitochondria is the powerhouse of the cell!']
    assert naive_bpe_medium_vocab.decode(text) == expected

    # Test case 2: paragraph
    with open(FILEPATHS['encode_integer_values'], 'r') as file:
        batch = json.load(file)
    
    expected = []
    with open(FILEPATHS['encode_text'], 'r') as file:
        for line in file.readlines():
            expected.append(line.strip())

    assert naive_bpe_medium_vocab.decode(batch) == expected


if __name__ == "__main__":
    pytest.main()