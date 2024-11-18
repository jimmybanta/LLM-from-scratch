import pytest
import os
import json
from dotenv import load_dotenv

from tokenize.bpe import BPETokenizer, NaiveBPETokenizer

load_dotenv()

# tokens to use to test lookup methods
TEST_TOKENS = [
        "<unknown>",
        "<endoftext>",
        "<newline>",
        "<space>",
        "<tab>",
        "*",
        "0",
        "A",
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
    'small': [0, 1, 2, 3, 4, 14, 36, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    'medium': [0, 1, 2, 3, 4, 14, 36, 46, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    'large': [0, 1, 2, 3, 4, 14, 36, 49, 3047, 3578, 4756, 5356, 5661, 5662, 5663, 5664, 5665, -1]
}

def test_bpe_init():

    # Test case 1: Initialize with default parameters
    bpe = BPETokenizer()
    assert bpe.model_dir is None
    assert bpe.corpus is None
    assert bpe.vocab == []
    assert bpe.vocab_size == 1024

    # Test case 2: Initialize with custom parameters
    model_dir = os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'model_dirs', 'test_bpe_model_dir')
    corpus = ["This is a test corpus."]
    vocab_file = os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json')
    bpe = BPETokenizer(model_dir=model_dir, corpus=corpus, vocab_file=vocab_file, vocab_size=500)
    assert bpe.model_dir == model_dir
    assert bpe.corpus == corpus
    assert bpe.vocab_size == 500
    assert bpe.vocab == bpe.read_from_file(vocab_file)

    # Clean up: Delete the model directory if it exists
    if os.path.exists(model_dir):
        os.rmdir(model_dir)

    # Test case 3: Initialize with only vocab_file
    bpe = BPETokenizer(vocab_file=vocab_file)
    assert bpe.vocab == bpe.read_from_file(vocab_file)

    # Test case 4: Initialize with only model_dir
    bpe = BPETokenizer(model_dir=model_dir)
    assert bpe.model_dir == model_dir
    assert bpe.vocab == []

    # Clean up: Delete the model directory if it exists
    if os.path.exists(model_dir):
        os.rmdir(model_dir)

    # Test case 5: Initialize with only corpus
    bpe = BPETokenizer(corpus=corpus)
    assert bpe.corpus == corpus
    assert bpe.vocab == []

def test_naive_bpe_init():
    # Test case 1: Initialize with default parameters
    bpe = NaiveBPETokenizer()
    assert bpe.model_dir is None
    assert bpe.corpus is None
    assert bpe.vocab == []
    assert bpe.vocab_size == 1024
    assert bpe.special_characters == bpe.read_special_char_from_file(
        os.path.join(os.getenv('HOME_DIR'), 'tokenize', 'assets', 'naive_bpe_special_char.json')
    )

    # Test case 2: Initialize with custom parameters
    model_dir = os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'model_dirs', 'test_bpe_model_dir')
    corpus = ["This is a test corpus."]
    vocab_file = os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json')
    bpe = NaiveBPETokenizer(model_dir=model_dir, corpus=corpus, vocab_file=vocab_file, vocab_size=500)
    assert bpe.model_dir == model_dir
    assert bpe.corpus == corpus
    assert bpe.vocab_size == 500
    assert bpe.vocab == bpe.read_from_file(vocab_file)
    assert bpe.special_characters == bpe.read_special_char_from_file(
        os.path.join(os.getenv('HOME_DIR'), 'tokenize', 'assets', 'naive_bpe_special_char.json')
    )

    # Clean up: Delete the model directory if it exists
    if os.path.exists(model_dir):
        os.rmdir(model_dir)

    # Test case 3: Initialize with only vocab_file
    bpe = NaiveBPETokenizer(vocab_file=vocab_file)
    assert bpe.vocab == bpe.read_from_file(vocab_file)
    assert bpe.special_characters == bpe.read_special_char_from_file(
        os.path.join(os.getenv('HOME_DIR'), 'tokenize', 'assets', 'naive_bpe_special_char.json')
    )

    # Test case 4: Initialize with only model_dir
    bpe = NaiveBPETokenizer(model_dir=model_dir)
    assert bpe.model_dir == model_dir
    assert bpe.vocab == []
    assert bpe.special_characters == bpe.read_special_char_from_file(
        os.path.join(os.getenv('HOME_DIR'), 'tokenize', 'assets', 'naive_bpe_special_char.json')
    )

    # Clean up: Delete the model directory if it exists
    if os.path.exists(model_dir):
        os.rmdir(model_dir)

    # Test case 5: Initialize with only corpus
    bpe = NaiveBPETokenizer(corpus=corpus)
    assert bpe.corpus == corpus
    assert bpe.vocab == []
    assert bpe.special_characters == bpe.read_special_char_from_file(
        os.path.join(os.getenv('HOME_DIR'), 'tokenize', 'assets', 'naive_bpe_special_char.json')
    )


def test_normalize():
    bpe = BPETokenizer()

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

def test_naive_pre_tokenize():
    bpe = NaiveBPETokenizer()

    # Test case 1: Simple sentence
    text = ["Hello, world!"]
    expected = ["Hello", ",", "<space>", "world", "!", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 2: Sentence with punctuation
    text = ["How's it going?"]
    expected = ["How", "'", "s", "<space>", "it", "<space>", "going", "?", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 3: Sentence with multiple punctuation marks
    text = ["Wait... What?!"]
    expected = ["Wait", ".", ".", ".", "<space>", "What", "?", "!", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 4: Sentence with numbers
    text = ["The price is $5.99."]
    expected = ["The", "<space>", "price", "<space>", "is", "<space>", "$", "5", ".", "99", ".", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 5: Sentence with mixed characters
    text = ["Café-naïve"]
    expected = ["Café", "-", "naïve", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 6: Sentence with special characters and whitespace
    text = ["Hello\tworld\nHow are you?"]
    expected = ["Hello", "<tab>", "world", "<newline>", "How", "<space>", "are", "<space>", "you", "?", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 7: complex sentence
    text = ['''     
         This is a sample text.. \n\t\n 
            I want to includ0e special characters like !@#$%^&*()_+{}|:"<>?[]\;',./`~ and numbers like 1234567890. 
to make sure they're all pre-tokenized correctly.

        bla bla bla 10$10
    ''']
    expected = ['This', '<space>', 'is', '<space>', 'a', '<space>', 'sample', '<space>', 'text', 
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
                '10', '$', '10', '<endoftext>']
    
def test_get_vocab_word_counts():
    bpe = NaiveBPETokenizer()

    # Test case 1: complex sentence
    text = [
        'This', '<space>', 'is', '<space>', 'a', '<space>', 'sample', '<space>', 'text', '.', '.', '<space>', 
        '<newline>', '<tab>', '<newline>', '<space>', '<newline>', '<space>', '<space>', '<space>', '<space>', 
        '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'I', '<space>', 
        'want', '<space>', 'to', '<space>', 'includ0e', '<space>', 'special', '<space>', 'characters', '<space>', 
        'like', '<space>', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':', '"', 
        '<', '>', '?', '[', ']', ';', "'", ',', '.', '/', '`', '~', '<space>', 'and', '<space>', 'numbers', 
        '<space>', 'like', '<space>', '1234567890', '.', '<newline>', '<space>', '<space>', '<space>', '<space>', 
        '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'to', '<space>', 
        'make', '<space>', 'sure', '<space>', 'they', "'", 're', '<space>', 'all', '<space>', 'pre', '-', 
        'tokenized', '<space>', 'correctly', '.', '<newline>', '<newline>', '<space>', '<space>', '<space>', 
        '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'bla', 
        '<space>', 'bla', '<space>', 'bla', '<space>', '10', '$', '10', '<newline>', '<newline>', '<space>', 
        '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
        '<space>', '<space>', 'the', '<space>', 'quick', '<space>', 'brown', '<space>', 'fox', '<space>', 'jumps', 
        '<space>', 'over', '<space>', 'the', '<space>', 'lazy', '<space>', 'dog', '.', '<newline>', '<space>', 
        '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 
        '<space>', '<space>', 'the', '<space>', 'hen', '<space>', 'is', '<space>', 'in', '<space>', 'the', 
        '<space>', 'pen', '.', '<endoftext>'
    ]

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

    assert bpe.get_vocab_word_counts(text) == expected

def test_update_vocab():
    bpe = NaiveBPETokenizer()

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

    assert bpe.update_vocab(text) == expected

def test_generate_lookup_table():

    
    # Test case 1: small vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json'),
        vocab_size=175
        )
    bpe.generate_lookup_table()
    expected = bpe.read_from_file(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_lookup_table.json'),)
    assert bpe.lookup_table == expected

    # Test case 2: medium vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_vocab.json'),
        vocab_size=1024
        )
    bpe.generate_lookup_table()
    expected = bpe.read_from_file(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_lookup_table.json'),)
    assert bpe.lookup_table == expected

    # Test case 3: large vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        vocab_size=5667
        )
    bpe.generate_lookup_table()
    expected = bpe.read_from_file(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_lookup_table.json'),)
    assert bpe.lookup_table == expected


def test_lookup_brute_search():
    
    # Test case 1: small vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json'),
        vocab_size=175
        )
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_brute_search(token)
        assert idx == expected[i]

    # Test case 2: medium vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_vocab.json'),
        vocab_size=1024
        )
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_brute_search(token)
        assert idx == expected[i]

    # Test case 3: large vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        vocab_size=5667
        )
    expected = EXPECTED_INDICES['large']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_brute_search(token)
        assert idx == expected[i]

def test_lookup_binary_search():
    
    # Test case 1: small vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json'),
        vocab_size=175
        )
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_binary_search(token)
        assert idx == expected[i]

    # Test case 2: medium vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_vocab.json'),
        vocab_size=1024
        )
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_binary_search(token)
        assert idx == expected[i]

    # Test case 3: large vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        vocab_size=5667
        )
    expected = EXPECTED_INDICES['large']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_binary_search(token)
        assert idx == expected[i]

def test_lookup_table_search():
    
    # Test case 1: small vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_vocab.json'),
        lookup_table_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'small_lookup_table.json'),
        vocab_size=175
        )
    expected = EXPECTED_INDICES['small']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_table_search(token)
        assert idx == expected[i]

    # Test case 2: medium vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_vocab.json'),
        lookup_table_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'medium_lookup_table.json'),
        vocab_size=1024
        )
    expected = EXPECTED_INDICES['medium']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_table_search(token)
        assert idx == expected[i]

    # Test case 3: large vocab
    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        lookup_table_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_lookup_table.json'),
        vocab_size=5667
        )
    expected = EXPECTED_INDICES['large']
    for i, token in enumerate(TEST_TOKENS):
        idx = bpe.lookup_table_search(token)
        assert idx == expected[i]
    

def test_encode():

    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        lookup_table_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_lookup_table.json'),
        vocab_size=5667
        )
    
    # Test case 1: simple sentence
    text = ['The mitochondria is the powerhouse of the cell!']
    expected = [636, 3, 3487, 3679, 2710, 1941, 728, 3, 3014, 3, 5053, 3, 4010, 2734, 3, 3689, 3, 5053, 3, 1447, 5, 1]
    assert bpe.encode(text) == expected

    # Test case 2: paragraph
    with open(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'encode_text.txt'), 'r') as file:
        text = [file.read()]
    
    with open(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'encode_values.json'), 'r') as file:
        expected = json.load(file)

    assert bpe.encode(text) == expected

def test_decode():

    bpe = NaiveBPETokenizer(
        vocab_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_vocab.json'),
        lookup_table_file=os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'large_lookup_table.json'),
        vocab_size=5667
        )

    # Test case 1: simple sentence
    text = [636, 3, 3487, 3679, 2710, 1941, 728, 3, 3014, 3, 5053, 3, 4010, 2734, 3, 3689, 3, 5053, 3, 1447, 5, 1]
    expected = 'The mitochondria is the powerhouse of the cell!'
    assert bpe.decode(text) == expected

    # Test case 2: paragraph
    with open(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'encode_values.json'), 'r') as file:
        text = json.load(file)
    
    with open(os.path.join(os.getenv('TESTS_DIR'), 'tokenize_tests', 'assets', 'encode_text.txt'), 'r') as file:
        expected = file.read()

    assert bpe.decode(text) == expected


if __name__ == "__main__":
    pytest.main()