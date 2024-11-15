import pytest
from tokenize.bpe import BPETokenizer, NaiveBPETokenizer

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
        '<', '>', '?', '[', ']', '\\', ';', "'", ',', '.', '/', '`', '~', '<space>', 'and', '<space>', 'numbers', 
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



if __name__ == "__main__":
    pytest.main()