import pytest
from tokenize.bpe import BPETokenizer, NaiveBPETokenizer

def test_normalize():
    bpe = BPETokenizer()

    # Test case 1: Normal text without lowercase
    text = "Café"
    expected = "Cafe"
    assert bpe.normalize(text) == expected

    # Test case 2: Normal text with lowercase
    text = "Café"
    expected = "cafe"
    assert bpe.normalize(text, lowercase=True) == expected

    # Test case 3: Text with special characters
    text = "naïve"
    expected = "naive"
    assert bpe.normalize(text) == expected

    # Test case 4: Text with special characters and lowercase
    text = "naïve"
    expected = "naive"
    assert bpe.normalize(text, lowercase=True) == expected

    # Test case 5: Text with uppercase letters
    text = "HELLO"
    expected = "HELLO"
    assert bpe.normalize(text) == expected

    # Test case 6: Text with uppercase letters and lowercase
    text = "HELLO"
    expected = "hello"
    assert bpe.normalize(text, lowercase=True) == expected

def test_naive_pre_tokenize():
    bpe = NaiveBPETokenizer()

    # Test case 1: Simple sentence
    text = "Hello, world!"
    expected = ["Hello", ",", "<space>", "world", "!", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 2: Sentence with punctuation
    text = "How's it going?"
    expected = ["How", "'", "s", "<space>", "it", "<space>", "going", "?", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 3: Sentence with multiple punctuation marks
    text = "Wait... What?!"
    expected = ["Wait", ".", ".", ".", "<space>", "What", "?", "!", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 4: Sentence with numbers
    text = "The price is $5.99."
    expected = ["The", "<space>", "price", "<space>", "is", "<space>", "$", "5", ".", "99", ".", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 5: Sentence with mixed characters
    text = "Café-naïve"
    expected = ["Café", "-", "naïve", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 6: Sentence with special characters and whitespace
    text = "Hello\tworld\nHow are you?"
    expected = ["Hello", "<tab>", "world", "<newline>", "How", "<space>", "are", "<space>", "you", "?", "<endoftext>"]
    assert bpe.pre_tokenize(text) == expected

    # Test case 7: complex sentence
    text = '''     
         This is a sample text.. \n\t\n 
            I want to includ0e special characters like !@#$%^&*()_+{}|:"<>?[]\;',./`~ and numbers like 1234567890. 
to make sure they're all pre-tokenized correctly.

        bla bla bla 10$10
    '''
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

if __name__ == "__main__":
    pytest.main()