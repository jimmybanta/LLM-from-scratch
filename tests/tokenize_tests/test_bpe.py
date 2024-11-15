import pytest
from tokenize.bpe import BPETokenizer

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

if __name__ == "__main__":
    pytest.main()