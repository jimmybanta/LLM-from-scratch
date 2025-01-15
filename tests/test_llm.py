import pytest
import numpy as np
import os
from llm import LLM
from dotenv import load_dotenv
from pre_process.pre_process import PreProcessor

load_dotenv()

FILEPATHS = {

    'medium_vocab': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_vocab.json'),
    'medium_lookup_table': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'medium_lookup_table.json'),
    'medium_embeddings': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'embed', 'assets', 'medium_embeddings.bin'),

    'naive_special_char': os.path.join(os.getenv('HOME_DIR'), 'pre_process', 'tokenize', 'assets', 'naive_bpe_special_char.json'),

    'encode_text': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_text.txt'),
    'encode_token_values': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_token_values.json'),
    'encode_integer_values': os.path.join(os.getenv('TESTS_DIR'), 'pre_process_tests', 'tokenize_tests', 'assets', 'encode_integer_values.json')
}

@pytest.fixture
def pre_processor():
    files = {
        'vocab': FILEPATHS['medium_vocab'],
        'lookup_table': FILEPATHS['medium_lookup_table'],
        'word_embeddings': FILEPATHS['medium_embeddings'],
        'positional_encodings': None
    }
    return PreProcessor(train=False, files=files, context_size=2048, embedding_dim=512)

@pytest.fixture
def llm(pre_processor):
    return LLM(d_model=512, num_blocks=2, pre_processor=pre_processor)

def test_initialization(llm):
    # Test if the LLM is initialized correctly
    assert len(llm.blocks) == 2

def test_forward_shape(llm):
    # Test that the shape of the input is the same as the shape of the output
    batch_size = 3
    seq_len = 10
    d_model = 512
    vocab_size = 4096

    x = np.random.randn(batch_size, seq_len, d_model)

    output = llm.forward(x)
    
    assert output.shape == (batch_size, seq_len, vocab_size)

if __name__ == '__main__':
    pytest.main()