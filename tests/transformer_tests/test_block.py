import pytest
import numpy as np
from transformer.block import TransformerBlock

@pytest.fixture
def transformer_block():
    return TransformerBlock(d_model=512, seq_len=20)

""" def test_shape(transformer_block):
    # Test that the shape of the input is the same as the shape of the output
    batch_size = 3
    seq_len = 20
    d_model = 512
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = transformer_block.forward(x)
    
    assert output.shape == x.shape """

def test_padding_tokens(transformer_block):
    # Test that padding tokens remain 0 after passing through the block
    batch_size = 3
    seq_len = 20
    d_model = 512
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Add padding tokens to the input
    x[0, :7, :] = 0  # Add padding tokens to the first sequence
    x[1, :18, :] = 0  # Add padding tokens to the second sequence
    x[2, :2, :] = 0  # Add padding tokens to the third sequence
    
    # Create the padding mask
    padding_mask = np.all(x == 0, axis=2)
    expanded_padding_mask = np.repeat(padding_mask[:, :, None], d_model, axis=2)
    
    output = transformer_block.forward(x, padding_mask=expanded_padding_mask)
    
    # Check that the shape of the output is the same as the input
    assert output.shape == x.shape
    
    # Check that padding tokens remain 0
    assert np.all(output[expanded_padding_mask] == 0.0)

if __name__ == '__main__':
    pytest.main()