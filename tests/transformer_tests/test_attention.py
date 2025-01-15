import pytest
import numpy as np
from transformer.attention import AttentionHead, MultiHeadAttention
from transformer.utils import softmax

@pytest.fixture
def attention_head():
    d_model = 512
    d_k = 64
    d_v = 64
    return AttentionHead(d_model, d_k, d_v)

@pytest.fixture
def multi_head_attention():
    d_model = 512
    num_heads = 8
    return MultiHeadAttention(d_model, num_heads=num_heads)

def test_head_initialization(attention_head):
    # Test if the AttentionHead is initialized correctly
    assert attention_head.d_k == 64
    assert attention_head.w_q.shape == (512, 64)
    assert attention_head.w_k.shape == (512, 64)
    assert attention_head.w_v.shape == (512, 64)

def test_multi_initialization(multi_head_attention):
    # Test if the MultiHeadAttention is initialized correctly
    assert multi_head_attention.d_k == 64
    assert len(multi_head_attention.heads) == 8

def test_head_forward(attention_head):
    # Test the forward method
    batch_size = 2
    seq_len = 10
    d_model = 512

    # Create a random input tensor of shape (batch_size, seq_len, d_model)
    x = np.random.randn(batch_size, seq_len, d_model)

    # Perform the forward pass
    output = attention_head.forward(x)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, 64)

def test_multi_forward(multi_head_attention):
    # Test the forward method
    batch_size = 2
    seq_len = 10
    d_model = 512

    # Create a random input tensor of shape (batch_size, seq_len, d_model)
    x = np.random.randn(batch_size, seq_len, d_model)

    # Perform the forward pass
    output = multi_head_attention.forward(x)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, d_model)

if __name__ == '__main__':
    pytest.main()