import pytest
import numpy as np
from transformer.attention import AttentionHead
from transformer.utils import softmax

@pytest.fixture
def attention_head():
    d_model = 512
    d_k = 64
    d_v = 64
    return AttentionHead(d_model, d_k, d_v)

def test_initialization(attention_head):
    # Test if the AttentionHead is initialized correctly
    assert attention_head.d_k == 64
    assert attention_head.w_q.shape == (512, 64)
    assert attention_head.w_k.shape == (512, 64)
    assert attention_head.w_v.shape == (512, 64)

def test_forward(attention_head):
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

if __name__ == '__main__':
    pytest.main()