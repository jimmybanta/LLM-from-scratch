import pytest
import numpy as np
from transformer.layer_norm import LayerNorm

@pytest.fixture
def layer_norm():
    return LayerNorm(d_model=512)

def test_initialization(layer_norm):
    # Test if the LayerNorm is initialized correctly
    assert layer_norm.eps == 1e-5
    assert layer_norm.gamma.shape == (512,)
    assert layer_norm.beta.shape == (512,)

def test_forward(layer_norm):
    # Test the forward method
    x = np.random.randn(2, 20, 512)  # Create a random input tensor of shape (batch_size, seq_len, d_model)
    output = layer_norm.forward(x)

    # Check the output shape
    assert output.shape == (2, 20, 512)

    # Check if the output mean is close to 0 and variance is close to 1
    output_mean = np.mean(output, axis=-1)
    output_variance = np.var(output, axis=-1)
    np.testing.assert_almost_equal(output_mean, np.zeros((2, 20)), decimal=5)
    np.testing.assert_almost_equal(output_variance, np.ones((2, 20)), decimal=5)

if __name__ == '__main__':
    pytest.main()