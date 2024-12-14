import pytest
import numpy as np
from transformer.mlp import Linear

def test_forward_no_bias():
    in_features = 3
    out_features = 2
    x = np.array([[1, 2, 3]])
    w = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    linear = Linear(in_features, out_features, bias=False, w=w)
    output = linear.forward(x)
    expected_output = np.array([[2.2, 2.8]])
    np.testing.assert_array_almost_equal(output, expected_output)

def test_forward_with_bias():
    in_features = 3
    out_features = 2
    x = np.array([[1, 2, 3]])
    w = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = np.array([0.1, 0.2])
    linear = Linear(in_features, out_features, bias=True, w=w)
    linear.b = b
    output = linear.forward(x)
    expected_output = np.array([[2.3, 3.0]])
    np.testing.assert_array_almost_equal(output, expected_output)

def test_forward_random_weights():
    in_features = 3
    out_features = 2
    x = np.array([[1, 2, 3]])
    linear = Linear(in_features, out_features, bias=False)
    output = linear.forward(x)
    assert output.shape == (1, out_features)

def test_forward_random_weights_with_bias():
    in_features = 3
    out_features = 2
    x = np.array([[1, 2, 3]])
    linear = Linear(in_features, out_features, bias=True)
    output = linear.forward(x)
    assert output.shape == (1, out_features)


if __name__ == '__main__':
    pytest.main()