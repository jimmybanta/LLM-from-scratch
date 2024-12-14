import pytest
import numpy as np
from transformer.mlp import Linear, TwoLayerMLP

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

def test_two_layer_mlp_forward_no_bias():
    input_dim = 3
    hidden_dim = 4
    x = np.array([[1, 2, 3]])
    w1 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    w2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, bias=False, layers_w=[w1, w2])
    output = mlp.forward(x)
    expected_output = np.array([[11.24, 13.12, 15]])
    np.testing.assert_array_almost_equal(output, expected_output)

def test_two_layer_mlp_forward_with_bias():
    input_dim = 3
    hidden_dim = 4
    x = np.array([[1, 2, 3]])
    w1 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    b1 = np.array([0.1, 0.2, 0.3, 0.4])
    w2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    b2 = np.array([0.1, 0.2, 0.3])
    mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, bias=True, layers_w=[w1, w2], layers_b=[b1, b2])
    output = mlp.forward(x)
    expected_output = np.array([[12.04, 14.12, 16.2]])
    np.testing.assert_array_almost_equal(output, expected_output)

def test_two_layer_mlp_forward_random_weights():
    input_dim = 3
    hidden_dim = 4
    x = np.array([[1, 2, 3]])
    mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, bias=False)
    output = mlp.forward(x)
    assert output.shape == (1, input_dim)

def test_two_layer_mlp_forward_random_weights_with_bias():
    input_dim = 3
    hidden_dim = 4
    x = np.array([[1, 2, 3]])
    mlp = TwoLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, bias=True)
    output = mlp.forward(x)
    assert output.shape == (1, input_dim)

if __name__ == '__main__':
    pytest.main()