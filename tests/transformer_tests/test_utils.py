import pytest
import numpy as np
from transformer.utils import softmax, relu, dropout

def test_softmax():
    # Test case 1: Simple 1D array
    x = np.array([1.0, 2.0, 3.0])
    expected = np.exp(x) / np.sum(np.exp(x))
    result = softmax(x, axis=0)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    assert np.sum(result, axis=0).all() == 1

    # Test case 2: 2D array along axis 0
    x = np.array([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]])
    expected = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    result = softmax(x, axis=0)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    assert np.sum(result, axis=0).all() == 1

    # Test case 3: 2D array along axis 1
    x = np.array([[1.0, 2.0, 3.0],
                  [1.0, 2.0, 3.0]])
    expected = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    result = softmax(x, axis=1)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    assert np.sum(result, axis=1).all() == 1

    # Test case 4: 3D array along axis 2
    x = np.array([[[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]],
                  [[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]]])
    expected = np.exp(x) / np.sum(np.exp(x), axis=2, keepdims=True)
    result = softmax(x, axis=2)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    assert np.sum(result, axis=2).all() == 1

def test_relu():
    x = np.random.randn(32, 20, 48)

    x = relu(x)

    assert np.all(x >= 0)

def test_dropout_no_dropout():
    # Test dropout with p=0 (no dropout)
    input = np.random.randn(10, 10)
    output = dropout(input, p=0)
    np.testing.assert_array_equal(output, input)

def test_dropout_full_dropout():
    # Test dropout with p=1 (invalid, should raise ValueError)
    input = np.random.randn(10, 10)
    with pytest.raises(ValueError):
        dropout(input, p=1)

def test_dropout_scaling():
    # Test dropout with scaling
    input = np.random.randn(10, 10)
    p = 0.5
    output = dropout(input, p=p, scale=True)
    mask = output != 0
    expected_output = input * mask / (1 - p)
    np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

def test_dropout_no_scaling():
    # Test dropout without scaling
    input = np.random.randn(10, 10)
    p = 0.5
    output = dropout(input, p=p, scale=False)
    mask = output != 0
    expected_output = input * mask
    np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

def test_dropout_probability():
    # Test dropout probability
    input = np.random.randn(1000, 1000)
    p = 0.5
    output = dropout(input, p=p, scale=False)
    dropout_ratio = np.mean(output == 0)
    assert np.isclose(dropout_ratio, p, atol=0.05), f"Expected dropout ratio close to {p}, but got {dropout_ratio}"

    

if __name__ == '__main__':
    pytest.main()