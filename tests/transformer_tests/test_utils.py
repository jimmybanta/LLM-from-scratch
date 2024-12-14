import pytest
import numpy as np
from transformer.utils import softmax, relu

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

    

if __name__ == '__main__':
    pytest.main()