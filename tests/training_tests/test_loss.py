import pytest
import numpy as np
from training.loss import cross_entropy_loss

def test_cross_entropy_loss_no_padding():
    # Test cross entropy loss without padding
    preds = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])
    expected_loss = -np.sum(targets * np.log(preds))
    
    loss = cross_entropy_loss(preds, targets)
    
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}"

def test_cross_entropy_loss_with_padding():
    # Test cross entropy loss with padding
    preds = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])
    padding_mask = np.array([[True, False], [False, False]])
    loss_values = targets * np.log(preds)
    loss_values[padding_mask] = 0

    expected_loss = -np.sum(loss_values)
    
    loss = cross_entropy_loss(preds, targets, padding_mask)
    
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}"

def test_cross_entropy_loss_with_all_padding():
    # Test cross entropy loss with all padding
    preds = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0, 1], [1, 0]])
    padding_mask = np.array([[True, True], [True, True]])
    expected_loss = 0.0
    
    loss = cross_entropy_loss(preds, targets, padding_mask)
    
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}"

if __name__ == '__main__':
    pytest.main()