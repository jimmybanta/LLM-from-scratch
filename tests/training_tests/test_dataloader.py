import pytest
import numpy as np
import math
from training.dataloader import DataLoader

@pytest.fixture
def sample_data():
    embeddings = np.random.randn(30, 20, 512)  # Example embeddings of shape (30, 20, 512)
    integers = np.random.randint(0, 10000, (30, 20))  # Example integer representations of shape (100, 20)
    return embeddings, integers

@pytest.fixture
def dataloader(sample_data):
    embeddings, integers = sample_data
    return DataLoader(embeddings, integers, vocab_size=10000, batch_size=8, shuffle=True)

def test_initialization(dataloader):
    # Test if the DataLoader is initialized correctly
    assert dataloader.batch_size == 8
    assert dataloader.vocab_size == 10000
    assert dataloader.shuffle is True

def test_batching(dataloader):
    # Test the batching functionality
    d_model = 512
    embeddings = dataloader.embeddings
    num_batches = math.ceil(len(embeddings) / dataloader.batch_size)
    batches = dataloader.batches

    total = 0
    assert len(batches) == num_batches
    for i, batch in enumerate(batches):
        batch_embeddings, batch_integers = batch

        if i != len(batches) - 1:
            assert batch_embeddings.shape == (dataloader.batch_size, embeddings.shape[1], d_model)
            assert batch_integers.shape == (dataloader.batch_size, dataloader.targets.shape[1], dataloader.vocab_size)
        
        total += len(batch_embeddings)
    
    assert total == len(embeddings)
    
if __name__ == '__main__':
    pytest.main()