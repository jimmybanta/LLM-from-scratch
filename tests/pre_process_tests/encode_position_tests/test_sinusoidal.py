import pytest
import os
import numpy as np
from pre_process.encode_position.sinusoidal import SinusoidalPE

@pytest.fixture
def sinusoidal_pe():
    return SinusoidalPE(context_size=256, d_embedding=512)

def test_calculate_pe(sinusoidal_pe):
    # Test the calculate_pe method
    pe = sinusoidal_pe.calculate_pe()
    assert isinstance(pe, np.ndarray)
    assert pe.shape == (256, 512)

def test_save_and_load_pe(sinusoidal_pe):
    # Test saving and loading positional encodings
    pe = sinusoidal_pe.calculate_pe()
    path = "test_pe.npy"
    sinusoidal_pe.save_pe(pe, path)
    loaded_pe = sinusoidal_pe.load_pe(path)
    assert np.array_equal(pe, loaded_pe)
    os.remove(path)

def test_encode(sinusoidal_pe):
    # Test the encode method
    batch_embeddings = np.random.rand(2, 256, 512)
    encoded_embeddings = sinusoidal_pe.encode(batch_embeddings)
    assert isinstance(encoded_embeddings, np.ndarray)
    assert encoded_embeddings.shape == (2, 256, 512)
    assert not np.array_equal(batch_embeddings, encoded_embeddings)

def test_invalid_pe_file():
    # Test loading positional encodings with an invalid file path
    with pytest.raises(TypeError):
        SinusoidalPE(pe_file=123)

def test_invalid_save_pe(sinusoidal_pe):
    # Test saving positional encodings with invalid parameters
    with pytest.raises(TypeError):
        sinusoidal_pe.save_pe("not_an_array", "test_pe.npy")
    with pytest.raises(TypeError):
        sinusoidal_pe.save_pe(np.zeros((256, 512)), 123)

def test_invalid_load_pe(sinusoidal_pe):
    # Test loading positional encodings with invalid parameters
    with pytest.raises(TypeError):
        sinusoidal_pe.load_pe(123)

def test_invalid_encode(sinusoidal_pe):
    # Test encoding with invalid parameters
    with pytest.raises(TypeError):
        sinusoidal_pe.encode("not_an_array")

if __name__ == '__main__':
    pytest.main()