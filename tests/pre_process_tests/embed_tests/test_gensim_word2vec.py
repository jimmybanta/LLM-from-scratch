import os
import numpy as np
import pytest
from pre_process.embed.gensim_word2vec import GensimWord2Vec

@pytest.fixture
def sample_tokenized_sentences():
    return [
        ["this", "is", "a", "test"] + ["<pad>"] * 96,
        ["another", "test", "sentence"] + ["<pad>"] * 97
    ]

@pytest.fixture
def gensim_w2v(sample_tokenized_sentences):
    return GensimWord2Vec(tokenized_sentences=sample_tokenized_sentences, 
                          size=50, window=3, min_count=1, workers=1,
                          context_size=100)

def test_train(gensim_w2v):
    # Test training the model
    gensim_w2v.train()
    assert gensim_w2v.model is not None
    assert gensim_w2v.model.vector_size == 50

def test_save_and_load_binary(gensim_w2v):
    # Test saving and loading the model in binary format
    gensim_w2v.train()
    model_file = "test_model.bin"
    gensim_w2v.save(model_file, file_type='binary')
    assert os.path.exists(model_file)

    # Load the model and check if it is loaded correctly
    loaded_model = GensimWord2Vec(model_file=model_file, size=50)
    assert loaded_model.model is not None
    assert loaded_model.model.vector_size == 50

    # Clean up
    os.remove(model_file)

def test_save_and_load_table(gensim_w2v):
    # Test saving and loading the table
    gensim_w2v.train()
    model_file = "test_model.txt"
    gensim_w2v.save(model_file, file_type='table')
    assert os.path.exists(model_file)

    # Load the model and check if it is loaded correctly
    loaded_model = GensimWord2Vec(model_file=model_file, size=50)
    assert loaded_model.model is not None
    assert loaded_model.model.vector_size == 50

    # Clean up
    os.remove(model_file)

def test_embed_batch(gensim_w2v, sample_tokenized_sentences):
    # Test embedding a batch of sentences
    gensim_w2v.train()
    batch_embeddings = gensim_w2v.embed_batch(sample_tokenized_sentences)
    assert len(batch_embeddings) == len(sample_tokenized_sentences)
    assert len(batch_embeddings[0]) == len(sample_tokenized_sentences[0])
    assert len(batch_embeddings[1]) == len(sample_tokenized_sentences[1])
    assert batch_embeddings[0][0].shape[0] == 50

if __name__ == '__main__':
    pytest.main()