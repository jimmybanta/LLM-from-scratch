# LLM from scratch

In an effort to gain a deep understanding of LLM's (and their underlying architecture with transformers), I want to implement and train one from scratch, without the use of any machine learning library (like PyTorch, Tensorflow, etc.)

## To Do

### Data Gathering

- [ ] Data - gather two datasets

  - [x] a small one, used for testing out/quick iteration
    - Using [LimaRP-augmented](https://huggingface.co/datasets/grimulkan/LimaRP-augmented)
  
  - [ ] a bigger one, used for training the LLM

### Transformer Implementation

- [x] Data Pre-Processing
  
  
  
  - [x] Tokenizer - train a tokenizer from scratch -- using BPE
    - [x] normalization
    - [x] pre-tokenization - 'Naive method' - have space, newline, tab, punctuation as special characters (along with <endoftext>, <unknown>)
    - [x] Training
    - [x] vocab -- save to file, read from file
    - [x] lookup - 3 methods for retrieving token indices
      - [x] brute force search
      - [x] binary search
      - [x] hash table
    - [x] encoder
    - [x] decoder
    - [ ] later
      - [ ]  train a byte-level BPE tokenizer?
      - [ ] optimize train loop?
      - [ ] optimize encoder?
    
  - [x] Word Embedding - using word2vec
    - [x] first - use gensim to train custom word embeddings with word2vec
    - [ ] later - train from scratch, using NN implementation that I'll develop later on
  - [x] Positional Encoding - using sinusoidal PE
    - [ ] optimize?
  - [x] PreProcess - object that encompasses all pre-processing
  
- [ ] Attention
  - [x] Single Attention Head - forward pass
  - [ ] Masked self-attention
    - [x] look-ahead mask
    - [ ] padding mask
  - [x] Multi-head attention

- [ ] Neural Network
  - [x] Linear layer
  - [ ] Combine linear + activation functions for an MLP

- [ ] Utility functions
  - [x] Softmax
  - [x] ReLU
  - [ ] LayerNorm

- [ ] Residual Connections

- [ ] Output layer



## Directories & Files

- data/ -- contains modules related to data import, processing, etc.
- tests/ -- contains unit tests for the modules
- text-data-info/ -- contains information about the datasets used (the full datasets aren't being uploaded to GitHub)
  - LimaRP.md - the small dataset used for testing

- pre_process/ -- contains modules for pre-processing (tokenize, embed, encode_position)

- dev_journal.md -- a development journal I'm keeping of the project

- logging.conf -- logging configuration