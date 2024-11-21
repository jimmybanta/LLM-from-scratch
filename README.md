# LLM from scratch

In an effort to gain a deep understanding of LLM's (and their underlying architecture with transformers), I want to implement and train one from scratch.

## To Do

### Data Gathering

- [ ] Data - gather two datasets

  - [x] a small one, used for testing out/quick iteration
    - Using [LimaRP-augmented](https://huggingface.co/datasets/grimulkan/LimaRP-augmented)
  
  - [ ] a bigger one, used for training the LLM

### Transformer Implementation

- [ ] Data Pre-Processing
  
  
  
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
    
  - [ ] Word Embedding
    - [ ] first - train using gensim
      - [ ] 
    - [ ] later - train from scratch, using NN implementation developed for later on
  - [ ] Positional Encoding
  
- [ ] Attention

- [ ] Neural Network

- [ ] Normalization

- [ ] Residual Connections

- [ ] Output layer



## Directories & Files

- data/ -- contains modules related to data import, processing, etc.
- tests/ -- contains unit tests for the modules
- text-data-info/ -- contains information about the datasets used (the full datasets aren't being uploaded to GitHub)
  - LimaRP.md - the small dataset used for testing

- tokenize/ -- contains modules related to tokenization
- embed/ -- contains modules related to word embedding