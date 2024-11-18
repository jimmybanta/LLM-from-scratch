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
  - [ ] Tokenizer - train a tokenizer from scratch -- using BPE
    - [x] normalization
    - [ ] pre-tokenization - two methods to try
      - [x] 'Naive method' - have space, newline, tab, punctuation as special characters (along with <endoftext>)
      - [ ] GPT method - include spaces 
    - [x] Training
      - [ ] optimize?
    - [x] vocab -- save to file, read from file
    - [x] lookup - 3 methods for retrieving token indices
      - [x] brute force search
      - [x] binary search
      - [x] hash table
    - [ ] encoder
    - [ ] decoder
    - [ ] later - train a byte-level BPE tokenizer?
  - [ ] Word Embedding - train a word embedder from scratch?
  - [ ] Positional Encoding
- [ ] Attention
- [ ] Neural Network
- [ ] Normalization
- [ ] Residual Connections
- [ ] Output layer



## Directories & Files

- data/ -- contains modules related to data import, processing, etc.
- tests/ -- contains unit tests for the modules
- tokenize/ -- contains modules related to tokenization
- text-data-info/ -- contains information about the datasets used (the full datasets aren't being uploaded to GitHub)
  - LimaRP.md - the small dataset used for testing