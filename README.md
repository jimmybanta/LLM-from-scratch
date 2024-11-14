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
  - [ ] Tokenizer - train a tokenizer from scratch
    - [ ] encoder
    - [ ] Decoder
  - [ ] Word Embedding - train a word embedder from scratch?
  - [ ] Positional Encoding
- [ ] Attention
- [ ] Neural Network
- [ ] Normalization
- [ ] Residual Connections
- [ ] Output layer



## Directories & Files

- text-data-info/ -- contains information about the datasets used (the full datasets aren't being uploaded to GitHub)
  - LimaRP.md - the small dataset used for testing
- data/ -- contains modules related to data import, processing, etc.