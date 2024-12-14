# Dev Journal

### Saturday - 12/14/23
Currently on a flight to Singapore so not sure what day it actually is - but I'll call it Saturday.

#### MLP
Need to create an MLP
First, create a simple linear layer - that takes in an input, calculates the linear transformation,
given dimensions


### Fri - 12/13/24

#### Attention - Single Head
Now to implement the forward pass of an attention head.
Each will need to know the d_model (the size of the word embeddings), the d_q/d_k/d_v (the size of our query, key, value vectors)

Each attention head will have the following learned weights:
- query - of size [d_model, d_q]
- key - of size [d_model, d_k]
- value - of size [d_model, d_v]

As far as I'm aware, d_q, d_k, and d_v will all be the same size

When an input flows in, it will be of size [batch_size, seq_len, d_model]
We need to generate query, key, and value vectors from it - by matrix multiplying input @ w_q/w_k/w_v

Then, we multiply query by a transposed key matrix: 
- query shape = [seq_len, d_k]
- key transposed shape = [d_k, seq_len]
- multiplied gives us [seq_len, seq_len] 
- then, when it gets multiplied by value gives us [seq_len, d_k] - which is what we want

Then, we divide the scores element-wise by the square root of d_k
- this is what they did in the original paper - they say it's because, as d_k gets larger, the dot product produces bigger and bigger values - which pushes softmax into 'regions where it has extremely small gradients'
    - what does this mean, exactly? - I'm interpreting it to mean that, because softmax involves exponentials, the larger values will dominate more and more, the larger the values in general are
    - thus, we would lose some nuance, in a sense, as the softmax outputs would be largely dominated by a small number of values


Then, we take the softmax of these scores, along each input token
    - so each input token ends up with a vector, of length seq_len - thus giving an attention score for each other token in the sequence

Then, we multiply these attention scores by the value vectors - giving us output of size [seq_len, d_v] - which is what we want - these are the values

#### Softmax 
Implementing the softmax function - 
- Using shifting (shift all values by the max value) - to avoid having to use such large numbers
- for a while I wasn't able to get the softmax scores to add up to exactly 1 - I looked at the scipy code for their softmax function, and they used keepdims argument in their np functions - and that made the difference

#### Masked self-attention
I need to implement the mask, so that tokens never attend to tokens that come after them in the sequence.

That brings up another questin - how will this handle padding tokens? 
- Padding embeddings will be 0, so they'll remain 0 in the dot product calculation - but then they'll affect the softmax calculation - so we need to mask them out in the same way as the position mask

This brings up a problem - as we have left padding, we'll have early sequences that get everything masked out - so softmax won't work, as every value will be negative infinity.



#### Multi-headed attention
Now I need to figure out how to combine multiple attention heads into one attention layer.
- I'll concatenate the final outputs from each head, and then multiply this combined array by a learned projection matrix
- This is what they did in the paper


### Friday - 11/22/24

#### positional encoding - optimize
ran a quick test - it seems like, of all steps, positional encoding takes the longest by far -- at least 75% of the time goes towards it.
If it can optimize it, it will speed up pre-processing significantly.

Optimized:
- made it so it didn't iterate, only used numpy vectorization - but it didn't make much of a difference.
Oh well - I can come back and try to optimize more later.

#### Pre-processing
I'll make a PreProcess object - that handles all pre-processing.
So, when it comes time to train, I can simply create that object, use it to train on the dataset, and then use it to pre-process all text before passing it into the transformer.

#### Positional encoding
I'll use sinusoidal positional encoding.

This brings up a question - how to handle padding? 
My original plan was to simply add the PE to all the embeddings, padding tokens included, then just mask out the padding tokens -- but I'm reading things that decoder-only LLMS should use left-padding, as opposed to right padding, as they use the final token to predict the next token, in the output layer.
So, I need to do left padding - but then that brings up the question of how to do positional encoding? 
Plan:
- do left-padding, calculate the positional encoding, and shift them so they only apply to the non-padding tokens.

Another question - what's the most efficient way to do this? These positional encoding values will be constant, so I could just save them to a file and load them when doing this, rather than calculating them every time.
- Test this out, see what's faster
- When it comes to training, this could make a difference
- What kind of batch size will I be using for training?
    - ultimately, batch size won't be super relevant for this - as the positional encoding will be the same for all sequences in the batch, so I can just calculate/load it one and use it over and over
    - let's test out with embedding dimension of 512, and sequence length of 2048
    - calculating them takes around 0.01 seconds
    - loading them from file takes around 0.001 seconds - I'll use this
    - Now that I think about it, I'll probably only need to do this once - then I'll just have them stored in memory as the model is training. so this is somewhat of a moot point


### Thursday - 11/21/24

#### More word embeddings
So I think that encoding tokens as their integers will actually end up being unnecessary - I can encode them as their actual token values, and to convert them into their word embeddings, I'll have a dictionary that maps the token values to their embeddings - so there won't be a need for the token values.
But the token lookup table will still be useful, as it will save time when tokenizing text - so we don't have to traverse the whole vocab to find the token values.

I want to save the word embeddings in a file, like how I did the vocab and lookup tables. I'll write that method now.

One small, tiny problem I'm noticing - there's sometimes a mismatch between the size of the vocab learned through training the tokenizer, and that when training the word embeddings - this is because the tokenizer learns subword tokens, that then (may) get combined into larger tokens - so when text is tokenized, there's a chance those subwords don't show up as tokens - so no word embedding is generated for it.
I think with a large enough corpus, this isn't a problem - every token will show up. Or at least the vast, vast majority.

Word embedding saving:
- gensim offers a couple saving methods
    - save - saves the whole model
    - save_word2vec_format - saves just the word vectors
        - this is more space efficient  
    - I quickly tested out writing a save function of my own, which saves the word embeddings to a json file - it's more human-readable, but less space efficient.
    I think I'll just use the save_word2vec_format, both with binary (most space efficient) and non-binary (human readable) formats.

Padding:
- I need to incorporate padding tokens into my model, so each input is of the same length.
- Maybe I'll do this in the tokenizer encoder - so just add padding tokens to fill out each input of a batch.


#### general cleanup/reorganization
cleaned up and refactored some things 
- moved tokenizer and embedder into 'pre_process' module
- gave each a train module where I can easily train them on my datasets, for testing purposes
- updated both so that they can handle tokenizing/embedding batches, as opposed to a single chunk of text

### Wednesday - 11/20/24

#### Logging
configured logging for the project - have it log to a file, so I can have some more fine-grained control over catching errors, etc.

#### Word Embedding
Word embedding seems to work through a self-supervised training task with a neural network - we basically give it our corpus, 
it learns to predict the next word given the context, and the weights of the neural network are the word embeddings.

I think for now I'll try using a library to train the word embeddings - and later, after I've implemented the neural network part of the transformer, I'll use that NN code to train word-embeddings from scratch as well.

One big question - the tokenizer gives us a vocabulary of subword tokens. Do I get subword embeddings for each of these, or do I need to combine them somehow to get word embeddings?
Options:
    - just train subword embeddings 
        - problem: it seems like the embeddings wouldn't contain much useful information - ex. for the token 'a', what does that mean? It could be a part of so many different words, whereas the embedding for 'apple' would be much more informative.
        - but this likely won't be as much of a problem as I think - as the vocab size will be big enough that lots of words will be a single token

Side note - I wish people would stop using king vs queen as their examples for word embeddings. Be original.

I needed to update the tokenizer encode/decode method to allow for encoding and decoding from token values, rather than the integer values - so that I can encode text into its tokens, and pass those into the word2vec model. 
- Ultimately this isn't completely necessary - the word2vec will learn the same embeddings whether they're word tokens or their integer values, but for interpretability sake, like checking similar words, etc, it's nice to have the word tokens.

Word embedding with gensim is done! Now, given a corpus, I can tokenize it, then train word embeddings on it.


### Monday - 11/18/24

#### Search
Completed brute and binary search.

Changed special character mechanism so I have a file that lists the special characters, and it reads them in - so I can add any special characters I want.
Had some trouble with puncutation - there were characters that weren't included in string.punctuation, but seem like punctuation to me (e.g. {), so I added those in.

Completed hash table search.

Ran some quick tests on the speed of each method:
    - used the large vocab - as this is closest to the real-world scenario
    Results:

    <endoftext> - index: 0: binary speedup: 1.3335816725647234, hash speedup: 1.7399756986634265
    <newline> - index: 1: binary speedup: 1.5705741626794258, hash speedup: 1.6949225473321858
    <space> - index: 2: binary speedup: 0.6932013769363167, hash speedup: 1.2888
    <tab> - index: 3: binary speedup: 1.1434262948207172, hash speedup: 1.606942889137738
    <unknown> - index: 4: binary speedup: 0.908675799086758, hash speedup: 1.9944320712694878
    * - index: 14: binary speedup: 0.7243639167309175, hash speedup: 2.102965864577504
    0 - index: 36: binary speedup: 0.25004190646477065, hash speedup: 1.2506987143655675
    A - index: 49: binary speedup: 0.5688276659281379, hash speedup: 3.002033553634977
    items - index: 3047: binary speedup: 27.002383300460224, hash speedup: 87.4308142629058
    nature - index: 3578: binary speedup: 43.85366705471478, hash speedup: 131.4844677137871
    spite - index: 4756: binary speedup: 64.3044582751175, hash speedup: 176.70820244328098
    uses - index: 5356: binary speedup: 77.52594465141033, hash speedup: 180.9018938217945
    zing - index: 5661: binary speedup: 73.44297776192175, hash speedup: 287.6488122962273
    zipper - index: 5662: binary speedup: 76.53995063011563, hash speedup: 328.9380234505863
    zy - index: 5663: binary speedup: 85.15185856754306, hash speedup: 350.0217391304348
    zz - index: 5664: binary speedup: 86.27349570200573, hash speedup: 373.7982619490999
    zzle - index: 5665: binary speedup: 82.82770270270271, hash speedup: 355.9140145170296
    thisisatokenthatdoesntexist - index: -1: binary speedup: 81.94699407281965, hash speedup: 180.28949329359165
    
    For special characters, it was all about the same - as expected, because they're the first items in the vocab.
    
    For early items that aren't special tokens, binary is a little slower, which makes sense.
    As we get later, binary gets better.
    Hash table is consistently the fastest.

#### Encoder/Decoder

Created encode and decode functions.
Encode works by pre-tokenizing, so I have individual words. Then, I iterate backwards thorugh each word - so start with the full word, then the full word minus the last character, then minus the last two characters, etc. 
    - each time I check to see if a token exists that matches it - if it does, then I encode that token and move onto the rest of the word
    - if it doesn't, I iterate backwards until I find a match (or it's an unknown token)
    - there may be a better way to do this? - I'm not sure.
    - this seems like the most straightforward way to do it, maybe not the most efficient

    Ex. 
    - 'hello' - 'hello' -> 'hell' -> 'hel' = match
        - then, 'lo' -> match
        so it becomes 'hel' + 'lo'

Encoding a single sentence ('the mitochondria is the powerhouse of the cell') took 0.0001 seconds, 
decoding it took 0.000006 seconds.
Encoding a paragraph took 0.001 seconds, decoding took 0.00006 seconds
Encoding Moby Dick (~215k words) took 1.01 seconds, decoding took 0.06 seconds.

I'm fine with this for now - maybe I can optimize later.

#### GPT tokenizer

I may try to implement this later - if I were to try it, I would try byte-level BPE with it, then compare the two, see how they do.
For now, I'm fine with my tokenizer. Time to move on to word embedding.


### Sunday - 11/17/24

Encoding - given tokens, how do I convert them to their integer values?
- most basic method - brute force search through the vocab
- more efficient method - binary search through the vocab, sorted alphabetically
- potentially even more efficient - have a hash table that stores the vocab by first character, then second character, etc - so it's a really quick lookup.

Try all 3, compare times.



### Friday -- 11/15/24

Created text normalization for BPE tokenizer - converts text to ASCII, to remove weird characters - then (optionally) lowercases it.
- unsure as to whether I want to lowercase text - I'm leaning towards no, because I want the model to be able to use capitalization, for proper nouns, etc.

Pre-tokenizer:
- My first-pass intuition is to have special tokens for space, newline, and tab characters, as well as punctuation/special characters - and these don't get included in the merges for BPE. But other tokenizers seem to include spaces with the characters - curious as to why this is.
    - seems to me like having separate tokens would reduce vocabulary size/allow for merges to cover characters & words more effectively.
    - but the tradeoff will be that token sequences will be longer

- Naive version
    - Special tokens - space, newline, tab, punctuation/special characters, end of text, unknown
- GPT version
    - only special tokens for newline, tab, end of text, unknown
    - it's also a byte-level tokenizer, so maybe that is an important difference

- Maybe I'll try both versions, and see which works better when training the LLM?
    - I'm guessing the GPT version, because that's what OpenAI used

- Special tokens - what special tokens do I need? - by special I mean they won't be included with the merges
    - End of sentence - not sure I need, periods & other punctuation should be enough.
    - End of text - I think i need this, so that during inference the model knows when to stop  
        - This brings up the question of when to add end of text tokens during training? 
        - <endoftext>
    - Space, newline, tab - want to preserve formatting.
        - space = <space>
        - newline = <newline>
        - Tab = <tab>
    - Unknown - to catch all other characters that aren't in the vocab.
        - <unknown>
    - Punctuation/special characters
    

Wrote train function - it works decently well on the LimaRP dataset - I think it will need to be optimized for a larger dataset (or I'll just let it run for a while). 

Added functions to write the vocab to a json file, and read it from a json file.

### Thursday -- 11/14/24

Gathered small dataset - LimaRP-augmented, a dataset of role-playing conversations.

Small-ish ~ 20MB, will be quick for testing out things.


### Wednesday -- 11/13/24

Started project.



