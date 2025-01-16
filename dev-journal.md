# Dev Journal

### Thursday - 1/16/25
Training! 

Time to work on training the model.
General overview of training:
- I'll need a dataset, that I'll split up into sentences, of a maximum length.
- I'll need to pre-process this dataset, so that it's ready to be passed into the model.
- Then, I'll put it into batches, and pass them through the forward pass of the model.
- Then, I'll calculate the loss.
    - One question - I'll want every sentence to be multiple training examples - ex. for the sentence 'The frog is green', I'll want to predict 'frog' given 'The', 'is' given 'The frog', 'green' given 'The frog is', etc. Naively, I could do many forward passes - but this seems inefficient. Plus, in one forward pass, I'm effectively doing that anyway, right? 
        - Yep - tested this out by running 'the', then 'the frog', then 'the frog is' through - and the output tokens at the beginning stay the same. This makes sense - tokens only attend to ones before them, so tokens after a token have no effect on the output of that token.
    - So, I can just run each sentence through, and calculate the loss for each token, for predicting the next token. 

#### Dataloader
I need a dataloader, that can set up my data for training - so I give it the embedded sentences, and the corresponding integer values of the tokens, and it will return batches of pairs of sentences, with the corresponding tokens to predict (I'll do this by just shifting the token lists back one - so that, in each position i, you have the embedding for token i, and the integer value for token i+1, i.e. the next token, that it's trying to predict.)

One thing I'm coming across - I'll need to make sure my vocab is uniform between the tokenizer and the embedder - I have some tokens that through BPE go into the vocab, but don't make it into the word embeddings. This is because they're subparts of words, and they only show up in that one word.
Ex. 'Abig' is in my tokenizer vocab, but so is 'Abigail' - and in the test corpus I'm using, 'Abig' only is ever part of 'Abigail' - so Abigail gets picked up by the word embedder, but 'Abig' doesn't. So, I'll need to remove tokens like that, to have uniform vocab. 


#### Loss function
I'll start off with cross entropy loss. 
One thing - I'll need to be sure to not include padding tokens into the loss calculation.
The way I'll do this is by calculating the inner loss values (before summing them), and then 
use a mask to set all the values corresponding to padding tokens to 0.
I need to be a little careful - the targets are the one-hot vectors, formed from the integer values of the tokens. And we dropped the first token - so the targets have one fewer token than the embeddings.
So, I'll drop the last token from the output of the LLM - because the last token isn't predicting anything.
That way, the shapes will match. 
It's input minus the last token, and targets minus the first token. 
So the padding mask will come from the input, as I want to ignore all predictions from the padding tokens - even the final padding token, which will be compared to the first 'real' token - I want to ignore this as well, when it comes to the loss. 

#### Label smoothing
In the original Transformer paper, they use label smoothing - so I'll add it.
I'll implement it within the dataloader.


### Wednesday - 1/15/25

#### Full Model
Now, time to put it all together into a full LLM.
I'll need to add a linear unembedding layer, to convert the output of the transformer blocks back into the word embeddings.
Then, I'll need to add a sampling function, to generate text from the model.

I want to form one general LLM object, where you can specify how many transformer blocks, how many attention heads per block, d_model, sequence length, etc.

##### Generate Text
I'll have this function, generate_text, which you can pass a message, and the LLM will generate a response to it. One thing I'm noticing is that the forward pass through the LLM is dependent on the size of the input - including padding tokens. So, I can optimize this by ensuring that there are only necessary padding tokens - in other words, for a batch of just one input, there should be no padding tokens. For a batch of multiple, the input length will be the max length.

### Tuesday - 1/14/25

#### Transformer Block
Putting it all together! So the inputs will flow in, and go through multi-headed attention. Then, we'll add that output to the residual stream, and normalize. Then, run that through the feedforward, then add that to the residual stream, and normalize.
Then that's our output.

One wrinkle - padding tokens don't get preserved - as it is now, the padding token values of all 0's don't get preserved as they flow through. This is because of the bias values in the feedforward layer - they get added, making the padding tokens non-zero.
I need to fix this. 

The most efficient way would probably be to stop the padding tokens from going through the feedforward layer in the first place - no need for all that computation, we know what we want as the output.

So maybe I'll try chopping off the padding tokens, and only running the 'real' tokens through the feedforward - then reattaching after.
- Ok - I don't know if this will work, because in a batch, the padding tokens will be different for each example - so I think I need to leave them in, to run them through the feedforward layer all at once - otherwise I'd have to split them up and run seperately, which would be more inefficient than running them through, with the padding tokens (I think).
- So, I'll run everything through, and zero out the padding tokens after they've gone through.
- This way, the padding tokens will remain 0 after going through a transformer block.

### Monday - 1/13/25
Getting back to work on this.

Today - I need to figure out the padding mask, and implement LayerNorm.
Then, complete my implementation of the Transformer Block.

#### LayerNorm
I need to figure out over what dimensions to calculate the mean and variance - I obviously do it for each sentence in the batch, so that there's no spillover between examples in the batch. But do I do it by token, or do I do it over all the tokens?
    - I feel like I shouldn't do it over all the tokens, as there are padding tokens, and these will give mean/variance of 0, which will throw everything off

Also - in Pytorch's docs, they have these gamma and beta parameters - they seem to allow the model to learn certain weights for scaling & shifting the normalized values. I'll include these too.

#### Padding Mask
I need to have a padding mask, so that tokens don't attend to padding tokens.
So, I'll get the positions of the padding tokens, and then set the attention scores for those tokens to negative infinity - so that when they're softmaxed, they become 0.

This presents one wrinkle - when we combine the padding mask with the position mask (that masks out future values) for the padding tokens themselves, they won't attend to anything - anything after will be masked out, anything before will be more padding tokens, so also masked out.
So, softmax will return nan because we're passing in all -inf values.
I think I'll handle this by taking the nan values from softmax and replace them with 0. 


### Saturday - 12/14/24
Currently on a flight to Singapore so not sure what day it actually is - but I'll call it Saturday.

#### MLP
Need to create an MLP 
First, create a simple linear layer - that takes in an input, calculates the linear transformation,
given dimensions

Then, I need to implement ReLU

Then, I'll combine them into a 2-layer MLP

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



