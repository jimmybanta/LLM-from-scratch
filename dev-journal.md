# Dev Journal



### Wednesday -- 11/13/24

Started project.


### Thursday -- 11/14/24

Gathered small dataset - LimaRP-augmented, a dataset of role-playing conversations.

Small-ish ~ 20MB, will be quick for testing out things.


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