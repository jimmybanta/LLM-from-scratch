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

Added functions to write the vocab to a json file, and read it from a json file.

### Sunday - 11/17/24

Encoding - given tokens, how do I convert them to their integer values?
- most basic method - brute force search through the vocab
- more efficient method - binary search through the vocab, sorted alphabetically
- potentially even more efficient - have a hash table that stores the vocab by first character, then second character, etc - so it's a really quick lookup.

Try all 3, compare times.


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
