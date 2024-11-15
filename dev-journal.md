# Dev Journal



### Wednesday -- 11/13/24

Started project.


### Thursday -- 11/14/24

Gathered small dataset - LimaRP-augmented, a dataset of role-playing conversations.

Small-ish ~ 20MB, will be quick for testing out things.


### Friday -- 11/15/24

Created text normalization for BPE tokenizer - converts text to ASCII, to remove weird characters - then (optionally) lowercases it.
    - unsure as to whether I want to lowercase text - I'm leaning towards no, because I want the model to be able to use capitalization, for proper nouns, etc.

