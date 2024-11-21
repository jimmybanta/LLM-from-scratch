

# a tiny batch of text, to be used for very quick testing
TINY_BATCH = [
    "Hello, World!\nThis is a test.",
    "Tab\tseparated\tvalues",
    "Special characters: !@#$%^&*()_+",
    "Whitespace    and    spaces",
    "Newline\nin\nthe\nmiddle"
]


def import_tiny():
    '''
    Import a tiny batch of text for testing purposes.

    Returns
    -------
    List[str]
        A list of strings.
    '''
    return TINY_BATCH