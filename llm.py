''' Contains the class for an LLM. '''

import numpy as np

from transformer.block import TransformerBlock
from transformer.mlp import Linear
from transformer.utils import softmax, dropout


class LLM:
    '''
    A Large Language Model.
    '''

    def __init__(self, d_model, pre_processor,
                    pe_dropout=0.1,
                    sublayer_dropout=0.1,
                    num_blocks=6,
                    parameter_dict=None,
                    # attention
                    num_heads=8,
                    # feedforward
                    hidden_dim=2048,
                    bias=True,
                    # layer norm
                    eps=1e-5,
                    scale_shift=True
                ):
        '''
        Initializes the language model.

        Parameters
        ----------
        d_model: int
            The size of the word embeddings.
        pre_processor: PreProcessor
            The preprocessor object.
        pe_dropout: float, optional
            The dropout rate to apply to the embeddings + positional encodings.
        sublayer_dropout: float, optional
            The dropout rate to apply to the outputs of the sublayers.
        num_blocks: int, optional
            The number of transformer blocks to use.
        parameter_dict: dict, optional
            Dictionary of parameters to use.
        num_heads: int, optional
            The number of attention heads to use in each transformer block.
        hidden_dim: int, optional
            The size of the hidden layer in the feedforward layers.
        bias: bool, optional
            Whether to include bias in the feedforward layers.
        eps: float, optional
            The epsilon value to use in layer normalization.
        scale_shift: bool, optional
            Whether to scale and shift the layer normalization.
        '''

        self.pre_processor = pre_processor

        # if parameters have been passed in
        if parameter_dict:
            pass

        else:
            # instantiate the transformer blocks
            self.blocks = [TransformerBlock(d_model, 
                                            dropout=sublayer_dropout,
                                            num_heads=num_heads,
                                            hidden_dim=hidden_dim,
                                            bias=bias,
                                            eps=eps,
                                            scale_shift=scale_shift) for _ in range(num_blocks)]
            
            # instantiate the unembedding layer
            self.unembed = Linear(in_features=d_model,
                                    out_features=pre_processor.vocab_size,
                                    bias=True)

        self.pe_dropout = pe_dropout
        self.sublayer_dropout = sublayer_dropout


    def forward(self, x, 
                temperature=1.0,
                attention_mask=None, padding_mask=None):
        '''
        Forward pass through the language model.

        Parameters
        ----------
        x: array
            Input array - of shape (batch_size, seq_len, d_model)
        temperature: float, optional
            The temperature to use in the softmax function.
        attention_mask: array, optional
            The attention mask to use.
        padding_mask: array, optional
            The padding mask to use.

        Returns
        -------
        array
            Output array - of shape (batch_size, seq_len, vocab_size)
        '''

        seq_len = x.shape[1]

        # form the masks
        if not attention_mask:
            # create attention mask - masks out future tokens, and padding tokens
            mesh = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
            position_mask = (mesh[1] < mesh[0])

            padding_tokens_mask = np.all(x == 0, axis=2)
            padding_attention_mask = np.repeat(padding_tokens_mask[:, None, :], padding_tokens_mask.shape[1], axis=1)

            attention_mask = np.logical_or(position_mask, padding_attention_mask)

        if not padding_mask:

            padding_tokens_mask = np.all(x == 0, axis=2)

            # create padding mask - masks out padding tokens, to set them back to 0 after going through the block
            padding_mask = np.repeat(padding_tokens_mask[:, :, None], x.shape[2], axis=2)

        # first - apply dropout to the inputs (sums of embeddings and positional encodings)
        x = dropout(x, self.pe_dropout)

        # pass through the blocks
        for block in self.blocks:
            x = block.forward(x, attention_mask=attention_mask, 
                              padding_mask=padding_mask)
        
        # pass through the unembedding layer
        x = self.unembed.forward(x)

        # apply softmax
        return softmax(x, temperature=temperature, axis=-1)
    
    def sample(self, probs, method='weighted'):
        '''
        Given a probability distribution, sample from it.

        Parameters
        ----------
        probs: array
            The probability distribution.
        method: str, optional
            The method to use for sampling. Either 'greedy' or 'weighted'.

        Returns
        -------
        int
            The index of the sampled token.
        '''

        if method == 'greedy':
            # get the most likely token
            return np.argmax(probs)
        elif method == 'weighted':
            # sample from the distribution, weighted on the probabilities
            return np.random.choice(len(probs), p=probs)
    
    def generate_text(self, message, max_len=100, temperature=1.0):
        '''
        Generate text from the language model.

        Parameters
        ----------
        message: str
            The message to start the generation with.
        max_len: int, optional
            The maximum length of the generated text.

        Returns
        -------
        str
            The generated text.
        '''

        # preprocess the message
        pre_processed_message = self.pre_processor.pre_process([message])

        current_message = pre_processed_message

        # pass through the model
        for _ in range(max_len):

            output = self.forward(current_message, temperature=temperature)

            # get the last token
            last_token_probs = output[0][-1]

            # get the most likely token
            most_likely = self.sample(last_token_probs, method='weighted')

            # add the decoded token to the raw message
            decoded_token = self.pre_processor.tokenizer.decode([[most_likely]])[0]
            message += decoded_token

            # if it's the end of text token, then break
            if most_likely == self.pre_processor.tokenizer.end_of_text_token_int:
                break

            # embed the token
            embedded = self.pre_processor.word_embedder.embed_batch([[most_likely]])
            
            # add the embedded token to the embeddings message
            current_message = np.concatenate([current_message, embedded], axis=1)

            
        return message

