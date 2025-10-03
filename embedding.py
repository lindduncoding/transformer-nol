import numpy as np

# This file is for the embedding components

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # initialize small random embeddings
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(vocab_size)

    def __call__(self, token_ids):
        return self.W[token_ids]

class PositionalEncoding:
    def __init__(self, d_model, max_len=2048):
        # sinusoidal positional encodings (non-learned)
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term) # for even position
        pe[:, 1::2] = np.cos(position * div_term) # for odd position
        self.pe = pe 

    def __call__(self, seq_len):
        # should return (1, seq_len, d_model) for the next layers/components -> attention
        return self.pe[:seq_len][np.newaxis, :, :]