import numpy as np
from embedding import TokenEmbedding, PositionalEncoding
from attention import ScaledDotProductAttention, MultiHeadAttention
from ffn import FeedForward, LayerNorm

# Main Transformer setup

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class TransformerDecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        # decoder consist of MHA and FFN with normalization between
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

    def __call__(self, x, mask=None, causal=True):
        # Pre-norm style:
        x_norm = self.ln1(x)
        mha_out, attn = self.mha(x_norm, x_norm, mask=mask, causal=causal)
        x = x + mha_out  # residual

        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # residual
        return x, attn

# Main loop/decoder process

class TransformerDecoder:
    def __init__(self, num_layers, vocab_size, d_model, num_heads, d_ff, max_len=512):
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = [TransformerDecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_final = LayerNorm(d_model)
        # output (unembedding) projection matrix
        self.unembed = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    def __call__(self, token_ids, causal=True):
        # token_ids: (batch, seq_len) ints
        b, seq_len = token_ids.shape
        x = self.token_emb(token_ids)  # (b, seq_len, d_model)
        x = x + self.pos_enc(seq_len)  # broadcast add

        attn_maps = []
        
        for layer in self.layers:
            x, attn = layer(x, causal=causal)
            attn_maps.append(attn)

        x = self.ln_final(x)
        logits = x @ self.unembed
        probs = softmax(logits, axis=-1)  # final probability distribution per token
        return logits, probs, attn_maps

if __name__ == "__main__":
    np.random.seed(42)
    B = 4
    S = 16
    VOCAB = 1000
    D = 64
    HEADS = 16
    D_FF = 256
    LAYERS = 2

    # dummy tokens -> random integers
    tokens = np.random.randint(0, VOCAB, size=(B, S))
    model = TransformerDecoder(num_layers=LAYERS, vocab_size=VOCAB, d_model=D, num_heads=HEADS, d_ff=D_FF, max_len=128)
    logits, probs, attn_maps = model(tokens, causal=True)

    # results
    print("logits at last layer", logits[:, -1, :])
    print("probs at last layer", probs[:, -1, :])
    print("attention mask", attn_maps[-1])