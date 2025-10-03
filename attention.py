import numpy as np

# This file is for attention, the SDPA and Multi-Head Attention

def causal_mask(seq_len, dtype=np.float32):
    # causal masking to block "future token"
    # 0 is blocked/masked 1 is allowed
    return np.tril(np.ones((seq_len, seq_len), dtype=dtype))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class ScaledDotProductAttention:
    def __init__(self):
        pass

    def __call__(self, Q, K, V, causal=True):
        # Q,K,V: (batch, heads, seq_q, head_dim) / (batch, heads, seq_k, head_dim)
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d_k)  # (batch, heads, seq_q, seq_k)

        if causal:
            seq_q = scores.shape[-2]
            seq_k = scores.shape[-1]
            cm = causal_mask(seq_k)
            cm = cm[np.newaxis, np.newaxis, :seq_q, :seq_k]
            # where cm == 0, set to large negative -> essentially blocking attention forward
            scores = np.where(cm == 1, scores, -1e9)

        attn = softmax(scores, axis=-1)  # (batch, heads, seq_q, seq_k)
        out = np.matmul(attn, V)         # (batch, heads, seq_q, head_dim)
        return out, attn

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # weight matrices for linear projections: using small random initialization
        k = 1 / np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) * k
        self.W_k = np.random.randn(d_model, d_model) * k
        self.W_v = np.random.randn(d_model, d_model) * k
        self.W_o = np.random.randn(d_model, d_model) * k

        self.attn = ScaledDotProductAttention()

    def split_heads(self, x):
        # split input into heads
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return x.transpose(0,2,1,3)

    def combine_heads(self, x):
        # combine attention layer's heads into one projection
        # x: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        b, h, s, hd = x.shape
        x = x.transpose(0,2,1,3).reshape(b, s, h*hd)
        return x

    def __call__(self, x_q, x_kv, mask=None, causal=True):
        # matrix multiplication with weights
        Q = x_q @ self.W_q
        K = x_kv @ self.W_k
        V = x_kv @ self.W_v

        Qh = self.split_heads(Q)  # (batch, heads, seq_q, head_dim)
        Kh = self.split_heads(K)  # (batch, heads, seq_k, head_dim)
        Vh = self.split_heads(V)  # (batch, heads, seq_k, head_dim)

        out_h, attn = self.attn(Qh, Kh, Vh, causal=causal)
        out = self.combine_heads(out_h)
        out = out @ self.W_o
        return out, attn