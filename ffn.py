import numpy as np

# This file is for the layers

def relu(x):
    return np.maximum(0, x)

class FeedForward:
    def __init__(self, d_model, d_ff, activation=relu):
        # two-layer FFN: linear -> activation -> linear
        k1 = 1 / np.sqrt(d_model)
        k2 = 1 / np.sqrt(d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * k1
        self.b1 = np.zeros((d_ff,))
        self.W2 = np.random.randn(d_ff, d_model) * k2
        self.b2 = np.zeros((d_model,))
        self.activation = activation

    def __call__(self, x):
        # x: (batch, seq_len, d_model)
        x1 = x @ self.W1 + self.b1
        x1 = self.activation(x1)
        x2 = x1 @ self.W2 + self.b2
        return x2

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        # normalization layer after the FFN
        self.eps = eps
        self.gamma = np.ones((dim,))
        self.beta = np.zeros((dim,))

    def __call__(self, x):
        # x: (..., dim)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta