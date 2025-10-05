## Transformer From Scratch

Implementing the Transformer (decoder only) architecture from scratch, using numpy. A canon event for every NLP students. 

This transformer model has 4 big components:
- Embeddings: to embed tokens and positional relevance (using sinusoidal).
- Attention: the heart of the architecture, uses scaled dot product attention and multi head attention. Since this is a decoder only model, this component also has a causal mask function.
- Fast-Forward Network: all the layers needed for the transformer "training".
- Transformer: the transformer decoder block itself. 

More info in the report file. 

## Credits

Fidelya Fredelina (22/496507/TK/54405)