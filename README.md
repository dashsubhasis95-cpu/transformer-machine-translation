# Transformer-Machine_Translation

This repository contains my **from-scratch implementation of the Transformer architecture for machine translation, based on the paper **“Attention Is All You Need”.

The project is being developed **step by step toward a full translation system.  
At the current stage, the "Transformer encoder is implemented first" to build a strong foundation before adding the decoder and training pipeline.

---

## Why I built this

Most tutorials use high-level APIs like 'nn.Transformer', which hide important internal details.  
I wanted to understand "how Transformers actually work internally", especially for translation tasks.

My goals were to:

- Understand "self-attention" step by step"
- Learn why embeddings are scaled by '√d_model'
- Understand what "Q · Kᵀ" represents
- See how multiplying with 'V' gives meaningful word information
- Clearly understand the role of **residual (skip) connections**
- Track **tensor shapes** inside attention

---

## Input Embedding and Scaling (Why √d_model?)

Token IDs are first converted into vectors using an embedding layer.

In the forward pass, the embedding output is multiplied by `√d_model`:


Embedding(x) * √d_model

Why this is done:
Embedding values are usually small
Attention uses dot products (Q · Kᵀ)
Without scaling, dot-product values become too small
Small values entering softmax reduce learning effectiveness
Scaling by √d_model keeps values stable and improves training behavior.

## Positional Encoding

Transformers do not process tokens sequentially, so they do not know word order by default.

Sinusoidal positional encoding is added to embeddings so that:
     -Each position has a unique representation
     -The model can learn the order of words in a sentence

## Multi-Head Self-Attention (Intuition + Shapes)
# Creating Q, K, and V

Input shape:
(batch_size, seq_len, d_model)

After linear projection:
Q, K, V → (batch_size, seq_len, d_model)

After splitting into heads:
(batch_size, num_heads, seq_len, d_k)

# Q · Kᵀ — How words are related
 Q @ Kᵀ


 Resulting shape:

 (batch_size, num_heads, seq_len, seq_len)


# Meaning:
    Each word is compared with every other word
    This step measures how strongly words are related
    It answers the question:
    “Which words should this word pay attention to?”
At this stage, this gives relationships, not information.

# Softmax — Importance scores
Softmax converts relationship scores into attention weights:
   Higher weight → more important word
   Lower weight → less important word

#Multiply with V — Getting information
    Attention = softmax(QKᵀ / √d_k) @ V
    Output shape:
    (batch_size, num_heads, seq_len, d_k)

   # Why multiply with V:
       Q · Kᵀ tells which words matter
       V contains the actual information of each word
       Multiplication mixes word information based on importance

   #In simple terms:
      QKᵀ → relationship
      V → information

## Residual Connections (Skip Connections)

After each major sub-layer (attention and feed-forward), a residual connection is applied:
Output = x + Sublayer(x)
This is followed by layer normalization.
Why residual connections are important:
    They allow gradients to flow easily
    Prevent vanishing gradient problems
    Make deep Transformer stacks train stably

Residual connections are used with:
      Multi-head self-attention
      Feed-forward network

## Encoder Block Structure
Input
 → Multi-Head Self-Attention
 → Add (skip connection) + LayerNorm
 → Feed-Forward Network
 → Add (skip connection) + LayerNorm
 → Output


Multiple encoder blocks are stacked to form the complete encoder.

## Code structure
Transformer/
│
├── model.py      # Transformer encoder implementation
├── README.md


All components are implemented manually to ensure clarity and deep understanding.

## Current status
✅ Transformer encoder implemented
🚧 Decoder (masked self-attention + encoder–decoder attention) coming next
🚧 Training and translation inference to be added later

## What I learned from this
Why embedding scaling is important
How attention models word relationships
How values carry actual semantic information
Why residual connections are critical in deep networks
How tensor shapes evolve inside attention

## Tech used
# Python
# PyTorch

## References
Attention Is All You Need — Vaswani et al.
PyTorch documentation

## Author
Subhasis Dash
B.Tech student
Learning Transformers for machine translation from first principles


---

