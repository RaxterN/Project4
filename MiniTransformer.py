import torch
from torch import nn as nn

class model(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ff_dim, max_len):
        super().__init__()

        #Define token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        #Define masked multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        #Define feedforward network with ReLU activation
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_dim),  # hidden size
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)   # project back to d_model
        )

        #Define LayerNorm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        #Define output projection to vocab size
        self.output_linear = nn.Linear(d_model, vocab_size)

        #Define softmax layer for output probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        seq_len = x.size(1)
        batch_size = x.size(0)

        #Token + positional embedding
        tok_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb

        #Create causal (upper-triangular) attention mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        #Apply masked self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)

        #Apply feedforward network
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)

        #Project to vocab and apply softmax
        logits = self.output_linear(x)
        probs = self.softmax(logits)

        return probs  # shape: (batch, seq_len, vocab_size)