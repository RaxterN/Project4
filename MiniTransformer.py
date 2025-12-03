import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):
    """
    Tiny transformer language model using only built-in PyTorch modules.

    Shapes:
        input_ids: [batch_size, seq_len]  (ints: token indices)
        logits:    [batch_size, seq_len, vocab_size]
    """

    def __init__(
        self, vocab_size: int, d_model: int = 64, n_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 256, max_seq_len: int = 128, dropout: float = 0.1,):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        #Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        #Position embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        #Define one layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,)

        #Get the stack of layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,)

        #Layer norm
        self.ln_f = nn.LayerNorm(d_model)

        #Linear
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, input_ids: torch.Tensor):
        """
        
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        #Token embeddings
        token_emb = self.token_embedding(input_ids)

        #Position indices: [0, 1, ..., seq_len-1], broadcast to batch
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        pos_emb = pos_emb.expand(batch_size, seq_len, self.d_model)

        #token + positional embeddings
        x = token_emb + pos_emb

        #Causal mask that blocks 'future' tokens in the sequence
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        attention_mask = torch.triu(mask, diagonal=1)

        #Transformer stack
        x = self.transformer(x, mask=attention_mask)

        # Final layer norm + output projection
        x = self.ln_f(x)
        logits = self.output_head(x)

        return logits
    
########
# References
########
# https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/
#    > used as reference for the code
#
# https://github.com/pytorch/examples/tree/main/word_language_model
#    > also used as reference