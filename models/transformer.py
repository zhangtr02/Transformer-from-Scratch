import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)   # (L, D)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (D/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)    # (L, D/2)
        pe[:, 1::2] = torch.cos(position * div_term)    # (L, D/2)
        pe = pe.unsqueeze(0)    # (1, L, D)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]   # (B, L, D) + (1, L, D) -> (B, L, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K: (batch_size, num_heads, seq_len, d_k)
        # V: (batch_size, num_heads, seq_len, d_k)
        # mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Apply mask
            
        attention_weights = F.softmax(scores, dim=-1)   # one query to all keys attention weights
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)   # query weighted by values (batch_size, num_heads, seq_len, d_k)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output = self.scaled_dot_product_attention(Q, K, V, mask)   # (batch_size, num_heads, seq_len, d_k)
        
        # Reshape and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)
        return self.W_o(output) # (batch_size, seq_len, d_model)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, max_seq_length: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, max_seq_length: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 max_seq_length: int = 100, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff,
                             num_layers, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff,
                             num_layers, max_seq_length, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask)    # (batch_size, src_len, d_model)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)  # (batch_size, tgt_len, d_model)
        return self.linear(dec_output)  # (batch_size, tgt_len, tgt_vocab_size)
    
    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks for source and target sequences
        
        Args:
            src: Source sequence tensor of shape [batch_size, src_len]
            tgt: Target sequence tensor of shape [batch_size, tgt_len]
            
        Returns:
            src_mask: (batch_size, 1, 1, src_len)
            tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        """
        device = src.device
        # Source mask: mask padding tokens
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, src_len]

        # Target mask: mask padding tokens and future tokens
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, tgt_len]
        tgt_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()  # [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & nopeak_mask.unsqueeze(0)  # [batch_size, 1, tgt_len, tgt_len]
        return src_mask, tgt_mask
    
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor,
                max_length: int, start_token: int, end_token: int) -> torch.Tensor:
        """
        Generate translation using greedy decoding
        
        Args:
            src: Source sequence tensor (batch_size, src_len)
            src_mask: Source mask tensor (batch_size, 1, 1, src_len)
            max_length: Maximum length of generated sequence
            start_token: Start token index
            end_token: End token index
            
        Returns:
            Generated sequence tensor (batch_size, max_length)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Initialize target sequence with start token
        tgt = torch.ones(batch_size, 1, dtype=torch.long, device=device) * start_token

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate sequence
        for _ in range(max_length - 1):
            # Create target mask
            _, tgt_mask = self.generate_mask(src, tgt)
            
            # Get predictions
            with torch.no_grad():
                output = self.forward(src, tgt, src_mask, tgt_mask) # (batch_size, tgt_len, tgt_vocab_size)
            
            # Get next token (greedy decoding)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True) # (batch_size, 1）
            next_token[finished] = end_token
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == end_token)
            
            # Check if all sequences are finished
            if finished.all():
                break
        
        return tgt