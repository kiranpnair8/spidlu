#modules/transformer_block.py
import torch
import torch.nn as nn
from modules.spidlu_layer import SpiDLU

class SpiDLU_FFN(nn.Module):
    """
    The Feed-Forward Network modified with Spiking Dynamics.
    """
    def __init__(self, d_model, d_ff, alpha=0.9, threshold=1.0, T=4, use_spiking=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.use_spiking = use_spiking
        
        if self.use_spiking:
            self.activation = SpiDLU(alpha=alpha, threshold=threshold, T=T)
        else:
            # The Student uses standard GeLU
            self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SpiDLU_Block(nn.Module):
    """
    A single Transformer Block using Spi-dLU.
    """
    def __init__(self, d_model, nhead, d_ff, alpha=0.9, threshold=1.0, T=4, use_spiking=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = SpiDLU_FFN(d_model, d_ff, alpha=alpha, threshold=threshold, T=T, use_spiking=use_spiking)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Attention Layer (Standard ANN)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN Layer 
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x

class SpiDLU_Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, d_ff=2048, alpha=0.9, threshold=1.0, T=4, use_spiking=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model)) # Max seq len 512
        
        # Stacking the Spi-dLU Blocks
        self.layers = nn.ModuleList([
            SpiDLU_Block(d_model, nhead, d_ff, alpha=alpha, threshold=threshold, T=T, use_spiking=use_spiking) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x)
            
        return self.fc_out(x)