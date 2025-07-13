"""
Minimal encoder for extreme compression experiments
"""

import torch
import torch.nn as nn


class MinimalEncoder(nn.Module):
    """Ultra-minimal encoder for 1-2 dimension experiments"""
    
    def __init__(self, d_model: int, n_dims: int):
        super().__init__()
        self.n_dims = n_dims
        
        if n_dims == 1:
            # Only base value
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
        elif n_dims == 2:
            # Base value + modulator
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Tanh()
            )
        else:
            raise ValueError(f"MinimalEncoder only supports 1-2 dimensions, got {n_dims}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encoder(embeddings)


class MinimalDecoder(nn.Module):
    """Decoder for ultra-minimal representations"""
    
    def __init__(self, d_model: int, n_dims: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(n_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
    
    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        return self.decoder(compressed)
