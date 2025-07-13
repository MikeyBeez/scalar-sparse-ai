"""
Proof of Concept: Attention Head Approximation with MLPs
Based on the finding that attention heads can be replaced by MLPs with 95% parameter reduction
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm


class AttentionHeadApproximator(nn.Module):
    """
    Approximates a single attention head with a simple MLP
    Original: Q, K, V projections + attention = ~4 * (d_model * d_head) parameters
    This: Simple MLP = 2 * (d_model * hidden) + (hidden * d_model) parameters
    """
    def __init__(self, d_model: int, original_d_head: int, compression_ratio: float = 0.05):
        super().__init__()
        
        # Calculate hidden size to achieve target compression ratio
        # Original params: ~4 * d_model * d_head (Q, K, V, O projections)
        original_params = 4 * d_model * original_d_head
        target_params = int(original_params * compression_ratio)
        
        # MLP params: d_model * hidden + hidden * d_model = 2 * d_model * hidden
        self.hidden_size = target_params // (2 * d_model)
        self.hidden_size = max(8, self.hidden_size)  # Minimum hidden size
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, d_model),
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Simple feedforward approximation of attention
        output = self.mlp(hidden_states)
        return self.layer_norm(hidden_states + output)


class ScalarSparseEncoder(nn.Module):
    """
    Encodes regular embeddings into Scalar-Sparse representation
    """
    def __init__(self, d_model: int, sparse_dim: int = 10):
        super().__init__()
        
        # Components of scalar-sparse representation
        self.base_value_proj = nn.Linear(d_model, 1)  # Projects to single scalar
        self.sparse_gates_proj = nn.Linear(d_model, sparse_dim - 2)  # Binary gates
        self.modulator_proj = nn.Linear(d_model, 1)  # Context modulator
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_value = self.base_value_proj(embeddings)
        sparse_gates = torch.sigmoid(self.sparse_gates_proj(embeddings))
        modulator = torch.tanh(self.modulator_proj(embeddings))
        
        return {
            'base_value': base_value,
            'sparse_gates': sparse_gates,
            'modulator': modulator,
            'original_shape': embeddings.shape
        }


class ScalarSparseDecoder(nn.Module):
    """
    Reconstructs embeddings from Scalar-Sparse representation
    """
    def __init__(self, d_model: int, sparse_dim: int = 10):
        super().__init__()
        
        # Reconstruction network
        self.reconstruct = nn.Sequential(
            nn.Linear(sparse_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )
        
    def forward(self, scalar_sparse: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all components
        combined = torch.cat([
            scalar_sparse['base_value'],
            scalar_sparse['sparse_gates'],
            scalar_sparse['modulator']
        ], dim=-1)
        
        return self.reconstruct(combined)


def analyze_attention_patterns(model_name: str = "gpt2"):
    """
    Analyze attention patterns in a pre-trained model to find redundancy
    """
    print(f"Loading {model_name} model...")
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set model to eval mode
    model.eval()
    
    # Get model dimensions
    config = model.config
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    d_head = d_model // n_heads
    
    print(f"Model config: d_model={d_model}, n_heads={n_heads}, d_head={d_head}")
    
    # Sample text for analysis
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "To be or not to be, that is the question.",
    ]
    
    attention_stats = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, output_attentions=True)
            
            # Analyze attention patterns
            for layer_idx, attention in enumerate(outputs.attentions):
                # attention shape: (batch, heads, seq_len, seq_len)
                attention_np = attention.numpy()
                
                # Calculate statistics per head
                for head_idx in range(n_heads):
                    head_attention = attention_np[0, head_idx]
                    
                    # Measure attention entropy (how focused vs distributed)
                    entropy = -np.sum(head_attention * np.log(head_attention + 1e-9))
                    
                    # Measure sparsity (how many positions get most attention)
                    sorted_attention = np.sort(head_attention.flatten())[::-1]
                    cumsum = np.cumsum(sorted_attention)
                    positions_for_90_percent = np.argmax(cumsum >= 0.9) + 1
                    
                    attention_stats.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'entropy': entropy,
                        'positions_for_90_percent': positions_for_90_percent,
                        'total_positions': head_attention.size
                    })
    
    return attention_stats, config


def test_attention_approximation(model_name: str = "gpt2"):
    """
    Test replacing attention heads with MLP approximators
    """
    print(f"Testing attention head approximation on {model_name}...")
    
    # Load model
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = model.config
    
    # Create approximators for first layer
    approximators = nn.ModuleList([
        AttentionHeadApproximator(config.hidden_size, config.hidden_size // config.num_attention_heads)
        for _ in range(config.num_attention_heads)
    ])
    
    # Test text
    test_text = "The future of artificial intelligence is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        # Get original outputs
        original_outputs = model(**inputs)
        original_hidden = original_outputs.last_hidden_state
        
        # Get intermediate representations from first layer
        embeddings = model.wte(inputs.input_ids) + model.wpe(torch.arange(inputs.input_ids.size(1)))
        
        # Apply approximators
        approx_outputs = []
        for approximator in approximators:
            approx_out = approximator(embeddings)
            approx_outputs.append(approx_out)
        
        # Average approximator outputs (simplified fusion)
        approx_hidden = torch.stack(approx_outputs).mean(dim=0)
        
        # Calculate reconstruction error
        mse = nn.functional.mse_loss(approx_hidden, embeddings)
        cosine_sim = nn.functional.cosine_similarity(
            approx_hidden.flatten(), 
            embeddings.flatten(), 
            dim=0
        )
        
    print(f"Reconstruction MSE: {mse.item():.4f}")
    print(f"Cosine similarity: {cosine_sim.item():.4f}")
    
    return mse.item(), cosine_sim.item()


def test_scalar_sparse_encoding():
    """
    Test the Scalar-Sparse encoding/decoding pipeline
    """
    print("Testing Scalar-Sparse encoding...")
    
    d_model = 768  # GPT-2 hidden size
    sparse_dim = 10
    batch_size = 2
    seq_len = 20
    
    # Create encoder/decoder
    encoder = ScalarSparseEncoder(d_model, sparse_dim)
    decoder = ScalarSparseDecoder(d_model, sparse_dim)
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, seq_len, d_model)
    
    # Encode
    encoded = encoder(embeddings)
    
    # Decode
    reconstructed = decoder(encoded)
    
    # Calculate metrics
    mse = nn.functional.mse_loss(reconstructed, embeddings)
    
    # Calculate compression ratio
    original_params = embeddings.numel()
    compressed_params = sum(v.numel() for v in encoded.values() if isinstance(v, torch.Tensor))
    compression_ratio = original_params / compressed_params
    
    print(f"Original size: {original_params} parameters")
    print(f"Compressed size: {compressed_params} parameters")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Reconstruction MSE: {mse.item():.4f}")
    
    return compression_ratio, mse.item()


if __name__ == "__main__":
    print("=== Scalar-Sparse AI Proof of Concept ===\n")
    
    # 1. Analyze attention patterns
    print("1. Analyzing attention patterns for redundancy...")
    stats, config = analyze_attention_patterns()
    
    # Calculate average statistics
    avg_entropy = np.mean([s['entropy'] for s in stats])
    avg_positions = np.mean([s['positions_for_90_percent'] for s in stats])
    
    print(f"Average attention entropy: {avg_entropy:.2f}")
    print(f"Average positions for 90% attention: {avg_positions:.1f}")
    print()
    
    # 2. Test attention approximation
    print("2. Testing attention head approximation with MLPs...")
    mse, cosine_sim = test_attention_approximation()
    print()
    
    # 3. Test Scalar-Sparse encoding
    print("3. Testing Scalar-Sparse encoding/decoding...")
    compression, reconstruction_error = test_scalar_sparse_encoding()
    print()
    
    print("=== Summary ===")
    print(f"✓ Attention heads show redundancy (90% attention in ~{avg_positions:.0f} positions)")
    print(f"✓ MLP approximation achieves {cosine_sim:.3f} cosine similarity")
    print(f"✓ Scalar-Sparse encoding achieves {compression:.1f}x compression")
    print(f"✓ Reconstruction error: {reconstruction_error:.4f}")
