"""
Scalar-Sparse AI: Proof of Concept
Exploring minimal information representation for transformer models

Based on the research showing:
1. Attention heads can be approximated with 95% fewer parameters
2. Only 5-10% of token interactions are meaningful (rest is noise)
3. Massive architectural redundancy exists in transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from tqdm import tqdm

from scalar_sparse_ai.minimal_encoder import MinimalEncoder, MinimalDecoder


@dataclass
class ScalarSparseToken:
    """Minimal representation of a token"""
    base_value: float  # Core semantic scalar
    sparse_gates: torch.Tensor  # Binary routing mask
    modulator: float  # Contextual adjustment
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for processing"""
        return torch.cat([
            torch.tensor([self.base_value]),
            self.sparse_gates,
            torch.tensor([self.modulator])
        ])


class InformationDialExperiment:
    """
    Test different compression levels to find the phase transition
    from noise to coherent behavior
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.lm_model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.lm_model.eval()
        
        self.d_model = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.n_layers = self.model.config.num_hidden_layers
        
        print(f"Model: {model_name}")
        print(f"Hidden size: {self.d_model}")
        print(f"Attention heads: {self.n_heads}")
        print(f"Layers: {self.n_layers}")


class ScalarSparseEncoder(nn.Module):
    """Encode embeddings to ultra-compressed scalar-sparse representation"""
    
    def __init__(self, d_model: int, n_sparse_dims: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_sparse_dims = max(0, n_sparse_dims)  # Ensure non-negative
        
        # Learn to extract the core semantic value
        self.base_value_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Learn sparse gates for routing (only if we have dimensions for it)
        if self.n_sparse_dims > 0:
            self.sparse_gates_net = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_sparse_dims),
                nn.Sigmoid()  # Binary-ish outputs
            )
        else:
            self.sparse_gates_net = None
        
        # Learn contextual modulator
        self.modulator_net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1] range
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [batch, seq_len, d_model]
        Returns:
            Dictionary with compressed representations
        """
        base_values = self.base_value_net(embeddings)
        modulators = self.modulator_net(embeddings)
        
        if self.sparse_gates_net is not None:
            sparse_gates = self.sparse_gates_net(embeddings)
            # Binarize gates with straight-through estimator
            sparse_gates_binary = (sparse_gates > 0.5).float()
            sparse_gates = sparse_gates_binary + sparse_gates - sparse_gates.detach()
        else:
            # No sparse gates for very low dimensions
            sparse_gates = torch.empty(embeddings.shape[0], embeddings.shape[1], 0).to(embeddings.device)
        
        return {
            'base_value': base_values,  # [batch, seq_len, 1]
            'sparse_gates': sparse_gates,  # [batch, seq_len, n_sparse_dims]
            'modulator': modulators,  # [batch, seq_len, 1]
            'full': torch.cat([base_values, sparse_gates, modulators], dim=-1)
        }


class AttentionApproximator(nn.Module):
    """Approximate attention using only sparse tokens"""
    
    def __init__(self, d_model: int, n_sparse_dims: int = 8, sparsity_ratio: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_sparse_dims = n_sparse_dims
        self.sparsity_ratio = sparsity_ratio
        
        # Token importance scorer (query-independent for efficiency)
        self.importance_scorer = nn.Sequential(
            nn.Linear(n_sparse_dims + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # MLP approximator for attention
        self.attention_mlp = nn.Sequential(
            nn.Linear((n_sparse_dims + 2) * 3, 64),  # Q, K, V concatenated
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_sparse_dims + 2)
        )
    
    def forward(self, sparse_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Approximate attention using only important token interactions
        """
        batch_size, seq_len, _ = sparse_tokens['full'].shape
        
        # Score token importance
        importance_scores = self.importance_scorer(sparse_tokens['full']).squeeze(-1)
        
        # Select top-k tokens
        k = max(1, int(seq_len * self.sparsity_ratio))
        _, top_indices = torch.topk(importance_scores, k, dim=1)
        
        # Process only important interactions
        output = torch.zeros_like(sparse_tokens['full'])
        
        for b in range(batch_size):
            for i in range(seq_len):
                query = sparse_tokens['full'][b, i:i+1]
                
                # Get selected keys and values
                selected_indices = top_indices[b]
                keys = sparse_tokens['full'][b, selected_indices]
                values = sparse_tokens['full'][b, selected_indices]
                
                # Repeat query for batch processing
                query_repeated = query.repeat(k, 1)
                
                # Concatenate Q, K, V and process
                qkv = torch.cat([query_repeated, keys, values], dim=-1)
                attention_out = self.attention_mlp(qkv)
                
                # Average the outputs
                output[b, i] = attention_out.mean(dim=0)
        
        return output


class ScalarSparseDecoder(nn.Module):
    """Decode from scalar-sparse back to embedding space"""
    
    def __init__(self, d_model: int, n_sparse_dims: int = 8):
        super().__init__()
        self.n_sparse_dims = max(0, n_sparse_dims)
        
        # Total input dimensions: base_value (1) + sparse_gates (n_sparse_dims) + modulator (1)
        input_dim = self.n_sparse_dims + 2
        
        # Progressive upsampling
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
    
    def forward(self, sparse_repr: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse_repr)


def compression_experiment(dimension_sizes: List[int] = [1, 4, 8, 16, 32, 64, 128, 256]):
    """
    Test different compression levels to find the phase transition point
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = InformationDialExperiment("gpt2", device)
    
    # Test text
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we understand and interact with the world.",
        "To be or not to be, that is the question.",
    ]
    
    results = []
    
    for n_dims in dimension_sizes:
        print(f"\nTesting {n_dims} dimensions...")
        
        # Create encoder/decoder
        # For very small dimensions, use minimal encoder
        if n_dims <= 2:
            encoder = MinimalEncoder(experiment.d_model, n_dims).to(device)
            decoder = MinimalDecoder(experiment.d_model, n_dims).to(device)
            use_minimal = True
        else:
            # Reserve 2 dimensions for base value and modulator
            sparse_dims = n_dims - 2
            encoder = ScalarSparseEncoder(experiment.d_model, sparse_dims).to(device)
            decoder = ScalarSparseDecoder(experiment.d_model, sparse_dims).to(device)
            use_minimal = False
        
        # Simple training loop
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.001
        )
        
        losses = []
        
        for epoch in range(10):
            epoch_loss = 0
            
            for text in test_texts:
                # Tokenize
                inputs = experiment.tokenizer(text, return_tensors="pt", padding=True).to(device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = experiment.model(**inputs)
                    original_embeddings = outputs.last_hidden_state
                
                # Encode -> Decode
                if use_minimal:
                    compressed = encoder(original_embeddings)
                    reconstructed = decoder(compressed)
                else:
                    sparse_repr = encoder(original_embeddings)
                    reconstructed = decoder(sparse_repr['full'])
                
                # Loss
                loss = F.mse_loss(reconstructed, original_embeddings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(test_texts))
        
        # Test reconstruction quality
        encoder.eval()
        decoder.eval()
        
        total_mse = 0
        total_perplexity = 0
        
        with torch.no_grad():
            for text in test_texts:
                inputs = experiment.tokenizer(text, return_tensors="pt", padding=True).to(device)
                
                # Original embeddings and logits
                outputs = experiment.model(**inputs)
                original_embeddings = outputs.last_hidden_state
                
                lm_outputs = experiment.lm_model(**inputs)
                original_logits = lm_outputs.logits
                
                # Compressed representation
                if use_minimal:
                    compressed = encoder(original_embeddings)
                    reconstructed = decoder(compressed)
                    compressed_size = compressed.numel() * 4
                else:
                    sparse_repr = encoder(original_embeddings)
                    reconstructed = decoder(sparse_repr['full'])
                    compressed_size = sparse_repr['full'].numel() * 4
                
                # Feed reconstructed embeddings through rest of model
                # This is a simplified test - in practice we'd need custom forward
                
                # Calculate metrics
                mse = F.mse_loss(reconstructed, original_embeddings).item()
                total_mse += mse
                
                # Compression ratio
                original_size = original_embeddings.numel() * 4  # float32
                compression_ratio = original_size / compressed_size
        
        avg_mse = total_mse / len(test_texts)
        
        result = {
            'dimensions': n_dims,
            'mse': avg_mse,
            'compression_ratio': compression_ratio,
            'final_loss': losses[-1]
        }
        
        results.append(result)
        print(f"MSE: {avg_mse:.6f}, Compression: {compression_ratio:.1f}x")
    
    return results


def visualize_results(results: List[Dict]):
    """Plot the results of compression experiments"""
    dims = [r['dimensions'] for r in results]
    mses = [r['mse'] for r in results]
    ratios = [r['compression_ratio'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE vs dimensions
    ax1.semilogy(dims, mses, 'b-o')
    ax1.set_xlabel('Number of Dimensions')
    ax1.set_ylabel('Reconstruction MSE (log scale)')
    ax1.set_title('Reconstruction Error vs Compression Level')
    ax1.grid(True, alpha=0.3)
    
    # Add phase transition annotation
    if len(mses) > 4:
        # Find elbow point (simplified)
        gradients = np.gradient(np.log(mses))
        elbow_idx = np.argmin(gradients[1:]) + 1
        ax1.axvline(dims[elbow_idx], color='r', linestyle='--', alpha=0.5)
        ax1.text(dims[elbow_idx], max(mses), f'Phase transition?\n({dims[elbow_idx]} dims)', 
                ha='center', va='top')
    
    # Compression ratio
    ax2.plot(dims, ratios, 'g-o')
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratio vs Dimensions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalar_sparse_results.png')
    plt.show()


if __name__ == "__main__":
    print("=== Scalar-Sparse AI: Finding the Minimum Viable Information ===\n")
    
    # Run compression experiments
    print("Running compression experiments...")
    results = compression_experiment([1, 4, 8, 16, 32, 64, 128])
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results)
    
    # Print summary
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['dimensions']:3d} dims: MSE={r['mse']:.6f}, Compression={r['compression_ratio']:.1f}x")
    
    # Find the "sweet spot"
    mses = [r['mse'] for r in results]
    min_viable_idx = next(i for i, mse in enumerate(mses) if mse < 0.1)
    if min_viable_idx < len(results):
        sweet_spot = results[min_viable_idx]
        print(f"\nMinimum viable dimensions: {sweet_spot['dimensions']} (MSE < 0.1)")
        print(f"Achieves {sweet_spot['compression_ratio']:.1f}x compression")
