"""
Quick test of the scalar-sparse concept with better training
"""

from scalar_sparse_ai.scalar_sparse_poc import (
    compression_experiment, visualize_results, 
    ScalarSparseEncoder, ScalarSparseDecoder,
    MinimalEncoder, MinimalDecoder
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt


def quick_compression_test():
    """Run a focused test with better training"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2").to(device)
    model.eval()
    
    # Test dimensions
    test_dims = [1, 4, 8, 16, 32, 64]
    
    # Test text
    text = "The future of artificial intelligence lies in efficient architectures."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        original_embeddings = outputs.last_hidden_state
    
    print(f"\nOriginal embedding shape: {original_embeddings.shape}")
    print(f"Original size: {original_embeddings.numel() * 4 / 1024:.1f} KB")
    
    results = []
    
    for n_dims in test_dims:
        print(f"\n{'='*50}")
        print(f"Testing {n_dims} dimensions...")
        
        # Create encoder/decoder
        if n_dims <= 2:
            encoder = MinimalEncoder(768, n_dims).to(device)
            decoder = MinimalDecoder(768, n_dims).to(device)
        else:
            sparse_dims = n_dims - 2
            encoder = ScalarSparseEncoder(768, sparse_dims).to(device)
            decoder = ScalarSparseDecoder(768, sparse_dims).to(device)
        
        # Train with more epochs
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.01
        )
        
        print("Training...", end="", flush=True)
        for epoch in range(50):
            if n_dims <= 2:
                compressed = encoder(original_embeddings)
                reconstructed = decoder(compressed)
            else:
                sparse_repr = encoder(original_embeddings)
                compressed = sparse_repr['full']
                reconstructed = decoder(compressed)
            
            loss = F.mse_loss(reconstructed, original_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f" epoch {epoch}: {loss.item():.4f}", end="", flush=True)
        
        print(" Done!")
        
        # Evaluate
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            if n_dims <= 2:
                compressed = encoder(original_embeddings)
                reconstructed = decoder(compressed)
            else:
                sparse_repr = encoder(original_embeddings)
                compressed = sparse_repr['full']
                reconstructed = decoder(compressed)
            
            mse = F.mse_loss(reconstructed, original_embeddings).item()
            
            # Calculate compression
            compressed_size = compressed.numel() * 4 / 1024  # KB
            original_size = original_embeddings.numel() * 4 / 1024
            compression_ratio = original_size / compressed_size
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(
                reconstructed.flatten(),
                original_embeddings.flatten(),
                dim=0
            ).item()
        
        print(f"MSE: {mse:.4f}")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"Compressed size: {compressed_size:.1f} KB")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        
        results.append({
            'dimensions': n_dims,
            'mse': mse,
            'cosine_sim': cosine_sim,
            'compression_ratio': compression_ratio
        })
    
    # Plot results
    dims = [r['dimensions'] for r in results]
    mses = [r['mse'] for r in results]
    cosines = [r['cosine_sim'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE plot
    ax1.semilogy(dims, mses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Dimensions')
    ax1.set_ylabel('MSE (log scale)')
    ax1.set_title('Reconstruction Error vs Compression')
    ax1.grid(True, alpha=0.3)
    
    # Add threshold line
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='MSE = 1.0')
    ax1.legend()
    
    # Cosine similarity plot
    ax2.plot(dims, cosines, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Reconstruction Quality')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('compression_test_results.png', dpi=150)
    print(f"\nSaved plot to compression_test_results.png")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for r in results:
        print(f"{r['dimensions']:2d} dims: MSE={r['mse']:6.3f}, "
              f"Cosine={r['cosine_sim']:.3f}, "
              f"Compression={r['compression_ratio']:5.1f}x")
    
    # Find phase transition
    for i, r in enumerate(results):
        if r['cosine_sim'] > 0.9:
            print(f"\n✨ Phase transition at {r['dimensions']} dimensions!")
            print(f"   Achieves {r['compression_ratio']:.1f}x compression")
            print(f"   With {r['cosine_sim']:.3f} cosine similarity")
            break


if __name__ == "__main__":
    quick_compression_test()
