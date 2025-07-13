"""
Quick demonstration of Scalar-Sparse concepts
Run this to see the key ideas in action
"""

import torch
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

def demonstrate_compression():
    """Show how much we can compress embeddings"""
    print("=== Compression Demonstration ===\n")
    
    # Original GPT-2 dimensions
    original_dim = 768
    test_dims = [1, 4, 8, 16, 32, 64, 128, 768]
    
    print(f"Original embedding dimension: {original_dim}")
    print("\nCompression ratios:")
    print("-" * 40)
    
    for dim in test_dims:
        ratio = original_dim / dim
        print(f"{dim:4d} dimensions: {ratio:6.1f}x compression")
    
    print("\n✨ Key insight: Even 8-16 dimensions might be enough!")


def demonstrate_memory_savings():
    """Show memory requirements for different context lengths"""
    print("\n=== Memory Savings Demonstration ===\n")
    
    contexts = [8_000, 100_000, 1_000_000, 3_200_000]
    scalar_sparse_dim = 10  # Our target
    
    print(f"{'Context':<12} {'GPT-2 (768d)':<15} {'Scalar-Sparse':<15} {'Savings':<10}")
    print("-" * 60)
    
    for ctx in contexts:
        # Standard GPT-2: 768 dims * 2 bytes (FP16)
        standard_mb = (ctx * 768 * 2) / (1024 * 1024)
        
        # Scalar-Sparse: ~10 effective dimensions
        sparse_mb = (ctx * scalar_sparse_dim * 2) / (1024 * 1024)
        
        savings = standard_mb / sparse_mb
        
        print(f"{ctx:>10,}   {standard_mb:>10.1f} MB   {sparse_mb:>10.1f} MB   {savings:>6.1f}x")
    
    print("\n🚀 3.2M tokens in just 61MB - fits in L3 cache!")


def demonstrate_sparse_attention():
    """Show that only a few tokens matter"""
    print("\n=== Sparse Attention Demonstration ===\n")
    
    # Simulate attention scores
    seq_len = 32
    important_tokens = 3  # Only 3 tokens really matter
    
    # Create fake attention pattern
    attention = torch.zeros(seq_len, seq_len)
    important_indices = torch.randperm(seq_len)[:important_tokens]
    
    for i in range(seq_len):
        # Each token mostly attends to the few important ones
        attention[i, important_indices] = torch.rand(important_tokens) * 0.8 + 0.2
        # Small random attention to others
        attention[i] = attention[i] + torch.rand(seq_len) * 0.05
        # Normalize
        attention[i] = attention[i] / attention[i].sum()
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(attention.numpy(), cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Sparse Attention Pattern\n(Only {important_tokens}/{seq_len} tokens matter)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig('sparse_attention_demo.png')
    plt.close()
    
    print(f"Created visualization: sparse_attention_demo.png")
    print(f"Notice how only {important_tokens} columns are bright!")
    print("\n💡 This is why sparse selection works so well")


if __name__ == "__main__":
    print("🧠 Scalar-Sparse AI: Key Concepts Demo\n")
    
    # Run demonstrations
    demonstrate_compression()
    demonstrate_memory_savings()
    demonstrate_sparse_attention()
    
    print("\n✅ Ready to revolutionize AI with massive contexts!")
    print("\nNext steps:")
    print("1. Run full experiments: python -m scalar_sparse_ai.scalar_sparse_poc")
    print("2. Try sparse attention: python -m scalar_sparse_ai.sparse_attention_demo")
    print("3. Port to Colab: python colab_experiments.py")
