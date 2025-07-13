"""Test that the compression experiment handles edge cases properly"""

import torch
from scalar_sparse_ai.scalar_sparse_poc import ScalarSparseEncoder, ScalarSparseDecoder

def test_edge_cases():
    """Test encoder/decoder with various dimension sizes including edge cases"""
    
    d_model = 768  # GPT-2 hidden size
    test_dims = [1, 2, 3, 4, 8, 16]
    batch_size = 2
    seq_len = 10
    
    print("Testing edge cases for Scalar-Sparse encoding...")
    print("-" * 50)
    
    for n_dims in test_dims:
        # Calculate sparse dimensions
        if n_dims <= 2:
            sparse_dims = 0
        else:
            sparse_dims = n_dims - 2
        
        print(f"\nTesting {n_dims} total dimensions (sparse: {sparse_dims})...")
        
        try:
            # Create encoder/decoder
            encoder = ScalarSparseEncoder(d_model, sparse_dims)
            decoder = ScalarSparseDecoder(d_model, sparse_dims)
            
            # Test data
            embeddings = torch.randn(batch_size, seq_len, d_model)
            
            # Encode
            encoded = encoder(embeddings)
            
            # Check dimensions
            assert encoded['base_value'].shape == (batch_size, seq_len, 1)
            assert encoded['modulator'].shape == (batch_size, seq_len, 1)
            assert encoded['sparse_gates'].shape == (batch_size, seq_len, sparse_dims)
            assert encoded['full'].shape == (batch_size, seq_len, n_dims)
            
            # Decode
            decoded = decoder(encoded['full'])
            assert decoded.shape == embeddings.shape
            
            # Calculate compression
            original_params = embeddings.numel()
            compressed_params = encoded['full'].numel()
            compression_ratio = original_params / compressed_params
            
            print(f"  ✓ Success! Compression ratio: {compression_ratio:.1f}x")
            print(f"    - Base value: {encoded['base_value'].shape}")
            print(f"    - Sparse gates: {encoded['sparse_gates'].shape}")
            print(f"    - Modulator: {encoded['modulator'].shape}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✅ Edge case testing complete!")


if __name__ == "__main__":
    test_edge_cases()
