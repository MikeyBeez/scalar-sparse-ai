"""
CPU-Optimized Scalar-Sparse ARC Experiments
Using tricks to make it work without GPU
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import time

# Force CPU and optimize
torch.set_num_threads(8)  # Use all CPU cores
torch.set_grad_enabled(False)  # We're just experimenting


class TinyScalarSparseARC:
    """
    Ultra-lightweight version for CPU experiments
    Key insight: We can test the concept with tiny models
    """
    
    def __init__(self):
        # Use int8 quantization for CPU efficiency
        self.dtype = torch.int8
        self.compression_ratio = 96  # 768 -> 8 dimensions
        
    def compress_arc_grid(self, grid: List[List[int]]) -> np.ndarray:
        """
        Compress ARC grid to 8 numbers
        This tests if patterns can be captured in minimal dimensions
        """
        grid_array = np.array(grid, dtype=np.uint8)
        
        # Extract 8 key features (ultra-compressed)
        features = np.zeros(8, dtype=np.float32)
        
        # Feature 1: Dominant color (excluding 0/black)
        colors, counts = np.unique(grid_array[grid_array > 0], return_counts=True)
        if len(colors) > 0:
            features[0] = colors[np.argmax(counts)] / 10.0
        
        # Feature 2: Color count
        features[1] = len(colors) / 10.0
        
        # Feature 3: Symmetry score (horizontal)
        if grid_array.shape[1] > 1:
            h_sym = np.mean(grid_array == np.fliplr(grid_array))
            features[2] = h_sym
        
        # Feature 4: Symmetry score (vertical)  
        if grid_array.shape[0] > 1:
            v_sym = np.mean(grid_array == np.flipud(grid_array))
            features[3] = v_sym
        
        # Feature 5: Density (non-zero cells)
        features[4] = np.mean(grid_array > 0)
        
        # Feature 6: Grid size encoding
        features[5] = grid_array.shape[0] / 30.0  # Normalize by max size
        features[6] = grid_array.shape[1] / 30.0
        
        # Feature 7: Pattern hash (simplified)
        # This captures structural patterns
        pattern_hash = 0
        for i in range(min(3, grid_array.shape[0])):
            for j in range(min(3, grid_array.shape[1])):
                pattern_hash ^= int(grid_array[i, j]) << ((i * 3 + j) % 8)
        features[7] = (pattern_hash % 100) / 100.0
        
        return features
    
    def find_similar_patterns(self, query_features: np.ndarray, 
                            pattern_library: List[np.ndarray],
                            top_k: int = 5) -> List[int]:
        """
        Find most similar patterns using compressed representations
        This is where sparse attention would help!
        """
        similarities = []
        
        for i, pattern in enumerate(pattern_library):
            # Cosine similarity in compressed space
            sim = np.dot(query_features, pattern) / (
                np.linalg.norm(query_features) * np.linalg.norm(pattern) + 1e-8
            )
            similarities.append((i, sim))
        
        # Return top-k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:top_k]]


def demonstrate_compression():
    """Show how ARC grids compress to 8 dimensions"""
    
    # Example ARC grid
    grid = [
        [0, 0, 1, 1, 0],
        [0, 1, 2, 2, 1],
        [1, 2, 3, 3, 2],
        [0, 1, 2, 2, 1],
        [0, 0, 1, 1, 0]
    ]
    
    compressor = TinyScalarSparseARC()
    compressed = compressor.compress_arc_grid(grid)
    
    print("Original grid (5x5 = 25 values):")
    for row in grid:
        print(row)
    
    print(f"\nCompressed to {len(compressed)} dimensions:")
    print(compressed)
    print(f"\nCompression ratio: {25 / len(compressed):.1f}x")
    
    # Show what each dimension captures
    feature_names = [
        "Dominant color", "Color diversity", "H-symmetry", "V-symmetry",
        "Density", "Height", "Width", "Pattern hash"
    ]
    
    print("\nFeature breakdown:")
    for name, value in zip(feature_names, compressed):
        print(f"  {name:15s}: {value:.3f}")


def benchmark_cpu_performance():
    """Test how many patterns we can process on CPU"""
    
    compressor = TinyScalarSparseARC()
    
    # Generate random grids
    n_patterns = 1000
    grids = []
    for _ in range(n_patterns):
        size = np.random.randint(3, 15)
        grid = np.random.randint(0, 5, (size, size)).tolist()
        grids.append(grid)
    
    # Time compression
    start = time.time()
    compressed_patterns = [compressor.compress_arc_grid(g) for g in grids]
    compress_time = time.time() - start
    
    # Time similarity search
    query = compressed_patterns[0]
    start = time.time()
    similar = compressor.find_similar_patterns(query, compressed_patterns[1:], top_k=10)
    search_time = time.time() - start
    
    print(f"\nCPU Performance Benchmark:")
    print(f"  Compressed {n_patterns} patterns in {compress_time:.3f}s")
    print(f"  ({n_patterns/compress_time:.0f} patterns/second)")
    print(f"  Similarity search in {search_time:.6f}s")
    print(f"  Total memory: {len(compressed_patterns) * 8 * 4 / 1024:.1f} KB")
    
    # Context capacity calculation
    memory_gb = 8  # Assume 8GB available
    bytes_per_pattern = 8 * 4  # 8 floats * 4 bytes
    max_patterns = (memory_gb * 1024**3) / bytes_per_pattern
    
    print(f"\nWith {memory_gb}GB RAM, could hold {max_patterns:,.0f} compressed patterns!")


if __name__ == "__main__":
    print("=== Scalar-Sparse ARC: CPU Experiments ===\n")
    
    # Demo 1: Compression
    demonstrate_compression()
    
    # Demo 2: Performance
    print("\n" + "="*50)
    benchmark_cpu_performance()
    
    print("\n✨ Key insight: Even on CPU, we can process thousands of patterns!")
    print("   With GPU, this scales to millions...")
