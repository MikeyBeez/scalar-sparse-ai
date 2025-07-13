"""Command line interface for scalar-sparse-ai"""

import argparse
from pathlib import Path

from scalar_sparse_ai.scalar_sparse_poc import compression_experiment, visualize_results
from scalar_sparse_ai.sparse_attention_demo import demonstrate_scalar_sparse_concept
from scalar_sparse_ai.minimal_encoder import MinimalEncoder, MinimalDecoder


def main():
    parser = argparse.ArgumentParser(
        description="Scalar-Sparse AI: Ultra-compressed transformer architectures"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compression experiment
    compress_parser = subparsers.add_parser(
        "compress", 
        help="Run compression experiments"
    )
    compress_parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128],
        help="Dimension sizes to test"
    )
    
    # Memory demo
    memory_parser = subparsers.add_parser(
        "memory",
        help="Demonstrate memory savings"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "compress":
        print("Running compression experiments...")
        results = compression_experiment(args.dimensions)
        visualize_results(results)
        
    elif args.command == "memory":
        demonstrate_scalar_sparse_concept()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
