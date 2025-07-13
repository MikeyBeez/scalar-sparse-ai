# scalar-sparse-ai

Implementation of Scalar-Sparse™ architecture for hyper-context AI - a radical compression approach to enable massive context windows on consumer hardware.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for fast Python package management

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/username/scalar-sparse-ai.git
cd scalar-sparse-ai

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Running Experiments

```bash
# Run the basic proof of concept
python -m scalar_sparse_ai.scalar_sparse_poc

# Run sparse attention demonstration
python -m scalar_sparse_ai.sparse_attention_demo

# Run the original proof of concept
python -m scalar_sparse_ai.proof_of_concept

# Use the CLI
scalar-sparse compress --dimensions 1 4 8 16 32 64
scalar-sparse memory

# Run Colab-ready experiments locally
python colab_experiments.py
```

## 📊 Key Concepts

Based on research showing:
1. **95% Parameter Reduction**: Attention heads can be approximated by simple MLPs
2. **5% Token Importance**: Only 5-10% of token interactions are meaningful
3. **Massive Compression**: <10 dimensions per token enables 3M+ token contexts

## 🧪 Experiments

### 1. Compression Level Testing
Tests different dimension sizes (1-256) to find the "phase transition" point where coherent behavior emerges.

### 2. Sparse Attention
Demonstrates that attention mechanisms contain massive redundancy, with most computation being noise.

### 3. Memory Efficiency
Shows how Scalar-Sparse representation enables:
- 8K tokens → 40KB (fits in L3 cache)
- 100K tokens → 488KB (laptop RAM)
- 1M tokens → 4.8MB (consumer GPU)
- 3.2M tokens → 15.6MB (still fits in GPU!)

## 🛠️ Development

```bash
# Run tests
pytest

# Format code
black scalar_sparse_ai/
ruff check scalar_sparse_ai/

# Type checking
mypy scalar_sparse_ai/

# Start Jupyter for interactive experiments
jupyter notebook
```

## 📝 Project Structure

```
scalar-sparse-ai/
├── scalar_sparse_ai/             # Main package
│   ├── __init__.py
│   ├── cli.py                    # Command line interface
│   ├── scalar_sparse_poc.py      # Main experiments
│   ├── sparse_attention_demo.py  # Sparse attention experiments
│   └── proof_of_concept.py       # Original implementation
├── colab_experiments.py          # Google Colab ready script
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔬 Research Papers

This implementation is based on:
1. "Attention Heads Can Be Approximated by Simple Neural Networks" (Bee, 2025)
2. "Sparse MLP Approximation of Attention" (Bee, 2025)
3. "From Redundancy to Revolution: A Scalar-Sparse Architecture" (Bee, 2025)

## 🚀 Next Steps

- [ ] Full implementation of Scalar-Sparse encoder/decoder
- [ ] Benchmark on downstream tasks
- [ ] Build 1M+ token context demonstration
- [ ] Optimize with CUDA kernels for production speed

## 📄 License

MIT License - see LICENSE file for details.
