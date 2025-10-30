#!/bin/bash
# Setup script to run inside Docker container

set -e

echo "🚀 Setting up Python environment with uv..."

# Sync dependencies
echo "📦 Running: GIT_LFS_SKIP_SMUDGE=1 uv sync"
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install additional packages
echo "📦 Installing additional packages..."
uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
