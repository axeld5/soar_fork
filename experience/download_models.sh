#!/bin/bash
# Standalone script to download required models for SOAR

echo "=== SOAR Model Downloader ==="
echo "This script will download the required models for SOAR experiments."
echo ""

# Create directories
echo "Creating model directories..."
mkdir -p hf
mkdir -p hf_cache

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "Error: UV is not installed. Please install UV first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if huggingface-cli is available
if ! uv run huggingface-cli --version &> /dev/null; then
    echo "Error: huggingface-cli is not available. Installing..."
    uv sync
fi

echo ""
echo "=== Downloading Qwen2.5-Coder-3B-Instruct model ==="
echo "This may take several minutes depending on your internet connection..."
uv run huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct \
    --local-dir hf/Qwen2.5-Coder-3B-Instruct \
    --local-dir-use-symlinks True \
    --cache-dir ./hf_cache

if [ $? -eq 0 ]; then
    echo "✅ Qwen2.5-Coder-3B-Instruct downloaded successfully!"
else
    echo "❌ Failed to download Qwen2.5-Coder-3B-Instruct"
    exit 1
fi

echo ""
echo "=== Downloading CodeRankEmbed model ==="
uv run huggingface-cli download nomic-ai/CodeRankEmbed \
    --local-dir hf/CodeRankEmbed \
    --local-dir-use-symlinks True \
    --cache-dir ./hf_cache

if [ $? -eq 0 ]; then
    echo "✅ CodeRankEmbed downloaded successfully!"
else
    echo "❌ Failed to download CodeRankEmbed"
    exit 1
fi

echo ""
echo "=== Validating downloads ==="

# Check if config files exist
if [ -f "hf/Qwen2.5-Coder-3B-Instruct/config.json" ]; then
    echo "✅ Qwen2.5-Coder-3B-Instruct config.json found"
else
    echo "❌ Qwen2.5-Coder-3B-Instruct config.json missing"
    exit 1
fi

if [ -f "hf/CodeRankEmbed/config.json" ]; then
    echo "✅ CodeRankEmbed config.json found"
else
    echo "❌ CodeRankEmbed config.json missing"
    exit 1
fi

echo ""
echo "🎉 All models downloaded and validated successfully!"
echo "You can now run the main experiment script: bash experience/qwen.sh"
echo ""
echo "Model locations:"
echo "  - Base model: $(pwd)/hf/Qwen2.5-Coder-3B-Instruct"
echo "  - Embed model: $(pwd)/hf/CodeRankEmbed"
