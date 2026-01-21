#!/bin/bash
# Full pipeline runner script

set -e  # Exit on error

echo "========================================="
echo "Iterative Hard-Negative Mining Pipeline"
echo "========================================="

# Step 1: Prepare corpus
echo ""
echo "Step 1: Preparing corpus..."
python prepare_corpus.py

# Step 2: Prepare queries
echo ""
echo "Step 2: Preparing queries..."
python prepare_queries.py

# Step 3: Train specialized model
echo ""
echo "Step 3: Training specialized model..."
echo "Note: This requires an LLM endpoint (OpenAI API or vLLM)"
echo "Set OPENAI_API_KEY environment variable or pass --llm-endpoint"

# Example with OpenAI API (requires OPENAI_API_KEY)
python iterative_miner.py \
    --iterations 2 \
    --num-queries 1000 \
    --llm-model "gpt-3.5-turbo"

# Example with vLLM endpoint (uncomment to use)
# python iterative_miner.py \
#     --iterations 2 \
#     --num-queries 1000 \
#     --llm-model "Qwen/Qwen2.5-7B-Instruct" \
#     --llm-endpoint "http://localhost:8000/v1"

# Step 4: Evaluate
echo ""
echo "Step 4: Evaluating models..."
python evaluate.py

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
