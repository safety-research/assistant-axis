#!/bin/bash
#
# Example pipeline for computing the assistant axis.
#
# This script runs all 5 steps of the pipeline for a given model.
# Adjust the parameters below for your setup.
#
# Usage:
#   ./pipeline/run_pipeline.sh
#
# Requirements:
#   - OPENAI_API_KEY environment variable (for step 3)
#   - Sufficient GPU memory for the model

set -e  # Exit on error

# Configuration
MODEL="Qwen/Qwen3-32B"
OUTPUT_DIR="/workspace/qwen-3-32b/roles"

echo "=== Assistant Axis Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Generate responses
# echo "=== Step 1: Generating responses ==="
# uv run 1_generate.py \
#     --model "$MODEL" \
#     --output_dir "$OUTPUT_DIR/responses"

# Step 2: Extract activations (SET BATCH SIZE APPROPRIATELY)
echo ""
echo "=== Step 2: Extracting activations ==="
uv run 2_activations.py \
    --model "$MODEL" \
    --responses_dir "/workspace/qwen-3-32b/roles_240/responses" \
    --output_dir "$OUTPUT_DIR/activations" --roles "aberration"

# Step 3: Score responses with judge LLM
# echo ""
# echo "=== Step 3: Scoring responses ==="
# uv run 3_judge.py \
#     --responses_dir "$OUTPUT_DIR/responses" \
#     --output_dir "$OUTPUT_DIR/scores"

# # Step 4: Compute per-role vectors
# echo ""
# echo "=== Step 4: Computing per-role vectors ==="
# uv run 4_vectors.py \
#     --activations_dir "$OUTPUT_DIR/activations" \
#     --scores_dir "$OUTPUT_DIR/scores" \
#     --output_dir "$OUTPUT_DIR/vectors"

# # Step 5: Compute final axis
# echo ""
# echo "=== Step 5: Computing axis ==="
# uv run 5_axis.py \
#     --vectors_dir "$OUTPUT_DIR/vectors" \
#     --output "$OUTPUT_DIR/axis.pt"

# echo ""
# echo "=== Pipeline complete ==="
# echo "Axis saved to: $OUTPUT_DIR/axis.pt"
