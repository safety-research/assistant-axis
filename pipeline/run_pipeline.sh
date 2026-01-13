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
OUTPUT_DIR="outputs/qwen-3-32b"
ROLES_DIR="data/prompts/roles"
QUESTIONS_FILE="data/prompts/questions.jsonl"

# Generation parameters
QUESTION_COUNT=240
MAX_TOKENS=512
TEMPERATURE=0.7

# Judge parameters
JUDGE_MODEL="gpt-4.1-mini"

echo "=== Assistant Axis Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Generate responses
echo "=== Step 1: Generating responses ==="
uv run pipeline/1_generate.py \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT_DIR/responses" \
    --question_count "$QUESTION_COUNT" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"

# Step 2: Extract activations
echo ""
echo "=== Step 2: Extracting activations ==="
uv run pipeline/2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations"

# Step 3: Score responses with judge LLM
echo ""
echo "=== Step 3: Scoring responses ==="
uv run pipeline/3_judge.py \
    --responses_dir "$OUTPUT_DIR/responses" \
    --roles_dir "$ROLES_DIR" \
    --output_dir "$OUTPUT_DIR/scores" \
    --judge_model "$JUDGE_MODEL"

# Step 4: Compute per-role vectors
echo ""
echo "=== Step 4: Computing per-role vectors ==="
uv run pipeline/4_vectors.py \
    --activations_dir "$OUTPUT_DIR/activations" \
    --scores_dir "$OUTPUT_DIR/scores" \
    --output_dir "$OUTPUT_DIR/vectors"

# Step 5: Compute final axis
echo ""
echo "=== Step 5: Computing axis ==="
uv run pipeline/5_axis.py \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --output "$OUTPUT_DIR/axis.pt"

echo ""
echo "=== Pipeline complete ==="
echo "Axis saved to: $OUTPUT_DIR/axis.pt"
