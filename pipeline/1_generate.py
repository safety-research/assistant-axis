#!/usr/bin/env python3
"""
Generate model responses for all roles using vLLM batch inference.

This script loads role files and generates model responses for each role
using the role-specific system prompts.

Usage:
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --roles_dir data/prompts/roles \
        --questions_file data/prompts/questions.jsonl \
        --output_dir outputs/gemma-2-27b/responses \
        --question_count 240
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import RoleResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate role responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--roles_dir', type=str, required=True, help='Directory containing role JSON files')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to questions JSONL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSONL files')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum model context length')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--question_count', type=int, default=240, help='Number of questions per role')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--prompt_indices', type=str, default=None, help='Comma-separated prompt indices (e.g., "0,1,2")')
    parser.add_argument('--roles', nargs='+', help='Specific roles to process')
    parser.add_argument('--no_skip_existing', action='store_true', help='Process all roles even if output exists')

    args = parser.parse_args()

    # Parse prompt indices
    prompt_indices = None
    if args.prompt_indices:
        prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(',')]

    generator = RoleResponseGenerator(
        model_name=args.model,
        roles_dir=args.roles_dir,
        output_dir=args.output_dir,
        questions_file=args.questions_file,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        question_count=args.question_count,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        prompt_indices=prompt_indices,
    )

    generator.process_all_roles(
        skip_existing=not args.no_skip_existing,
        roles=args.roles
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
