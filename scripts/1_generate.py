#!/usr/bin/env python3
"""
Generate model responses for all roles.

This script loads role files and generates model responses for each role
using both the role-specific system prompts and default neutral prompts.

Usage:
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --roles_dir data/prompts/roles \
        --questions_file data/prompts/questions.jsonl \
        --output outputs/gemma-2-27b/rollouts.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.models import get_config, get_short_name
from assistant_axis.generation import generate_response, supports_system_prompt


def load_questions(questions_file: str) -> list:
    """Load questions from JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            questions.append(entry['question'])
    return questions


def load_role(role_file: str) -> dict:
    """Load a role JSON file."""
    with open(role_file, 'r') as f:
        return json.load(f)


def format_system_prompt(prompt: str, model_name: str, short_name: str) -> str:
    """Format system prompt, replacing {model_name} placeholder."""
    return prompt.replace("{model_name}", short_name)


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for roles")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--roles_dir", type=str, required=True, help="Directory containing role JSON files")
    parser.add_argument("--questions_file", type=str, required=True, help="Path to questions JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    parser.add_argument("--max_questions", type=int, default=240, help="Maximum questions per role")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process (default: all)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output exists")
    args = parser.parse_args()

    # Check if output exists
    output_path = Path(args.output)
    if args.skip_existing and output_path.exists():
        print(f"Output file exists, skipping: {output_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get model config
    config = get_config(args.model)
    short_name = config["short_name"]
    print(f"Model: {args.model} ({short_name})")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load questions
    print(f"Loading questions from {args.questions_file}")
    questions = load_questions(args.questions_file)[:args.max_questions]
    print(f"Loaded {len(questions)} questions")

    # Get role files
    roles_dir = Path(args.roles_dir)
    if args.roles:
        role_files = [roles_dir / f"{r}.json" for r in args.roles]
    else:
        role_files = sorted(roles_dir.glob("*.json"))

    print(f"Found {len(role_files)} role files")

    # Check system prompt support
    use_system_prompt = supports_system_prompt(args.model)
    print(f"System prompt support: {use_system_prompt}")

    # Generate responses
    rows = []

    for role_file in tqdm(role_files, desc="Processing roles"):
        role_name = role_file.stem
        role_data = load_role(role_file)

        instructions = role_data.get("instruction", [])
        if not instructions:
            print(f"Warning: No instructions for role {role_name}")
            continue

        # Process each instruction variant
        for prompt_idx, inst in enumerate(instructions):
            system_prompt_raw = inst.get("pos", "")
            system_prompt = format_system_prompt(system_prompt_raw, args.model, short_name)

            for q_idx, question in enumerate(tqdm(questions, desc=f"{role_name}[{prompt_idx}]", leave=False)):
                # Format conversation
                if use_system_prompt and system_prompt:
                    conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]
                else:
                    if system_prompt:
                        user_content = f"{system_prompt}\n\n{question}"
                    else:
                        user_content = question
                    conversation = [{"role": "user", "content": user_content}]

                # Generate response
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    conversation=conversation,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

                rows.append({
                    "role": role_name,
                    "prompt_idx": prompt_idx,
                    "question_idx": q_idx,
                    "system_prompt": system_prompt,
                    "user_prompt": question,
                    "response": response,
                })

    # Save to Parquet
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} responses to {output_path}")


if __name__ == "__main__":
    main()
