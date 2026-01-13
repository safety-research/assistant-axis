#!/usr/bin/env python3
"""
Score role responses using a judge LLM.

This script scores how well model responses adhere to their assigned roles
using an LLM judge (e.g., GPT-4). Scores are on a 0-3 scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Usage:
    uv run scripts/3_judge.py \
        --rollouts outputs/gemma-2-27b/rollouts.parquet \
        --roles_dir data/prompts/roles \
        --output_dir outputs/gemma-2-27b/scores \
        --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import score_responses, RateLimiter, call_judge_batch, parse_judge_score
import openai

load_dotenv()


def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


async def process_role(
    role: str,
    role_df: pd.DataFrame,
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Process a single role and return scores."""
    # Build prompts for each response
    prompts = []
    keys = []

    for _, row in role_df.iterrows():
        prompt_idx = row["prompt_idx"]
        question_idx = row["question_idx"]
        question = row["user_prompt"]
        response = row["response"]

        # Fill in template
        judge_prompt = eval_prompt_template.format(
            question=question,
            answer=response
        )
        prompts.append(judge_prompt)
        keys.append(f"q{question_idx}_p{prompt_idx}")

    # Call judge
    responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = {}
    for key, response_text in zip(keys, responses):
        if response_text:
            score = parse_judge_score(response_text)
            if score is not None:
                scores[key] = score

    return scores


async def main_async():
    parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts Parquet file")
    parser.add_argument("--roles_dir", type=str, required=True, help="Directory containing role JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for score JSON files")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini", help="Judge model to use")
    parser.add_argument("--max_tokens", type=int, default=10, help="Max tokens for judge response")
    parser.add_argument("--batch_size", type=int, default=50, help="Concurrent batch size")
    parser.add_argument("--requests_per_second", type=int, default=100, help="Rate limit")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--skip_existing", action="store_true", help="Skip roles with existing scores")
    parser.add_argument("--skip_default", action="store_true", default=True, help="Skip default role (no judging needed)")
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollouts
    print(f"Loading rollouts from {args.rollouts}")
    df = pd.read_parquet(args.rollouts)
    print(f"Loaded {len(df)} rollouts")

    # Get roles to process
    roles_dir = Path(args.roles_dir)
    if args.roles:
        roles = args.roles
    else:
        roles = df["role"].unique().tolist()

    # Skip default role (no eval_prompt)
    if args.skip_default:
        roles = [r for r in roles if r != "default"]

    print(f"Processing {len(roles)} roles")

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(args.requests_per_second)

    # Process each role
    for role in tqdm(roles, desc="Scoring roles"):
        output_file = output_dir / f"{role}.json"

        # Skip if exists
        if args.skip_existing and output_file.exists():
            continue

        # Get role data
        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            print(f"Warning: Role file not found: {role_file}")
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            print(f"Warning: No eval_prompt for role {role}, skipping")
            continue

        # Get role responses
        role_df = df[df["role"] == role]
        if role_df.empty:
            continue

        # Score responses
        try:
            scores = await process_role(
                role=role,
                role_df=role_df,
                eval_prompt_template=eval_prompt_template,
                client=client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )

            # Save scores
            with open(output_file, 'w') as f:
                json.dump(scores, f, indent=2)

            print(f"Saved {len(scores)} scores for {role}")

        except Exception as e:
            print(f"Error processing {role}: {e}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
