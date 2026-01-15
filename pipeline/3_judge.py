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
        --responses_dir outputs/gemma-2-27b/responses \
        --roles_dir data/prompts/roles \
        --output_dir outputs/gemma-2-27b/scores \
        --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score
import openai

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")


def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


async def process_role(
    role: str,
    responses: List[dict],
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
) -> dict:
    """Process a single role and return scores."""
    # Build prompts for each response
    prompts = []
    keys = []

    for resp in responses:
        prompt_idx = resp["prompt_index"]
        question_idx = resp["question_index"]
        question = resp["question"]
        label = resp["label"]

        # Get assistant response from conversation
        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        key = f"{label}_p{prompt_idx}_q{question_idx}"

        # Skip if already scored
        if key in existing_scores:
            continue

        # Fill in template
        judge_prompt = eval_prompt_template.format(
            question=question,
            answer=assistant_response
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {}

    # Call judge
    logger.info(f"Scoring {len(prompts)} new responses for {role}...")
    responses_text = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = {}
    for key, response_text in zip(keys, responses_text):
        if response_text:
            score = parse_judge_score(response_text)
            if score is not None:
                scores[key] = score

    return scores


async def main_async():
    parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--roles_dir", type=str, default="../data/roles/instructions", help="Directory containing role JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for score JSON files")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini", help="Judge model to use")
    parser.add_argument("--max_tokens", type=int, default=10, help="Max tokens for judge response")
    parser.add_argument("--batch_size", type=int, default=50, help="Concurrent batch size")
    parser.add_argument("--requests_per_second", type=int, default=100, help="Rate limit")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--dry_run", action="store_true", help="Preview what would be processed without making API calls")
    args = parser.parse_args()

    # Check for API key (not needed for dry run)
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)

    # Get response files
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    # Filter roles if specified
    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    logger.info(f"Processing {len(response_files)} roles")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        sample_shown = False

        for response_file in response_files:
            role = response_file.stem
            output_file = output_dir / f"{role}.json"

            # Load existing scores
            existing_scores = {}
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_scores = json.load(f)
                except Exception:
                    pass

            # Get role eval prompt
            role_file = roles_dir / f"{role}.json"
            if not role_file.exists():
                logger.info(f"  {role}: no role file, skipping")
                continue

            eval_prompt_template = load_role_eval_prompt(role_file)
            if not eval_prompt_template:
                logger.info(f"  {role}: no eval_prompt, skipping")
                continue

            # Load responses and count prompts to be scored
            responses = load_responses(response_file)
            prompts_for_role = 0
            sample_prompt = None

            for resp in responses:
                prompt_idx = resp["prompt_index"]
                question_idx = resp["question_index"]
                label = resp["label"]
                key = f"{label}_p{prompt_idx}_q{question_idx}"

                if key not in existing_scores:
                    prompts_for_role += 1
                    if sample_prompt is None:
                        # Build sample prompt
                        assistant_response = ""
                        for msg in resp["conversation"]:
                            if msg["role"] == "assistant":
                                assistant_response = msg["content"]
                                break
                        sample_prompt = eval_prompt_template.format(
                            question=resp["question"],
                            answer=assistant_response
                        )

            if prompts_for_role > 0:
                total_prompts += prompts_for_role
                logger.info(f"  {role}: {prompts_for_role} prompts")

                # Show one sample
                if not sample_shown and sample_prompt:
                    logger.info("\n" + "=" * 60)
                    logger.info("SAMPLE JUDGE PROMPT:")
                    logger.info("=" * 60)
                    logger.info(f"Model: {args.judge_model}")
                    logger.info(f"Max tokens: {args.max_tokens}")
                    logger.info("-" * 60)
                    logger.info(sample_prompt)
                    logger.info("=" * 60 + "\n")
                    sample_shown = True

        logger.info(f"\nTotal prompts to send: {total_prompts}")
        return

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(args.requests_per_second)

    # Track results
    successful = 0
    skipped = 0
    failed = 0
    errors = []

    # Process each role
    for response_file in tqdm(response_files, desc="Scoring roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.json"

        # Load existing scores
        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        # Get role eval prompt
        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            logger.info(f"Skipping {role}: no role file found")
            skipped += 1
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {role}: no eval_prompt in role file")
            skipped += 1
            continue

        # Load responses
        responses = load_responses(response_file)
        if not responses:
            errors.append(f"{role}: no responses found")
            failed += 1
            continue

        # Check if all responses are already scored
        all_scored = True
        for resp in responses:
            key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
            if key not in existing_scores:
                all_scored = False
                break

        if all_scored:
            logger.info(f"Skipping {role}: all {len(responses)} responses already scored")
            skipped += 1
            continue

        # Score responses
        try:
            new_scores = await process_role(
                role=role,
                responses=responses,
                eval_prompt_template=eval_prompt_template,
                client=client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                existing_scores=existing_scores,
            )

            # Merge scores
            all_scores = {**existing_scores, **new_scores}

            # Save scores
            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)

            logger.info(f"Saved {len(all_scores)} scores for {role} ({len(new_scores)} new)")
            successful += 1

        except Exception as e:
            errors.append(f"{role}: {e}")
            failed += 1

    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped:    {skipped}")
    logger.info(f"Failed:     {failed}")

    if errors:
        logger.info("\nErrors:")
        for error in errors[:10]:
            logger.info(f"  - {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
