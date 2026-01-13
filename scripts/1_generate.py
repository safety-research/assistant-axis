#!/usr/bin/env python3
"""
Generate model responses for all roles using vLLM batch inference.

This script loads role files and generates model responses for each role
using both the role-specific system prompts and default neutral prompts.

Usage:
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --roles_dir data/prompts/roles \
        --questions_file data/prompts/questions.jsonl \
        --output_dir outputs/gemma-2-27b/responses \
        --question_count 240
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.models import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_questions(questions_file: str) -> List[str]:
    """Load questions from JSONL file."""
    questions = []
    with jsonlines.open(questions_file, 'r') as reader:
        for entry in reader:
            questions.append(entry['question'])
    return questions


def load_role(role_file: str) -> dict:
    """Load a role JSON file."""
    with open(role_file, 'r') as f:
        return json.load(f)


def format_system_prompt(prompt: str, short_name: str) -> str:
    """Format system prompt, replacing {model_name} placeholder."""
    return prompt.replace("{model_name}", short_name)


def supports_system_prompt(model_name: str) -> bool:
    """Check if model supports system prompts."""
    model_lower = model_name.lower()
    if "gemma-2" in model_lower:
        return False
    return True


def format_conversation(instruction: Optional[str], question: str, model_name: str) -> List[Dict[str, str]]:
    """Format conversation for model input."""
    if supports_system_prompt(model_name):
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    else:
        # Gemma: concatenate instruction and question
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]


class RoleResponseGenerator:
    """Generator for role-based model responses using vLLM batch inference."""

    def __init__(
        self,
        model_name: str,
        roles_dir: str,
        output_dir: str,
        questions_file: str,
        max_model_len: int = 2048,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 240,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        prompt_indices: Optional[List[int]] = None,
    ):
        self.model_name = model_name
        self.roles_dir = Path(roles_dir)
        self.output_dir = Path(output_dir)
        self.questions_file = questions_file
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.question_count = question_count
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))

        # Get model config
        config = get_config(model_name)
        self.short_name = config["short_name"]

        self.llm = None
        self.sampling_params = None
        self.questions = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RoleResponseGenerator with model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_model(self):
        """Load the vLLM model."""
        if self.llm is not None:
            return

        logger.info(f"Loading vLLM model: {self.model_name}")

        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        logger.info("Model loaded successfully")

    def load_questions(self):
        """Load questions from file."""
        if self.questions is not None:
            return

        logger.info(f"Loading questions from {self.questions_file}")
        self.questions = load_questions(self.questions_file)[:self.question_count]
        logger.info(f"Loaded {len(self.questions)} questions")

    def load_role_files(self) -> Dict[str, dict]:
        """Load all role JSON files."""
        role_files = {}
        for file_path in sorted(self.roles_dir.glob("*.json")):
            role_name = file_path.stem
            try:
                role_data = load_role(file_path)
                if 'instruction' not in role_data:
                    logger.warning(f"Skipping {role_name}: missing 'instruction' field")
                    continue
                role_files[role_name] = role_data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        return role_files

    def generate_role_responses(self, role_name: str, role_data: dict) -> List[dict]:
        """Generate responses for a single role."""
        instructions = role_data['instruction']
        questions = self.questions

        logger.info(f"Processing role '{role_name}' with {len(questions)} questions")

        # Get positive instructions
        pos_instructions = [inst.get('pos', '') for inst in instructions]

        # Prepare all conversations for batch inference
        all_conversations = []
        all_metadata = []

        for prompt_idx in self.prompt_indices:
            if prompt_idx >= len(pos_instructions):
                continue

            instruction = pos_instructions[prompt_idx]
            instruction = format_system_prompt(instruction, self.short_name)

            for q_idx, question in enumerate(questions):
                conversation = format_conversation(instruction, question, self.model_name)
                all_conversations.append(conversation)
                all_metadata.append({
                    "system_prompt": instruction,
                    "label": "pos",
                    "prompt_index": prompt_idx,
                    "question_index": q_idx,
                    "question": question
                })

        if not all_conversations:
            return []

        # Format prompts for vLLM
        tokenizer = self.llm.get_tokenizer()
        prompts = []
        for conv in all_conversations:
            prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Batch inference
        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Combine responses with metadata
        results = []
        for i, (metadata, output) in enumerate(zip(all_metadata, outputs)):
            response_text = output.outputs[0].text
            result = {
                "system_prompt": metadata["system_prompt"],
                "label": metadata["label"],
                "prompt_index": metadata["prompt_index"],
                "question_index": metadata["question_index"],
                "question": metadata["question"],
                "conversation": all_conversations[i] + [{"role": "assistant", "content": response_text}]
            }
            results.append(result)

        return results

    def save_role_responses(self, role_name: str, responses: List[dict]):
        """Save role responses to JSONL file."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        with jsonlines.open(output_file, mode='w') as writer:
            for response in responses:
                writer.write(response)
        logger.info(f"Saved {len(responses)} responses to {output_file}")

    def should_skip_role(self, role_name: str) -> bool:
        """Check if role output already exists."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        return output_file.exists()

    def process_all_roles(self, skip_existing: bool = True, roles: Optional[List[str]] = None):
        """Process all roles and generate responses."""
        self.load_model()
        self.load_questions()

        role_files = self.load_role_files()
        logger.info(f"Found {len(role_files)} role files")

        # Filter roles if specified
        if roles:
            role_files = {k: v for k, v in role_files.items() if k in roles}

        # Filter existing
        if skip_existing:
            role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}

        logger.info(f"Processing {len(role_files)} roles")

        for role_name, role_data in tqdm(role_files.items(), desc="Processing roles"):
            try:
                responses = self.generate_role_responses(role_name, role_data)
                if responses:
                    self.save_role_responses(role_name, responses)
            except Exception as e:
                logger.error(f"Error processing {role_name}: {e}")


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
