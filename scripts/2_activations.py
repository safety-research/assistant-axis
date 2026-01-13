#!/usr/bin/env python3
"""
Extract activations from response JSONL files.

This script loads responses from per-role JSONL files and extracts mean response
activations for each conversation, saving them as .pt files per role.

Usage:
    uv run scripts/2_activations.py \
        --model google/gemma-2-27b-it \
        --responses_dir outputs/gemma-2-27b/responses \
        --output_dir outputs/gemma-2-27b/activations
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.activations import get_model_layers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_response_indices(tokenizer, conversation: List[Dict[str, str]], model_name: str = None) -> List[int]:
    """Get token indices for assistant response tokens."""
    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Tokenize without assistant
    non_assistant = [m for m in conversation if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        non_assistant, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_start = len(prompt_tokens)
    return list(range(response_start, len(full_tokens)))


def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def extract_activations_batch(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    batch_size: int = 16,
    max_length: int = 2048,
    model_name: str = None,
) -> List[Optional[torch.Tensor]]:
    """Extract mean response activations for a batch of conversations."""
    model_layers = get_model_layers(model)
    all_activations = []
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        # Format prompts
        formatted_prompts = []
        for conv in batch_conversations:
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            formatted_prompts.append(prompt)

        # Tokenize batch
        batch_tokens = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length
        )

        device = next(model.parameters()).device
        input_ids = batch_tokens["input_ids"].to(device)
        attention_mask = batch_tokens["attention_mask"].to(device)

        # Get response indices for each conversation
        batch_response_indices = []
        for conv in batch_conversations:
            response_indices = get_response_indices(tokenizer, conv, model_name)
            batch_response_indices.append(response_indices)

        # Set up hooks
        layer_outputs = {}

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                layer_outputs[layer_idx] = act
            return hook_fn

        handles = []
        for layer_idx in layers:
            handle = model_layers[layer_idx].register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        # Forward pass
        try:
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        # Extract activations for each conversation
        for i, response_indices in enumerate(batch_response_indices):
            if not response_indices:
                all_activations.append(None)
                continue

            max_seq_len = layer_outputs[layers[0]].size(1)
            valid_indices = [idx for idx in response_indices if 0 <= idx < max_seq_len]

            if not valid_indices:
                all_activations.append(None)
                continue

            conv_activations = []
            for layer_idx in sorted(layers):
                layer_acts = layer_outputs[layer_idx][i]  # (seq_len, hidden_size)
                response_acts = layer_acts[valid_indices]  # (n_response_tokens, hidden_size)
                mean_act = response_acts.mean(dim=0).cpu()  # (hidden_size,)
                conv_activations.append(mean_act)

            conv_activations = torch.stack(conv_activations)  # (n_layers, hidden_size)
            all_activations.append(conv_activations)

        # Cleanup
        for layer_idx in list(layer_outputs.keys()):
            del layer_outputs[layer_idx]
        del input_ids, attention_mask

        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def main():
    parser = argparse.ArgumentParser(description="Extract activations from responses")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--layers", type=str, default="all", help="Layers to extract (all or comma-separated)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip roles with existing output")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Determine layers
    model_layers = get_model_layers(model)
    n_layers = len(model_layers)
    logger.info(f"Model has {n_layers} layers")

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger.info(f"Extracting {len(layers)} layers")

    # Get response files
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    # Filter roles if specified
    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    for response_file in tqdm(response_files, desc="Processing roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.pt"

        # Skip if exists
        if args.skip_existing and output_file.exists():
            continue

        # Load responses
        responses = load_responses(response_file)
        if not responses:
            continue

        # Extract conversations and metadata
        conversations = []
        metadata = []
        for resp in responses:
            conversations.append(resp["conversation"])
            metadata.append({
                "prompt_index": resp["prompt_index"],
                "question_index": resp["question_index"],
                "label": resp["label"],
            })

        logger.info(f"Processing {role}: {len(conversations)} conversations")

        # Extract activations
        activations_list = extract_activations_batch(
            model=model,
            tokenizer=tokenizer,
            conversations=conversations,
            layers=layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_name=args.model,
        )

        # Build activation dict
        activations_dict = {}
        for i, (act, meta) in enumerate(zip(activations_list, metadata)):
            if act is not None:
                key = f"q{meta['question_index']}_p{meta['prompt_index']}"
                activations_dict[key] = act

        # Save
        if activations_dict:
            torch.save(activations_dict, output_file)
            logger.info(f"Saved {len(activations_dict)} activations for {role}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Done!")


if __name__ == "__main__":
    main()
