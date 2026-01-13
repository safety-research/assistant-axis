#!/usr/bin/env python3
"""
Extract activations from rollouts.

This script loads rollouts from Parquet and extracts mean response activations
for each conversation, saving them as .pt files per role.

Usage:
    uv run scripts/2_activations.py \
        --model google/gemma-2-27b-it \
        --rollouts outputs/gemma-2-27b/rollouts.parquet \
        --output_dir outputs/gemma-2-27b/activations
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.activations import get_model_layers, get_response_token_indices


def main():
    parser = argparse.ArgumentParser(description="Extract activations from rollouts")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts Parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--layers", type=str, default="all", help="Layers to extract (all or comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip roles with existing output")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
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
    print(f"Model has {n_layers} layers")

    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Extracting {len(layers)} layers")

    # Load rollouts
    print(f"Loading rollouts from {args.rollouts}")
    df = pd.read_parquet(args.rollouts)
    print(f"Loaded {len(df)} rollouts")

    # Group by role
    roles = df["role"].unique()
    print(f"Processing {len(roles)} roles")

    for role in tqdm(roles, desc="Processing roles"):
        output_file = output_dir / f"{role}.pt"

        # Skip if exists
        if args.skip_existing and output_file.exists():
            continue

        role_df = df[df["role"] == role]
        activations_dict = {}

        for _, row in tqdm(role_df.iterrows(), total=len(role_df), desc=f"{role}", leave=False):
            prompt_idx = row["prompt_idx"]
            question_idx = row["question_idx"]
            system_prompt = row["system_prompt"]
            user_prompt = row["user_prompt"]
            response = row["response"]

            # Build conversation
            conversation = []
            if system_prompt:
                # Check if model supports system prompts
                if "gemma-2" not in args.model.lower():
                    conversation.append({"role": "system", "content": system_prompt})
                    conversation.append({"role": "user", "content": user_prompt})
                else:
                    conversation.append({"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"})
            else:
                conversation.append({"role": "user", "content": user_prompt})

            conversation.append({"role": "assistant", "content": response})

            # Get response token indices
            try:
                response_indices = get_response_token_indices(tokenizer, conversation)
            except Exception as e:
                print(f"Warning: Could not get response indices for {role} p{prompt_idx} q{question_idx}: {e}")
                continue

            if not response_indices:
                continue

            # Tokenize
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length
            )

            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Set up hooks
            layer_outputs = {}

            def make_hook(layer_idx):
                def hook(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    layer_outputs[layer_idx] = act
                return hook

            handles = []
            for layer_idx in layers:
                handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                handles.append(handle)

            # Forward pass
            try:
                with torch.no_grad():
                    model(input_ids, attention_mask=attention_mask)
            finally:
                for handle in handles:
                    handle.remove()

            # Extract mean response activations
            seq_len = input_ids.shape[1]
            valid_indices = [i for i in response_indices if i < seq_len]

            if not valid_indices:
                continue

            layer_acts = []
            for layer_idx in sorted(layers):
                acts = layer_outputs[layer_idx][0]  # Remove batch dim
                response_acts = acts[valid_indices]  # (n_tokens, hidden_dim)
                mean_act = response_acts.mean(dim=0).cpu()  # (hidden_dim,)
                layer_acts.append(mean_act)

            activation_tensor = torch.stack(layer_acts)  # (n_layers, hidden_dim)
            key = f"q{question_idx}_p{prompt_idx}"
            activations_dict[key] = activation_tensor

            # Cleanup
            del layer_outputs
            torch.cuda.empty_cache()

        # Save
        if activations_dict:
            torch.save(activations_dict, output_file)
            print(f"Saved {len(activations_dict)} activations for {role}")


if __name__ == "__main__":
    main()
