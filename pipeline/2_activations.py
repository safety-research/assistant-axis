#!/usr/bin/env python3
"""
Extract activations from response JSONL files.

This script loads responses from per-role JSONL files and extracts mean response
activations for each conversation, saving them as .pt files per role.

Supports automatic multi-worker parallelization when total GPUs > tensor_parallel_size.
Number of workers = total_gpus // tensor_parallel_size

Usage:
    uv run scripts/2_activations.py \
        --model google/gemma-2-27b-it \
        --responses_dir outputs/gemma-2-27b/responses \
        --output_dir outputs/gemma-2-27b/activations

    # With tensor parallelism (auto-parallelizes across workers)
    uv run scripts/2_activations.py \
        --model google/gemma-2-27b-it \
        --tensor_parallel_size 2 \
        ...
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses


def extract_activations_batch(
    pm: ProbingModel,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    batch_size: int = 16,
    max_length: int = 2048,
) -> List[Optional[torch.Tensor]]:
    """Extract mean response activations for a batch of conversations."""
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)
    span_mapper = SpanMapper(pm.tokenizer)

    all_activations = []
    num_conversations = len(conversations)

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_conversations = conversations[batch_start:batch_end]

        # Use ActivationExtractor.batch_conversations to get activations
        batch_activations, batch_metadata = extractor.batch_conversations(
            batch_conversations,
            layer=layers,
            max_length=max_length,
        )

        # batch_activations shape: (num_layers, batch_size, max_seq_len, hidden_size)

        # Build spans for this batch
        _, batch_spans, span_metadata = encoder.build_batch_turn_spans(batch_conversations)

        # Use SpanMapper to get per-turn mean activations
        # Returns list of tensors, each (num_turns, num_layers, hidden_size)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        # For each conversation, we want the assistant turn activations
        # In single-turn conversations: turn 0 = user, turn 1 = assistant
        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                all_activations.append(None)
                continue

            # conv_acts shape: (num_turns, num_layers, hidden_size)
            # For single-turn: (2, num_layers, hidden_size) - take turn 1 (assistant)
            # For multi-turn: take odd indices (assistant turns)
            if conv_acts.shape[0] >= 2:
                # Take the last assistant turn (index 1 for single-turn)
                assistant_act = conv_acts[1::2]  # All assistant turns
                if assistant_act.shape[0] > 0:
                    # Take mean across all assistant turns, transpose to (num_layers, hidden_size)
                    mean_act = assistant_act.mean(dim=0).cpu()  # (num_layers, hidden_size)
                    all_activations.append(mean_act)
                else:
                    all_activations.append(None)
            else:
                all_activations.append(None)

        # Cleanup
        del batch_activations
        if (batch_start // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    return all_activations


def process_role(pm: ProbingModel, role_file: Path, output_dir: Path, layers: List[int], batch_size: int, max_length: int) -> bool:
    """Process a single role file and save activations."""
    role = role_file.stem
    output_file = output_dir / f"{role}.pt"

    # Load responses
    responses = load_responses(role_file)
    if not responses:
        return False

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
        pm=pm,
        conversations=conversations,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Build activation dict
    activations_dict = {}
    for i, (act, meta) in enumerate(zip(activations_list, metadata)):
        if act is not None:
            key = f"{meta['label']}_p{meta['prompt_index']}_q{meta['question_index']}"
            activations_dict[key] = act

    # Save
    if activations_dict:
        torch.save(activations_dict, output_file)
        logger.info(f"Saved {len(activations_dict)} activations for {role}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return True


def process_roles_on_worker(worker_id: int, gpu_ids: List[int], role_files: List[Path], args):
    """Process a subset of roles on a worker."""
    # Set CUDA_VISIBLE_DEVICES for this worker's GPU subset
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    # Set up logging for this process
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    worker_logger.info(f"Starting Worker {worker_id} with GPUs {gpu_ids} and {len(role_files)} roles")

    output_dir = Path(args.output_dir)

    try:
        # Load model
        worker_logger.info(f"Loading model: {args.model}")
        pm = ProbingModel(args.model)

        # Determine layers
        n_layers = len(pm.get_layers())
        if args.layers == "all":
            layers = list(range(n_layers))
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]

        worker_logger.info(f"Extracting {len(layers)} layers")

        # Process assigned roles
        completed_count = 0
        failed_count = 0

        for role_file in tqdm(role_files, desc=f"Worker-{worker_id}", position=worker_id):
            try:
                success = process_role(pm, role_file, output_dir, layers, args.batch_size, args.max_length)
                if success:
                    completed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                worker_logger.error(f"Exception processing {role_file.stem}: {e}")

        worker_logger.info(f"Worker {worker_id} completed: {completed_count} successful, {failed_count} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")

    finally:
        worker_logger.info(f"Worker {worker_id} cleanup completed")


def run_multi_worker(args) -> int:
    """Run multi-worker processing."""
    # Get available GPUs
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)

    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1

    tensor_parallel_size = args.tensor_parallel_size

    if tensor_parallel_size > total_gpus:
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tensor_parallel_size

    if total_gpus % tensor_parallel_size != 0:
        logger.warning(f"GPUs ({total_gpus}) not evenly divisible by tensor_parallel_size ({tensor_parallel_size}). "
                      f"Using {num_workers} workers.")

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Number of workers: {num_workers}")

    # Get role files
    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_files = []
    for f in sorted(responses_dir.glob("*.jsonl")):
        # Filter by --roles if specified
        if args.roles and f.stem not in args.roles:
            continue
        # Skip existing
        output_file = output_dir / f"{f.stem}.pt"
        if output_file.exists():
            logger.info(f"Skipping {f.stem} (already exists)")
            continue
        role_files.append(f)

    if not role_files:
        logger.info("No roles to process")
        return 0

    logger.info(f"Processing {len(role_files)} roles across {num_workers} workers")

    # Partition GPUs
    gpu_chunks = []
    for i in range(num_workers):
        start = i * tensor_parallel_size
        end = start + tensor_parallel_size
        gpu_chunks.append(gpu_ids[start:end])

    # Distribute roles
    role_chunks = [[] for _ in range(num_workers)]
    for i, role_file in enumerate(role_files):
        role_chunks[i % num_workers].append(role_file)

    for i in range(num_workers):
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(role_chunks[i])} roles")

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Launch workers
    processes = []
    for worker_id in range(num_workers):
        if role_chunks[worker_id]:
            p = mp.Process(
                target=process_roles_on_worker,
                args=(worker_id, gpu_chunks[worker_id], role_chunks[worker_id], args)
            )
            p.start()
            processes.append(p)

    # Wait for completion
    logger.info(f"Launched {len(processes)} worker processes")
    for p in processes:
        p.join()

    logger.info("Multi-worker processing completed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Extract activations from responses")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--layers", type=str, default="all", help="Layers to extract (all or comma-separated)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="GPUs per model (auto-detect if None)")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    args = parser.parse_args()

    # Detect GPUs for multi-worker decision
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
        total_gpus = len(available_gpus)
    else:
        total_gpus = torch.cuda.device_count()

    # Determine tensor parallel size
    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus

    # Use multi-worker mode if we have more GPUs than tensor_parallel_size
    use_multi_worker = (
        total_gpus > 1 and
        tensor_parallel_size > 0 and
        total_gpus > tensor_parallel_size
    )

    if use_multi_worker:
        logger.info(f"Multi-worker mode: {total_gpus} GPUs with tensor_parallel_size={tensor_parallel_size}")
        logger.info(f"Number of workers: {total_gpus // tensor_parallel_size}")
        args.tensor_parallel_size = tensor_parallel_size
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        # Single-worker mode
        logger.info(f"Single-worker mode: Using {tensor_parallel_size} GPU(s)")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = Path(args.responses_dir)

        # Load model
        logger.info(f"Loading model: {args.model}")
        pm = ProbingModel(args.model)

        # Determine layers
        n_layers = len(pm.get_layers())
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

        # Filter out existing
        role_files = []
        for f in response_files:
            output_file = output_dir / f"{f.stem}.pt"
            if output_file.exists():
                logger.info(f"Skipping {f.stem} (already exists)")
                continue
            role_files.append(f)

        for role_file in tqdm(role_files, desc="Processing roles"):
            process_role(pm, role_file, output_dir, layers, args.batch_size, args.max_length)

    logger.info("Done!")


if __name__ == "__main__":
    main()
