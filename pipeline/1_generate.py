#!/usr/bin/env python3
"""
Generate model responses for all roles using vLLM batch inference.

This script loads role files and generates model responses for each role
using the role-specific system prompts. It can be restarted and won't overwrite existing roles.

Supports automatic multi-worker parallelization when total GPUs > tensor_parallel_size.
Number of workers = total_gpus // tensor_parallel_size

Usage:
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --roles_dir data/prompts/roles \
        --questions_file data/prompts/questions.jsonl \
        --output_dir outputs/gemma-2-27b/responses \
        --question_count 240

    # With explicit tensor parallelism (will auto-parallelize across workers)
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --tensor_parallel_size 2 \
        ...
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import RoleResponseGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_roles_on_worker(worker_id: int, gpu_ids: List[int], role_names: List[str], args):
    """Process a subset of roles on a worker with tensor parallelism support."""
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

    worker_logger.info(f"Starting processing on Worker {worker_id} with GPUs {gpu_ids} and {len(role_names)} roles")

    try:
        # Create generator for this worker
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
        )

        # Load model
        generator.generator.load()

        # Load role files and filter to assigned roles
        role_files = {}
        roles_dir = Path(args.roles_dir)
        for file_path in sorted(roles_dir.glob("*.json")):
            role_name = file_path.stem
            if role_name in role_names:
                try:
                    role_data = generator.load_role(file_path)
                    if 'instruction' in role_data:
                        role_files[role_name] = role_data
                except Exception as e:
                    worker_logger.error(f"Error loading {file_path}: {e}")

        # Process assigned roles
        completed_count = 0
        failed_count = 0

        from tqdm import tqdm
        for role_name, role_data in tqdm(role_files.items(), desc=f"Worker-{worker_id}", position=worker_id):
            try:
                responses = generator.generate_role_responses(role_name, role_data)
                if responses:
                    generator.save_responses(role_name, responses)
                    completed_count += 1
                else:
                    failed_count += 1
                    worker_logger.warning(f"No responses generated for role '{role_name}'")
            except Exception as e:
                failed_count += 1
                worker_logger.error(f"Exception processing role {role_name}: {e}")

        worker_logger.info(f"Worker {worker_id} completed: {completed_count} successful, {failed_count} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")

    finally:
        worker_logger.info(f"Worker {worker_id} cleanup completed")


def run_multi_worker(args) -> int:
    """Run multi-worker processing with tensor parallelism support."""
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
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) cannot be greater than available GPUs ({total_gpus})")
        return 1

    num_workers = total_gpus // tensor_parallel_size

    if total_gpus % tensor_parallel_size != 0:
        logger.warning(f"Total GPUs ({total_gpus}) not evenly divisible by tensor_parallel_size ({tensor_parallel_size}). "
                      f"Using {num_workers} workers, leaving {total_gpus % tensor_parallel_size} GPU(s) unused.")

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Number of workers: {num_workers}")

    # Get all role names
    roles_dir = Path(args.roles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_names = []
    for file_path in sorted(roles_dir.glob("*.json")):
        role_name = file_path.stem
        # Filter by --roles if specified
        if args.roles and role_name not in args.roles:
            continue
        # Skip existing
        output_file = output_dir / f"{role_name}.jsonl"
        if output_file.exists():
            logger.info(f"Skipping role '{role_name}' (already exists)")
            continue
        role_names.append(role_name)

    if not role_names:
        logger.info("No roles to process")
        return 0

    logger.info(f"Processing {len(role_names)} roles across {num_workers} workers")

    # Partition GPUs into chunks for each worker
    gpu_chunks = []
    for i in range(num_workers):
        start_gpu_idx = i * tensor_parallel_size
        end_gpu_idx = start_gpu_idx + tensor_parallel_size
        worker_gpus = gpu_ids[start_gpu_idx:end_gpu_idx]
        gpu_chunks.append(worker_gpus)

    # Distribute roles across workers
    roles_per_worker = len(role_names) // num_workers
    remainder = len(role_names) % num_workers

    role_chunks = []
    start_idx = 0

    for i in range(num_workers):
        chunk_size = roles_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunk = role_names[start_idx:end_idx]
        role_chunks.append(chunk)
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(chunk)} roles")
        start_idx = end_idx

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Launch worker processes
    processes = []
    for worker_id in range(num_workers):
        if role_chunks[worker_id]:
            p = mp.Process(
                target=process_roles_on_worker,
                args=(worker_id, gpu_chunks[worker_id], role_chunks[worker_id], args)
            )
            p.start()
            processes.append(p)

    # Wait for all processes
    logger.info(f"Launched {len(processes)} worker processes")
    for p in processes:
        p.join()

    logger.info("Multi-worker processing completed!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate role responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--roles_dir', type=str, default="../data/roles/instructions", help='Directory containing role JSON files')
    parser.add_argument('--questions_file', type=str, default="../data/extraction_questions.jsonl", help='Path to questions JSONL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSONL files')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum model context length')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization')
    parser.add_argument('--question_count', type=int, default=240, help='Number of questions per role')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--roles', nargs='+', help='Specific roles to process')

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
        # Ensure tensor_parallel_size is set for multi-worker
        args.tensor_parallel_size = tensor_parallel_size
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
    else:
        # Single-worker mode
        logger.info(f"Single-worker mode: Using {tensor_parallel_size} GPU(s)")

        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

        generator.process_all_roles(
            skip_existing=True,
            roles=args.roles
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
