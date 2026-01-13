"""
Response generation utilities for transformer models.

This module provides functions for generating model responses using both
HuggingFace transformers (for interactive use) and vLLM (for batch inference).

Example (HuggingFace - interactive):
    from assistant_axis import load_model
    from assistant_axis.generation import generate_response

    model, tokenizer = load_model("google/gemma-2-27b-it")
    response = generate_response(model, tokenizer, conversation)

Example (vLLM - batch inference):
    from assistant_axis.generation import VLLMGenerator

    generator = VLLMGenerator("google/gemma-2-27b-it")
    responses = generator.generate_batch(conversations)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def supports_system_prompt(tokenizer) -> bool:
    """
    Check if a tokenizer's chat template supports system prompts.

    Detects support by applying the template with a test system message
    and checking if it appears in the output.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        True if model supports system prompts, False otherwise
    """
    test_message = "__SYSTEM_TEST__"
    test_conversation = [
        {"role": "system", "content": test_message},
        {"role": "user", "content": "hello"},
    ]

    try:
        output = tokenizer.apply_chat_template(
            test_conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        return test_message in output
    except Exception:
        return False


def format_conversation(
    instruction: Optional[str],
    question: str,
    tokenizer,
) -> List[Dict[str, str]]:
    """
    Format a conversation for model input.

    Args:
        instruction: Optional system instruction
        question: User question
        tokenizer: HuggingFace tokenizer (to check system prompt support)

    Returns:
        List of message dicts for the conversation
    """
    if supports_system_prompt(tokenizer):
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    else:
        # Concatenate instruction with question for models without system support
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]


# =============================================================================
# HuggingFace Generation (for interactive use / notebooks)
# =============================================================================

def generate_response(
    model,
    tokenizer,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate a single response for a conversation using HuggingFace.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversation: List of message dicts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample (False = greedy)

    Returns:
        Generated response text
    """
    # Disable thinking for Qwen models (https://github.com/vllm-project/vllm/issues/18066)
    chat_template_kwargs = {}
    if hasattr(tokenizer, 'name_or_path') and "qwen" in tokenizer.name_or_path.lower():
        chat_template_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return response


def generate_responses(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    show_progress: bool = True,
) -> List[str]:
    """
    Generate responses for multiple conversations using HuggingFace (sequential).

    For batch inference, use VLLMGenerator instead.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversations: List of conversations
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample
        show_progress: Whether to show progress bar

    Returns:
        List of generated response texts
    """
    responses = []
    iterator = tqdm(conversations, desc="Generating") if show_progress else conversations

    for conversation in iterator:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            conversation=conversation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        responses.append(response)

    return responses


# =============================================================================
# vLLM Generation (for batch inference / pipeline)
# =============================================================================

class VLLMGenerator:
    """
    Generator for batch inference using vLLM.

    Example:
        generator = VLLMGenerator("google/gemma-2-27b-it")
        responses = generator.generate_batch(conversations)
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 2048,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ):
        """
        Initialize vLLM generator.

        Args:
            model_name: HuggingFace model name
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs (None for auto-detect)
            gpu_memory_utilization: GPU memory utilization
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.llm = None
        self.sampling_params = None

    def load(self):
        """Load the vLLM model."""
        if self.llm is not None:
            return

        from vllm import LLM, SamplingParams

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

    def generate_batch(
        self,
        conversations: List[List[Dict[str, str]]],
    ) -> List[str]:
        """
        Generate responses for a batch of conversations.

        Args:
            conversations: List of conversations (each is a list of message dicts)

        Returns:
            List of generated response texts
        """
        self.load()

        tokenizer = self.llm.get_tokenizer()

        # Disable thinking for Qwen models (https://github.com/vllm-project/vllm/issues/18066)
        chat_template_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_template_kwargs["enable_thinking"] = False

        prompts = []
        for conv in conversations:
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompts.append(prompt)

        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        responses = [output.outputs[0].text for output in outputs]
        return responses

    def generate_for_role(
        self,
        instructions: List[str],
        questions: List[str],
        prompt_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Generate responses for a role across all instruction variants and questions.

        Args:
            instructions: List of system prompt variants
            questions: List of questions
            prompt_indices: Which instruction indices to use (default: all)

        Returns:
            List of result dicts with conversation, prompt_index, question_index
        """
        self.load()
        tokenizer = self.llm.get_tokenizer()

        if prompt_indices is None:
            prompt_indices = list(range(len(instructions)))

        # Build all conversations
        all_conversations = []
        all_metadata = []

        for prompt_idx in prompt_indices:
            if prompt_idx >= len(instructions):
                continue

            instruction = instructions[prompt_idx]

            for q_idx, question in enumerate(questions):
                conversation = format_conversation(instruction, question, tokenizer)
                all_conversations.append(conversation)
                all_metadata.append({
                    "system_prompt": instruction,
                    "prompt_index": prompt_idx,
                    "question_index": q_idx,
                    "question": question,
                })

        if not all_conversations:
            return []

        # Generate
        responses = self.generate_batch(all_conversations)

        # Build results
        results = []
        for conv, meta, response in zip(all_conversations, all_metadata, responses):
            result = {
                "system_prompt": meta["system_prompt"],
                "prompt_index": meta["prompt_index"],
                "question_index": meta["question_index"],
                "question": meta["question"],
                "conversation": conv + [{"role": "assistant", "content": response}],
            }
            results.append(result)

        return results


class RoleResponseGenerator:
    """
    Generator for role-based model responses using vLLM batch inference.

    Processes role JSON files and generates responses for all roles.

    Example:
        generator = RoleResponseGenerator(
            model_name="google/gemma-2-27b-it",
            roles_dir="data/prompts/roles",
            output_dir="outputs/responses",
            questions_file="data/prompts/questions.jsonl"
        )
        generator.process_all_roles()
    """

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
        short_name: Optional[str] = None,
    ):
        """
        Initialize role response generator.

        Args:
            model_name: HuggingFace model name
            roles_dir: Directory containing role JSON files
            output_dir: Output directory for JSONL files
            questions_file: Path to questions JSONL file
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs
            gpu_memory_utilization: GPU memory utilization
            question_count: Number of questions per role
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
            prompt_indices: Which prompt indices to use (default: 0-4)
            short_name: Short model name for formatting (auto-detected if None)
        """
        self.model_name = model_name
        self.roles_dir = Path(roles_dir)
        self.output_dir = Path(output_dir)
        self.questions_file = questions_file
        self.question_count = question_count
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))

        # Get short name for {model_name} placeholder
        if short_name is None:
            from .models import get_config
            config = get_config(model_name)
            self.short_name = config["short_name"]
        else:
            self.short_name = short_name

        self.generator = VLLMGenerator(
            model_name=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self.questions = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RoleResponseGenerator with model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_questions(self) -> List[str]:
        """Load questions from JSONL file."""
        if self.questions is not None:
            return self.questions

        import jsonlines

        questions = []
        with jsonlines.open(self.questions_file, 'r') as reader:
            for entry in reader:
                questions.append(entry['question'])

        self.questions = questions[:self.question_count]
        logger.info(f"Loaded {len(self.questions)} questions")
        return self.questions

    def load_role(self, role_file: Path) -> dict:
        """Load a role JSON file."""
        with open(role_file, 'r') as f:
            return json.load(f)

    def format_instruction(self, instruction: str) -> str:
        """Format instruction, replacing {model_name} placeholder."""
        return instruction.replace("{model_name}", self.short_name)

    def generate_role_responses(self, role_name: str, role_data: dict) -> List[dict]:
        """Generate responses for a single role."""
        instructions = role_data.get('instruction', [])
        if not instructions:
            return []

        questions = self.load_questions()

        # Get and format instructions
        formatted_instructions = []
        for inst in instructions:
            raw = inst.get('pos', '')
            formatted_instructions.append(self.format_instruction(raw))

        logger.info(f"Processing role '{role_name}' with {len(questions)} questions")

        # Generate
        results = self.generator.generate_for_role(
            instructions=formatted_instructions,
            questions=questions,
            prompt_indices=self.prompt_indices,
        )

        # Add label
        for r in results:
            r["label"] = "pos"

        return results

    def save_responses(self, role_name: str, responses: List[dict]):
        """Save responses to JSONL file."""
        import jsonlines

        output_file = self.output_dir / f"{role_name}.jsonl"
        with jsonlines.open(output_file, mode='w') as writer:
            for response in responses:
                writer.write(response)
        logger.info(f"Saved {len(responses)} responses to {output_file}")

    def should_skip_role(self, role_name: str) -> bool:
        """Check if role output already exists."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        return output_file.exists()

    def process_all_roles(
        self,
        skip_existing: bool = True,
        roles: Optional[List[str]] = None,
    ):
        """
        Process all roles and generate responses.

        Args:
            skip_existing: Skip roles with existing output files
            roles: Specific role names to process (None for all)
        """
        # Load model
        self.generator.load()
        self.load_questions()

        # Get role files
        role_files = {}
        for file_path in sorted(self.roles_dir.glob("*.json")):
            role_name = file_path.stem
            try:
                role_data = self.load_role(file_path)
                if 'instruction' not in role_data:
                    logger.warning(f"Skipping {role_name}: missing 'instruction' field")
                    continue
                role_files[role_name] = role_data
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Found {len(role_files)} role files")

        # Filter
        if roles:
            role_files = {k: v for k, v in role_files.items() if k in roles}

        if skip_existing:
            role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}

        logger.info(f"Processing {len(role_files)} roles")

        # Process
        for role_name, role_data in tqdm(role_files.items(), desc="Processing roles"):
            try:
                responses = self.generate_role_responses(role_name, role_data)
                if responses:
                    self.save_responses(role_name, responses)
            except Exception as e:
                logger.error(f"Error processing {role_name}: {e}")
