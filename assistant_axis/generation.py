"""
Response generation utilities for transformer models.

This module provides functions for generating model responses using
HuggingFace transformers, with support for chat formatting.

Example:
    from assistant_axis import load_model
    from assistant_axis.generation import generate_responses

    model, tokenizer = load_model("google/gemma-2-27b-it")
    conversations = [[{"role": "user", "content": "Hello!"}]]
    responses = generate_responses(model, tokenizer, conversations)
"""

import torch
from typing import List, Dict, Optional, Union
from tqdm import tqdm


def supports_system_prompt(model_name: str) -> bool:
    """
    Check if a model supports system prompts in chat templates.

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model supports system prompts, False otherwise
    """
    # Gemma 2 models don't support system prompts properly
    model_lower = model_name.lower()
    if "gemma-2" in model_lower:
        return False
    return True


def format_conversation(
    instruction: Optional[str],
    question: str,
    model_name: str
) -> List[Dict[str, str]]:
    """
    Format a conversation for model input.

    Args:
        instruction: Optional system instruction
        question: User question
        model_name: Model name (to determine formatting)

    Returns:
        List of message dicts for the conversation
    """
    if supports_system_prompt(model_name):
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    else:
        # For Gemma: concatenate instruction and question
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]


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
    Generate a single response for a conversation.

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
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
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
    batch_size: int = 1,
    show_progress: bool = True,
) -> List[str]:
    """
    Generate responses for multiple conversations.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversations: List of conversations (each is a list of message dicts)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample
        batch_size: Batch size for generation (currently only 1 supported)
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


def generate_with_conversation(
    model,
    tokenizer,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple:
    """
    Generate a response and return both the response and updated conversation.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversation: Current conversation history
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Tuple of (response_text, updated_conversation)
    """
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        conversation=conversation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    updated_conversation = conversation + [{"role": "assistant", "content": response}]

    return response, updated_conversation
