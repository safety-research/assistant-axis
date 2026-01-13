"""
Activation extraction utilities for transformer models.

This module provides functions for extracting activations from model outputs,
particularly for analyzing response tokens in conversations.

Example:
    from assistant_axis import load_model
    from assistant_axis.activations import extract_response_activations

    model, tokenizer = load_model("google/gemma-2-27b-it")
    conversations = [[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]]
    activations = extract_response_activations(model, tokenizer, conversations)
"""

import gc
import torch
from typing import List, Dict, Optional, Union
from tqdm import tqdm


def get_response_token_indices(
    tokenizer,
    conversation: List[Dict[str, str]],
) -> List[int]:
    """
    Get the token indices corresponding to the assistant's response.

    Args:
        tokenizer: HuggingFace tokenizer
        conversation: Conversation with assistant response

    Returns:
        List of token indices for the assistant response
    """
    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Tokenize conversation without assistant response
    non_assistant = [m for m in conversation if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        non_assistant, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Response tokens are those after the prompt
    response_start = len(prompt_tokens)
    response_indices = list(range(response_start, len(full_tokens)))

    return response_indices


def get_model_layers(model) -> list:
    """
    Get the list of transformer layers from a model.

    Args:
        model: HuggingFace model

    Returns:
        List of layer modules
    """
    possible_attrs = [
        "model.layers",           # Llama/Mistral/Gemma 2/Qwen
        "transformer.h",          # GPT-2/Neo
        "language_model.layers",  # Gemma 3
        "encoder.layer",          # BERT
        "gpt_neox.layers",        # GPT-NeoX
    ]

    for attr in possible_attrs:
        parts = attr.split(".")
        obj = model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                break
        else:
            return list(obj)

    raise ValueError("Could not find transformer layers in model")


def extract_response_activations(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    layers: Optional[List[int]] = None,
    batch_size: int = 1,
    max_length: int = 2048,
    show_progress: bool = True,
) -> List[torch.Tensor]:
    """
    Extract mean activations from assistant response tokens.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversations: List of conversations (each with assistant response)
        layers: List of layer indices to extract (None for all)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        List of activation tensors, each of shape (n_layers, hidden_dim)
    """
    model_layers = get_model_layers(model)
    n_total_layers = len(model_layers)

    if layers is None:
        layers = list(range(n_total_layers))

    all_activations = []
    iterator = tqdm(conversations, desc="Extracting activations") if show_progress else conversations

    for conversation in iterator:
        # Get response token indices
        response_indices = get_response_token_indices(tokenizer, conversation)

        if not response_indices:
            all_activations.append(None)
            continue

        # Tokenize conversation
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )

        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Set up hooks to capture activations
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

        # Extract and average response activations
        conv_activations = []
        seq_len = input_ids.shape[1]
        valid_indices = [i for i in response_indices if i < seq_len]

        if not valid_indices:
            all_activations.append(None)
            continue

        for layer_idx in sorted(layers):
            layer_acts = layer_outputs[layer_idx][0]  # Remove batch dim
            response_acts = layer_acts[valid_indices]  # (n_response_tokens, hidden_dim)
            mean_act = response_acts.mean(dim=0).cpu()  # (hidden_dim,)
            conv_activations.append(mean_act)

        # Stack into (n_layers, hidden_dim)
        activation_tensor = torch.stack(conv_activations)
        all_activations.append(activation_tensor)

        # Cleanup
        del layer_outputs, input_ids, attention_mask
        torch.cuda.empty_cache()

    return all_activations


def extract_last_token_activations(
    model,
    tokenizer,
    conversations: List[List[Dict[str, str]]],
    layers: Optional[List[int]] = None,
    batch_size: int = 1,
    max_length: int = 2048,
    show_progress: bool = True,
) -> List[torch.Tensor]:
    """
    Extract activations from the last token position (pre-response).

    This is useful for analyzing the model's state before generating a response.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversations: List of conversations (without assistant response, or will be truncated)
        layers: List of layer indices to extract (None for all)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        List of activation tensors, each of shape (n_layers, hidden_dim)
    """
    model_layers = get_model_layers(model)
    n_total_layers = len(model_layers)

    if layers is None:
        layers = list(range(n_total_layers))

    all_activations = []
    iterator = tqdm(conversations, desc="Extracting activations") if show_progress else conversations

    for conversation in iterator:
        # Truncate to non-assistant messages
        truncated = [m for m in conversation if m["role"] != "assistant"]

        # Tokenize with generation prompt
        text = tokenizer.apply_chat_template(
            truncated, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
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

        # Extract last token activation
        last_idx = attention_mask.sum().item() - 1
        conv_activations = []

        for layer_idx in sorted(layers):
            layer_acts = layer_outputs[layer_idx][0]  # Remove batch dim
            last_act = layer_acts[last_idx].cpu()  # (hidden_dim,)
            conv_activations.append(last_act)

        activation_tensor = torch.stack(conv_activations)
        all_activations.append(activation_tensor)

        # Cleanup
        del layer_outputs, input_ids, attention_mask
        torch.cuda.empty_cache()

    return all_activations


def project_onto_axis(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
    """
    Project activations onto an axis at a specific layer.

    Args:
        activations: Activation tensor of shape (n_layers, hidden_dim)
        axis: Axis tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use for projection
        normalize: Whether to normalize the axis before projection

    Returns:
        Projection value (scalar)
    """
    act = activations[layer].float()
    ax = axis[layer].float()

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return float(act @ ax)
