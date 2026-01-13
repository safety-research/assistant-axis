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

import torch
from typing import List, Dict, Optional
from tqdm import tqdm


def tokenize_conversation(
    tokenizer,
    conversation: List[Dict[str, str]],
) -> Dict:
    """
    Tokenize a conversation and get response content token indices.

    Uses the tokenizer's built-in `return_assistant_tokens_mask` feature
    which parses the Jinja chat template to identify assistant content tokens
    (excludes special format tokens like <|assistant|>).

    Args:
        tokenizer: HuggingFace tokenizer
        conversation: Conversation with assistant response

    Returns:
        Dict with 'input_ids' and 'response_indices'
    """
    result = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )

    assistant_mask = result["assistant_mask"]
    response_indices = [i for i, is_asst in enumerate(assistant_mask) if is_asst]

    return {
        "input_ids": result["input_ids"],
        "response_indices": response_indices,
    }


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
    return tokenize_conversation(tokenizer, conversation)["response_indices"]


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
    max_length: int = 2048,
    show_progress: bool = True,
) -> List[torch.Tensor]:
    """
    Extract mean activations from assistant response content tokens.

    Only extracts activations for the actual response content, not
    special chat format tokens (like <|assistant|>, <|im_end|>, etc.).

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversations: List of conversations (each with assistant response)
        layers: List of layer indices to extract (None for all)
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
        # Tokenize and get response indices in one call
        tokenized = tokenize_conversation(tokenizer, conversation)
        response_indices = tokenized["response_indices"]
        token_ids = tokenized["input_ids"]

        if not response_indices:
            all_activations.append(None)
            continue

        # Truncate if needed
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            response_indices = [i for i in response_indices if i < max_length]

        device = next(model.parameters()).device
        input_ids = torch.tensor([token_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

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

        for layer_idx in sorted(layers):
            layer_acts = layer_outputs[layer_idx][0]  # Remove batch dim
            response_acts = layer_acts[response_indices]  # (n_response_tokens, hidden_dim)
            mean_act = response_acts.mean(dim=0).cpu()  # (hidden_dim,)
            conv_activations.append(mean_act)

        # Stack into (n_layers, hidden_dim)
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
