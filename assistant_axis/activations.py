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

    Tries multiple approaches in order:
    1. return_assistant_tokens_mask (if template supports {% generation %})
    2. Model-specific special token detection (Qwen, Llama)
    3. Character offset mapping fallback

    Args:
        tokenizer: HuggingFace tokenizer
        conversation: Conversation with assistant response

    Returns:
        Dict with 'input_ids' and 'response_indices'
    """
    # Try return_assistant_tokens_mask first (canonical approach)
    try:
        result = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        assistant_mask = result["assistant_masks"]
        response_indices = [i for i, is_asst in enumerate(assistant_mask) if is_asst]

        if response_indices:
            return {
                "input_ids": result["input_ids"],
                "response_indices": response_indices,
            }
    except Exception:
        pass

    # Fall back to model-specific approaches
    model_name = getattr(tokenizer, "name_or_path", "").lower()

    if "qwen" in model_name:
        return _tokenize_qwen(tokenizer, conversation)
    else:
        return _tokenize_with_offset_mapping(tokenizer, conversation)


def _tokenize_qwen(
    tokenizer,
    conversation: List[Dict[str, str]],
) -> Dict:
    """
    Qwen-specific tokenization using special token boundaries.

    Finds assistant responses by locating <|im_start|>assistant...<|im_end|> patterns.
    """
    # Get full tokenized conversation
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer(full_text, add_special_tokens=False)
    token_ids = full_tokens["input_ids"]

    # Get special token IDs
    try:
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    except (KeyError, ValueError):
        # Fallback if special tokens not found
        return _tokenize_with_offset_mapping(tokenizer, conversation)

    # Find assistant response sections
    response_indices = []
    i = 0
    while i < len(token_ids):
        # Look for <|im_start|>assistant pattern
        if (i + 1 < len(token_ids) and
            token_ids[i] == im_start_id and
            token_ids[i + 1] == assistant_token_id):

            # Found start of assistant response, skip header tokens
            # Pattern: <|im_start|>assistant\n
            response_start = i + 2

            # Skip newline token if present
            if response_start < len(token_ids):
                newline_text = tokenizer.decode([token_ids[response_start]])
                if newline_text.strip() == "":
                    response_start += 1

            # Find the corresponding <|im_end|>
            response_end = None
            for j in range(response_start, len(token_ids)):
                if token_ids[j] == im_end_id:
                    response_end = j
                    break

            if response_end is not None:
                response_indices.extend(range(response_start, response_end))
                i = response_end + 1
            else:
                i += 1
        else:
            i += 1

    return {
        "input_ids": token_ids,
        "response_indices": response_indices,
    }


def _tokenize_with_offset_mapping(
    tokenizer,
    conversation: List[Dict[str, str]],
) -> Dict:
    """
    Tokenization using character offset mapping.

    Processes each assistant turn incrementally to find content boundaries.
    """
    # Get full tokenized conversation
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    response_indices = []

    # Process each assistant turn
    for i, turn in enumerate(conversation):
        if turn["role"] != "assistant":
            continue

        content = turn["content"]
        if not content:
            continue

        # Get conversation up to and including this turn
        conv_including = conversation[:i + 1]
        text_including = tokenizer.apply_chat_template(
            conv_including, tokenize=False, add_generation_prompt=False
        )

        # Find where content appears in the formatted text for this turn
        # Search from the end of the previous content to avoid matching earlier occurrences
        content_start = text_including.rfind(content)
        if content_start == -1:
            continue

        content_end = content_start + len(content)

        # Map to token indices using offset mapping
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start < content_end and tok_end > content_start:
                if idx not in response_indices:
                    response_indices.append(idx)

    return {
        "input_ids": token_ids,
        "response_indices": sorted(response_indices),
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
