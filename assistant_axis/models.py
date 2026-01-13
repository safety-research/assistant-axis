"""
Model loading and configuration utilities.

This module provides functions for loading transformer models and
accessing model-specific configuration like target layers.

Example:
    from assistant_axis import load_model, get_config

    model, tokenizer = load_model("google/gemma-2-27b-it")
    config = get_config("google/gemma-2-27b-it")
    target_layer = config["target_layer"]
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CONFIGS = {
    # Format: model_name -> {target_layer, total_layers, short_name}
    # target_layer is the recommended layer for axis computation (typically ~middle)
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
    "google/gemma-2-9b-it": {
        "target_layer": 21,
        "total_layers": 42,
        "short_name": "Gemma",
    },
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "target_layer": 14,
        "total_layers": 28,
        "short_name": "Qwen",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "target_layer": 16,
        "total_layers": 32,
        "short_name": "Llama",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
    },
}


def get_config(model_name: str) -> dict:
    """
    Get configuration for a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dict with target_layer, total_layers, and short_name.
        If model is not in known configs, infers values from model architecture.
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()

    # Try to infer config from model
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        total_layers = config.num_hidden_layers
        target_layer = total_layers // 2  # Default to middle layer

        # Infer short name from model name
        model_lower = model_name.lower()
        if "gemma" in model_lower:
            short_name = "Gemma"
        elif "qwen" in model_lower:
            short_name = "Qwen"
        elif "llama" in model_lower:
            short_name = "Llama"
        elif "mistral" in model_lower:
            short_name = "Mistral"
        else:
            short_name = model_name.split("/")[-1].split("-")[0]

        return {
            "target_layer": target_layer,
            "total_layers": total_layers,
            "short_name": short_name,
        }
    except Exception as e:
        raise ValueError(f"Could not infer config for model {model_name}: {e}")


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs
) -> tuple:
    """
    Load a model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model name
        device_map: Device mapping strategy (default: "auto")
        torch_dtype: Model dtype (default: bfloat16)
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **kwargs
    )

    return model, tokenizer


def get_short_name(model_name: str) -> str:
    """Get short display name for a model."""
    return get_config(model_name)["short_name"]
