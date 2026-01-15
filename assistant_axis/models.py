"""
Model configuration utilities.

This module provides configuration lookup for known models, including
project-specific values like target layers for axis computation.

For model loading, use ProbingModel from assistant_axis.internals instead.

Example:
    from assistant_axis import get_config
    from assistant_axis.internals import ProbingModel

    pm = ProbingModel("google/gemma-2-27b-it")
    config = get_config("google/gemma-2-27b-it")
    target_layer = config["target_layer"]
"""


MODEL_CONFIGS = {
    # Format: model_name -> {target_layer, total_layers, short_name, capping_config, capping_experiment}
    # target_layer is the recommended layer for axis computation (typically ~middle)
    # capping_config is the HuggingFace path to the capping config file
    # capping_experiment is the recommended experiment ID for activation capping
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
        "capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
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
