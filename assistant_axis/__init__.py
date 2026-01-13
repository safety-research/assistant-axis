"""
Assistant Axis - Tools for computing and steering with the assistant axis.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Example:
    from assistant_axis import load_model, load_axis, ActivationSteering

    # Load model and axis
    model, tokenizer = load_model("google/gemma-2-27b-it")
    axis = load_axis("outputs/axis.pt")

    # Steer model outputs
    with ActivationSteering(model, steering_vectors=[axis[22]],
                           coefficients=[1.0], layer_indices=[22]):
        output = model.generate(...)
"""

from .models import load_model, get_config, get_short_name, MODEL_CONFIGS
from .axis import (
    compute_axis,
    load_axis,
    save_axis,
    project,
    project_batch,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
    aggregate_role_vectors,
)
from .activations import (
    extract_response_activations,
    tokenize_conversation,
    get_response_token_indices,
    project_onto_axis,
)
from .generation import (
    generate_response,
    generate_responses,
    format_conversation,
    supports_system_prompt,
    VLLMGenerator,
    RoleResponseGenerator,
)
from .steering import (
    ActivationSteering,
    create_feature_ablation_steerer,
    create_multi_feature_steerer,
    create_mean_ablation_steerer,
)
from .pca import (
    compute_pca,
    plot_variance_explained,
    MeanScaler,
    L2MeanScaler,
)

__all__ = [
    # Models
    "load_model",
    "get_config",
    "get_short_name",
    "MODEL_CONFIGS",
    # Axis
    "compute_axis",
    "load_axis",
    "save_axis",
    "project",
    "project_batch",
    "cosine_similarity_per_layer",
    "axis_norm_per_layer",
    "aggregate_role_vectors",
    # Activations
    "extract_response_activations",
    "tokenize_conversation",
    "get_response_token_indices",
    "project_onto_axis",
    # Generation
    "generate_response",
    "generate_responses",
    "format_conversation",
    "supports_system_prompt",
    "VLLMGenerator",
    "RoleResponseGenerator",
    # Steering
    "ActivationSteering",
    "create_feature_ablation_steerer",
    "create_multi_feature_steerer",
    "create_mean_ablation_steerer",
    # PCA
    "compute_pca",
    "plot_variance_explained",
    "MeanScaler",
    "L2MeanScaler",
]
