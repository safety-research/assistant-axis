"""
Assistant Axis - Tools for computing and steering with the assistant axis.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Example:
    from assistant_axis import get_config, load_axis, ActivationSteering
    from assistant_axis.internals import ProbingModel

    # Load model and axis
    pm = ProbingModel("google/gemma-2-27b-it")
    axis = load_axis("outputs/axis.pt")
    config = get_config("google/gemma-2-27b-it")

    # Steer model outputs
    with ActivationSteering(pm.model, steering_vectors=[axis[config["target_layer"]]],
                           coefficients=[1.0], layer_indices=[config["target_layer"]]):
        output = pm.model.generate(...)
"""

from .models import get_config, MODEL_CONFIGS
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
from .generation import (
    generate_response,
    format_conversation,
    VLLMGenerator,
    RoleResponseGenerator,
)
from .steering import (
    ActivationSteering,
    create_feature_ablation_steerer,
    create_multi_feature_steerer,
    create_mean_ablation_steerer,
    load_capping_config,
    build_capping_steerer,
)
from .pca import (
    compute_pca,
    plot_variance_explained,
    MeanScaler,
    L2MeanScaler,
)

__all__ = [
    # Models
    "get_config",
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
    # Generation
    "generate_response",
    "format_conversation",
    "VLLMGenerator",
    "RoleResponseGenerator",
    # Steering
    "ActivationSteering",
    "create_feature_ablation_steerer",
    "create_multi_feature_steerer",
    "create_mean_ablation_steerer",
    "load_capping_config",
    "build_capping_steerer",
    # PCA
    "compute_pca",
    "plot_variance_explained",
    "MeanScaler",
    "L2MeanScaler",
]
