"""
Assistant axis computation and projection utilities.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Formula:
    axis = mean(default_activations) - mean(pos_3_activations)

Where:
    - default_activations: activations from neutral system prompts
    - pos_3_activations: activations from responses fully playing a role (score=3)

The axis points FROM role-playing TOWARD default assistant behavior.

Example:
    from assistant_axis import load_axis, project

    axis = load_axis("outputs/gemma-2-27b/axis.pt")
    projection = project(activation, axis, layer=22)
"""

import torch
import numpy as np
from typing import Union, Optional


def compute_axis(
    role_activations: torch.Tensor,
    default_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the assistant axis from role and default activations.

    Formula: axis = mean(default) - mean(role)

    The axis points FROM role-playing TOWARD assistant behavior.

    Args:
        role_activations: Tensor of shape (n_role, n_layers, hidden_dim)
                         Activations from role-playing responses (score=3)
        default_activations: Tensor of shape (n_default, n_layers, hidden_dim)
                            Activations from default/neutral system prompts

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
    # Compute means
    role_mean = role_activations.mean(dim=0)       # (n_layers, hidden_dim)
    default_mean = default_activations.mean(dim=0)  # (n_layers, hidden_dim)

    # axis points from role toward default
    axis = default_mean - role_mean

    return axis


def project(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
    """
    Project activations onto the axis at a specific layer.

    Args:
        activations: Tensor of shape (n_layers, hidden_dim) or (hidden_dim,)
        axis: Tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use for projection
        normalize: Whether to normalize the axis before projection

    Returns:
        Projection value (scalar). Higher values indicate more "assistant-like",
        lower values indicate more "role-playing".
    """
    # Get layer activations
    if activations.ndim == 2:
        act = activations[layer].float()
    else:
        act = activations.float()

    ax = axis[layer].float()

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return float(act @ ax)


def project_batch(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project a batch of activations onto the axis.

    Args:
        activations: Tensor of shape (batch, n_layers, hidden_dim)
        axis: Tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use
        normalize: Whether to normalize the axis

    Returns:
        Projection values of shape (batch,)
    """
    # Get layer activations for all samples
    acts = activations[:, layer, :].float()  # (batch, hidden_dim)
    ax = axis[layer].float()  # (hidden_dim,)

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return acts @ ax  # (batch,)


def cosine_similarity_per_layer(
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> np.ndarray:
    """
    Compute cosine similarity between two vectors at each layer.

    Args:
        v1: Tensor of shape (n_layers, hidden_dim)
        v2: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of cosine similarities, one per layer
    """
    v1 = v1.float()
    v2 = v2.float()

    # Normalize both vectors
    v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
    v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)

    # Compute dot product per layer
    similarities = (v1_norm * v2_norm).sum(dim=1)

    return similarities.numpy()


def axis_norm_per_layer(axis: torch.Tensor) -> np.ndarray:
    """
    Compute the L2 norm of the axis at each layer.

    Args:
        axis: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of norms, one per layer
    """
    return axis.float().norm(dim=1).numpy()


def save_axis(
    axis: torch.Tensor,
    path: str,
    metadata: Optional[dict] = None,
):
    """
    Save axis to a .pt file.

    Args:
        axis: Axis tensor of shape (n_layers, hidden_dim)
        path: Path to save to
        metadata: Optional metadata dict to include
    """
    save_dict = {"axis": axis}
    if metadata:
        save_dict["metadata"] = metadata
    torch.save(save_dict, path)


def load_axis(path: str) -> torch.Tensor:
    """
    Load axis from a .pt file.

    Args:
        path: Path to load from

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both formats: dict with 'axis' key or raw tensor
    if isinstance(data, dict):
        if "axis" in data:
            return data["axis"]
        else:
            raise ValueError("Expected 'axis' key in saved dict")
    else:
        return data


def aggregate_role_vectors(
    vectors: dict,
    exclude_roles: Optional[list] = None,
) -> torch.Tensor:
    """
    Aggregate per-role vectors into a single mean vector.

    Args:
        vectors: Dict mapping role names to vectors (n_layers, hidden_dim)
        exclude_roles: List of role names to exclude (e.g., ["default"])

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
    exclude_roles = exclude_roles or []

    filtered = [v for name, v in vectors.items() if name not in exclude_roles]

    if not filtered:
        raise ValueError("No vectors remaining after exclusions")

    stacked = torch.stack(filtered)  # (n_roles, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
