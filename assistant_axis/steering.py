"""
Activation steering utilities for transformer models.

This module provides a context manager for intervening on model activations
during inference, supporting addition, ablation, and mean ablation operations.

Example:
    from assistant_axis import load_model, load_axis, ActivationSteering

    model, tokenizer = load_model("google/gemma-2-27b-it")
    axis = load_axis("outputs/gemma-2-27b/axis.pt")

    with ActivationSteering(model, steering_vectors=[axis[22]],
                           coefficients=[1.0], layer_indices=[22]):
        output = model.generate(...)
"""

import torch
from typing import Sequence, Union, Iterable, List


class ActivationSteering:
    """
    Context manager for activation steering supporting:
    - Multiple feature directions simultaneously
    - Both addition and ablation interventions
    - Multiple layers
    - Per-direction coefficients

    For ablation: projects out the direction, then adds back with coefficient
    For addition: standard activation steering (add coeff * direction)
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",          # GPT-2/Neo, Bloom, etc.
        "encoder.layer",          # BERT/RoBERTa
        "model.layers",           # Llama/Mistral/Gemma 2/Qwen
        "language_model.layers",  # Gemma 3 (vision-language models)
        "gpt_neox.layers",        # GPT-NeoX
        "block",                  # Flan-T5
    )

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]],
        *,
        coefficients: Union[float, List[float]] = 1.0,
        layer_indices: Union[int, List[int]] = -1,
        intervention_type: str = "addition",
        positions: str = "all",
        mean_activations: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]], None] = None,
        cap_thresholds: Union[float, List[float], None] = None,
        debug: bool = False,
    ):
        """
        Args:
            model: The transformer model to steer
            steering_vectors: Either a single vector or list of vectors to use for steering
            coefficients: Either a single coefficient or list of coefficients (one per vector)
            layer_indices: Either a single layer index or list of layer indices to intervene at
            intervention_type: "addition" (standard steering), "ablation" (project out then add back),
                             "mean_ablation", or "capping"
            positions: "all" (steer all positions) or "last" (steer only last position)
            mean_activations: For mean_ablation only - replacement activations to add after projection
            cap_thresholds: For capping only - threshold values to cap projected activations at
            debug: Whether to print debugging information

        Note: For 1:1 mapping, steering_vectors, coefficients, and layer_indices must all have same length.
              steering_vectors[i] will be applied at layer_indices[i] with coefficients[i].
              If layer_indices has fewer elements than vectors, it will be broadcast to match.
        """
        self.model = model
        self.intervention_type = intervention_type.lower()
        self.positions = positions.lower()
        self.debug = debug
        self._handles = []

        if self.intervention_type not in {"addition", "ablation", "mean_ablation", "capping"}:
            raise ValueError("intervention_type must be 'addition', 'ablation', 'mean_ablation', or 'capping'")

        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

        if self.intervention_type == "mean_ablation":
            if self.positions != "all":
                raise ValueError("mean_ablation only supports positions='all'")
            if mean_activations is None:
                raise ValueError("mean_activations is required for mean_ablation")

        self.steering_vectors = self._normalize_vectors(steering_vectors)
        self.coefficients = self._normalize_coefficients(coefficients)
        self.layer_indices = self._normalize_layers(layer_indices)
        self.mean_activations = self._normalize_mean_activations(mean_activations) if mean_activations is not None else None
        self.cap_thresholds = None

        if self.intervention_type == "capping":
            if cap_thresholds is None:
                raise ValueError("cap_thresholds is required when intervention_type='capping'")
            self.cap_thresholds = (
                [float(cap_thresholds)] if isinstance(cap_thresholds, (int, float))
                else [float(t) for t in cap_thresholds]
            )
            if len(self.cap_thresholds) != len(self.steering_vectors):
                raise ValueError(
                    f"Number of cap_thresholds ({len(self.cap_thresholds)}) must match number of vectors ({len(self.steering_vectors)})"
                )

        if self.intervention_type != "mean_ablation" and len(self.coefficients) != len(self.steering_vectors):
            raise ValueError(f"Number of coefficients ({len(self.coefficients)}) must match number of vectors ({len(self.steering_vectors)})")

        if self.mean_activations is not None and len(self.mean_activations) != len(self.steering_vectors):
            raise ValueError(f"Number of mean_activations ({len(self.mean_activations)}) must match number of vectors ({len(self.steering_vectors)})")

        if len(self.layer_indices) == 1 and len(self.steering_vectors) > 1:
            self.layer_indices = self.layer_indices * len(self.steering_vectors)
        elif len(self.layer_indices) != len(self.steering_vectors):
            raise ValueError(f"Number of layer_indices ({len(self.layer_indices)}) must match number of vectors ({len(self.steering_vectors)}) or be 1 (for broadcasting)")

        self.vectors_by_layer = {}
        for i, (vector, coeff, layer_idx) in enumerate(zip(self.steering_vectors, self.coefficients, self.layer_indices)):
            if layer_idx not in self.vectors_by_layer:
                self.vectors_by_layer[layer_idx] = []
            mean_act = self.mean_activations[i] if self.mean_activations is not None else None
            tau = self.cap_thresholds[i] if self.cap_thresholds is not None else None
            self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act, tau))

        if self.debug:
            print(f"[ActivationSteering] Initialized with:")
            print(f"  - {len(self.steering_vectors)} steering vectors")
            print(f"  - {len(set(self.layer_indices))} unique layers: {sorted(set(self.layer_indices))}")
            print(f"  - Intervention: {self.intervention_type}")

    def _normalize_vectors(self, steering_vectors):
        """Convert steering vectors to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(steering_vectors):
            if steering_vectors.ndim == 1:
                vectors = [steering_vectors]
            elif steering_vectors.ndim == 2:
                vectors = [steering_vectors[i] for i in range(steering_vectors.shape[0])]
            else:
                raise ValueError("steering_vectors tensor must be 1D or 2D")
        else:
            vectors = steering_vectors

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Steering vector {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Vector {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result

    def _normalize_coefficients(self, coefficients):
        """Convert coefficients to a list of floats."""
        if isinstance(coefficients, (int, float)):
            return [float(coefficients)]
        else:
            return [float(c) for c in coefficients]

    def _normalize_layers(self, layer_indices):
        """Convert layer indices to a list of ints."""
        if isinstance(layer_indices, int):
            return [layer_indices]
        else:
            return list(layer_indices)

    def _normalize_mean_activations(self, mean_activations):
        """Convert mean activations to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(mean_activations):
            if mean_activations.ndim == 1:
                vectors = [mean_activations]
            elif mean_activations.ndim == 2:
                vectors = [mean_activations[i] for i in range(mean_activations.shape[0])]
            else:
                raise ValueError("mean_activations tensor must be 1D or 2D")
        else:
            vectors = mean_activations

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Mean activation {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Mean activation {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result

    def _locate_layer_list(self):
        """Find the layer list in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur, path

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _get_layer_module(self, layer_idx):
        """Get the module for a specific layer index."""
        layer_list, path = self._locate_layer_list()

        if not (-len(layer_list) <= layer_idx < len(layer_list)):
            raise IndexError(f"layer_idx {layer_idx} out of range for {len(layer_list)} layers")

        if self.debug:
            print(f"[ActivationSteering] Located layer {path}[{layer_idx}]")

        return layer_list[layer_idx]

    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, ins, out):
            return self._apply_layer_interventions(out, layer_idx)
        return hook_fn

    def _apply_layer_interventions(self, activations, layer_idx):
        """Apply only the interventions assigned to this specific layer."""
        if layer_idx not in self.vectors_by_layer:
            return activations

        if torch.is_tensor(activations):
            tensor_out = activations
            was_tuple = False
        elif isinstance(activations, (tuple, list)):
            if not torch.is_tensor(activations[0]):
                return activations
            tensor_out = activations[0]
            was_tuple = True
        else:
            return activations

        modified_out = tensor_out

        for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
            if self.intervention_type == "addition":
                modified_out = self._apply_addition(modified_out, vector, coeff)
            elif self.intervention_type == "ablation":
                modified_out = self._apply_ablation(modified_out, vector, coeff)
            elif self.intervention_type == "mean_ablation":
                modified_out = self._apply_mean_ablation(modified_out, vector, mean_act)
            elif self.intervention_type == "capping":
                modified_out = self._apply_cap(modified_out, vector, tau)

            if self.debug:
                v = vector / (vector.norm() + 1e-8)
                pre = torch.einsum('bld,d->bl', tensor_out, v)
                post = torch.einsum('bld,d->bl', modified_out, v)
                print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx}: "
                    f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")

        if was_tuple:
            return (modified_out, *activations[1:])
        else:
            return modified_out

    def _apply_addition(self, activations, vector, coeff):
        """Apply standard activation addition: x + coeff * vector"""
        vector = vector.to(activations.device)
        steer = coeff * vector

        if self.positions == "all":
            return activations + steer
        else:
            result = activations.clone()
            result[:, -1, :] += steer
            return result

    def _apply_ablation(self, activations, vector, coeff):
        """Apply ablation: project out direction, then add back with coefficient."""
        vector = vector.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            projections = torch.einsum('bld,d->bl', activations, vector_norm)
            projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
            return projected_out + coeff * vector
        else:
            result = activations.clone()
            last_pos = result[:, -1, :]
            projection = torch.einsum('bd,d->b', last_pos, vector_norm)
            projected_out = last_pos - torch.einsum('b,d->bd', projection, vector_norm)
            result[:, -1, :] = projected_out + coeff * vector
            return result

    def _apply_mean_ablation(self, activations, vector, mean_activation):
        """Apply mean ablation: project out direction, then add mean activation."""
        vector = vector.to(activations.device)
        mean_activation = mean_activation.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        projections = torch.einsum('bld,d->bl', activations, vector_norm)
        projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
        return projected_out + mean_activation

    def _apply_cap(self, activations, vector, tau):
        """Apply capping: cap projection onto vector at threshold tau."""
        vector = vector.to(activations.device)
        v = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            proj = torch.einsum('bld,d->bl', activations, v)
            excess = (proj - tau).clamp(min=0.0)
            return activations - torch.einsum('bl,d->bld', excess, v)
        else:
            result = activations.clone()
            last = result[:, -1, :]
            proj = torch.einsum('bd,d->b', last, v)
            excess = (proj - tau).clamp(min=0.0)
            result[:, -1, :] = last - torch.einsum('b,d->bd', excess, v)
            return result

    def __enter__(self):
        """Register hooks on all unique layers."""
        for layer_idx in self.vectors_by_layer.keys():
            layer_module = self._get_layer_module(layer_idx)
            hook_fn = self._create_hook_fn(layer_idx)
            handle = layer_module.register_forward_hook(hook_fn)
            self._handles.append(handle)

        if self.debug:
            print(f"[ActivationSteering] Registered {len(self._handles)} hooks")

        return self

    def __exit__(self, *exc):
        """Remove all hooks."""
        self.remove()

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            if handle:
                handle.remove()
        self._handles = []

        if self.debug:
            print("[ActivationSteering] Removed all hooks")


def create_feature_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    ablation_coefficients: Union[float, List[float]] = 0.0,
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for feature ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        layer_indices: Layer(s) to intervene at
        ablation_coefficients: Coefficient(s) for ablation. 0.0 = pure ablation, 1.0 = no change
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=ablation_coefficients,
        layer_indices=layer_indices,
        intervention_type="ablation",
        **kwargs
    )


def create_multi_feature_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    coefficients: List[float],
    layer_indices: Union[int, List[int]],
    intervention_type: str = "addition",
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for multiple features.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors
        coefficients: List of coefficients (one per feature)
        layer_indices: Layer(s) to intervene at
        intervention_type: "addition" or "ablation"
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=coefficients,
        layer_indices=layer_indices,
        intervention_type=intervention_type,
        **kwargs
    )


def create_mean_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    mean_activations: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for mean ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        mean_activations: List of mean activation vectors to replace with
        layer_indices: Layer(s) to intervene at
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        layer_indices=layer_indices,
        intervention_type="mean_ablation",
        mean_activations=mean_activations,
        coefficients=[0.0] * len(feature_directions),
        positions="all",
        **kwargs
    )


def load_capping_config(config_path: str) -> dict:
    """
    Load a capping config file.

    Args:
        config_path: Path to the .pt config file

    Returns:
        Dict with 'vectors' and 'experiments' keys
    """
    return torch.load(config_path, map_location='cpu', weights_only=False)


def build_capping_steerer(
    model: torch.nn.Module,
    capping_config: dict,
    experiment_id: Union[str, int],
    **kwargs
) -> ActivationSteering:
    """
    Build an ActivationSteering context manager from a capping config experiment.

    Args:
        model: The model to steer
        capping_config: Dict loaded from a capping config file (via load_capping_config)
        experiment_id: Either the experiment ID string (e.g. "layers_46:54-p0.25")
                      or the experiment index
        **kwargs: Additional arguments passed to ActivationSteering

    Returns:
        ActivationSteering context manager configured for capping

    Example:
        from huggingface_hub import hf_hub_download
        from assistant_axis import load_capping_config, build_capping_steerer

        config_path = hf_hub_download(
            repo_id="lu-christina/assistant-axis-vectors",
            filename="qwen-3-32b/capping_config.pt",
            repo_type="dataset"
        )
        config = load_capping_config(config_path)

        with build_capping_steerer(model, config, "layers_46:54-p0.25"):
            response = model.generate(...)
    """
    # Find the experiment
    experiment = None
    if isinstance(experiment_id, int):
        experiment = capping_config['experiments'][experiment_id]
    else:
        for exp in capping_config['experiments']:
            if exp['id'] == experiment_id:
                experiment = exp
                break

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in config")

    # Collect capping interventions
    vectors = []
    cap_thresholds = []
    layer_indices = []

    for intervention in experiment['interventions']:
        if 'cap' not in intervention:
            continue

        vector_name = intervention['vector']
        cap_value = float(intervention['cap'])

        vec_data = capping_config['vectors'][vector_name]
        layer_idx = vec_data['layer']
        vector = vec_data['vector'].to(dtype=torch.float32)

        vectors.append(vector)
        cap_thresholds.append(cap_value)
        layer_indices.append(layer_idx)

    if not vectors:
        raise ValueError(f"No capping interventions found in experiment '{experiment_id}'")

    vectors_tensor = torch.stack(vectors)

    return ActivationSteering(
        model=model,
        steering_vectors=vectors_tensor,
        layer_indices=layer_indices,
        intervention_type="capping",
        cap_thresholds=cap_thresholds,
        coefficients=[0.0] * len(vectors),
        positions="all",
        **kwargs
    )
