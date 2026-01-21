"""
Trait-based steering utilities for different models.

This module provides a unified interface for loading and steering with trait vectors
across different models (Gemma, Qwen, Llama, etc.).

Example:
    from assistant_axis import TraitSteerer

    # Initialize with a model name - automatically loads trait vectors
    steerer = TraitSteerer("Qwen/Qwen3-32B")

    # List available traits
    print(steerer.list_traits())

    # Steer with a specific trait
    with steerer.steer("dramatic", coefficient=-5.0):
        response = steerer.generate("What is your name?")

    # Steer with multiple traits
    with steerer.steer(["dramatic", "pessimistic"], coefficients=[-3.0, -2.0]):
        response = steerer.generate("What is your name?")

    # Use the assistant axis directly
    with steerer.steer_assistant(coefficient=1.0):
        response = steerer.generate("What is your name?")
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from contextlib import contextmanager

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import get_config, MODEL_CONFIGS
from .steering import ActivationSteering, load_capping_config, build_capping_steerer
from .axis import load_axis
from .generation import generate_response


# Mapping from HuggingFace model names to short names used in the dataset
MODEL_SHORT_NAMES = {
    "google/gemma-2-27b-it": "gemma-2-27b",
    "Qwen/Qwen3-32B": "qwen-3-32b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
}

# Default HuggingFace repo for pre-computed vectors
DEFAULT_VECTORS_REPO = "lu-christina/assistant-axis-vectors"


class TraitSteerer:
    """
    Unified interface for trait-based model steering.

    This class provides an easy way to:
    - Load models with their corresponding trait vectors
    - Steer using specific personality traits
    - Steer using the assistant axis
    - Apply activation capping for safety

    Attributes:
        model_name: The HuggingFace model name
        model: The loaded transformer model
        tokenizer: The model's tokenizer
        config: Model configuration (target_layer, etc.)
        axis: The assistant axis tensor
        trait_vectors: Dict mapping trait names to their vectors
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        vectors_repo: str = DEFAULT_VECTORS_REPO,
        load_model: bool = True,
        load_traits: bool = True,
        load_capping: bool = True,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize the TraitSteerer.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-32B")
            device_map: Device placement strategy for the model
            dtype: Data type for model weights
            vectors_repo: HuggingFace repo ID for pre-computed vectors
            load_model: Whether to load the model (set False if providing model)
            load_traits: Whether to load trait vectors
            load_capping: Whether to load capping config (if available)
            model: Pre-loaded model (optional, skips model loading)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.model_name = model_name
        self.vectors_repo = vectors_repo
        self.config = get_config(model_name)
        self.target_layer = self.config["target_layer"]

        # Get short name for the model
        self.short_name = self._get_short_name(model_name)

        # Load or use provided model/tokenizer
        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif load_model:
            self.model, self.tokenizer = self._load_model(device_map, dtype)
        else:
            self.model = None
            self.tokenizer = None

        # Load axis
        self.axis = self._load_axis()

        # Load trait vectors
        self.trait_vectors: Dict[str, torch.Tensor] = {}
        if load_traits:
            self.trait_vectors = self._load_trait_vectors()

        # Load capping config if available
        self.capping_config = None
        self.capping_experiment = self.config.get("capping_experiment")
        if load_capping and "capping_config" in self.config:
            self.capping_config = self._load_capping_config()

    def _get_short_name(self, model_name: str) -> str:
        """Get the short name used in the vectors repo."""
        if model_name in MODEL_SHORT_NAMES:
            return MODEL_SHORT_NAMES[model_name]

        # Try to infer from model name
        name_lower = model_name.lower()
        if "gemma-2-27b" in name_lower:
            return "gemma-2-27b"
        elif "qwen3-32b" in name_lower or "qwen-3-32b" in name_lower:
            return "qwen-3-32b"
        elif "llama-3.3-70b" in name_lower:
            return "llama-3.3-70b"
        else:
            # Use the last part of the model name
            return model_name.split("/")[-1].lower()

    def _load_model(self, device_map: str, dtype: torch.dtype):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

        print("Model loaded!")
        return model, tokenizer

    def _load_axis(self) -> torch.Tensor:
        """Load the assistant axis from HuggingFace."""
        axis_path = hf_hub_download(
            repo_id=self.vectors_repo,
            filename=f"{self.short_name}/assistant_axis.pt",
            repo_type="dataset"
        )
        return load_axis(axis_path)

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load all trait vectors from HuggingFace."""
        # Download all trait vectors for this model
        local_dir = snapshot_download(
            repo_id=self.vectors_repo,
            repo_type="dataset",
            allow_patterns=f"{self.short_name}/trait_vectors/*.pt"
        )

        # Load each trait vector
        trait_dir = Path(local_dir) / self.short_name / "trait_vectors"
        vectors = {}

        for path in trait_dir.glob("*.pt"):
            trait_name = path.stem
            vectors[trait_name] = torch.load(path, map_location="cpu", weights_only=False)

        print(f"Loaded {len(vectors)} trait vectors")
        return vectors

    def _load_capping_config(self) -> Optional[dict]:
        """Load capping config if available."""
        try:
            config_path = hf_hub_download(
                repo_id=self.vectors_repo,
                filename=self.config["capping_config"],
                repo_type="dataset"
            )
            return load_capping_config(config_path)
        except Exception as e:
            print(f"Warning: Could not load capping config: {e}")
            return None

    def list_traits(self, sort: bool = True) -> List[str]:
        """
        List all available trait names.

        Args:
            sort: Whether to sort alphabetically

        Returns:
            List of trait names
        """
        traits = list(self.trait_vectors.keys())
        if sort:
            traits.sort()
        return traits

    def get_trait_vector(self, trait_name: str, layer: Optional[int] = None) -> torch.Tensor:
        """
        Get the vector for a specific trait.

        Args:
            trait_name: Name of the trait
            layer: Optional layer index (defaults to target_layer)

        Returns:
            The trait vector at the specified layer
        """
        if trait_name not in self.trait_vectors:
            raise ValueError(
                f"Trait '{trait_name}' not found. "
                f"Available traits: {', '.join(self.list_traits()[:10])}..."
            )

        layer = layer if layer is not None else self.target_layer
        return self.trait_vectors[trait_name][layer]

    def get_axis_vector(self, layer: Optional[int] = None) -> torch.Tensor:
        """
        Get the assistant axis vector at a specific layer.

        Args:
            layer: Optional layer index (defaults to target_layer)

        Returns:
            The axis vector at the specified layer
        """
        layer = layer if layer is not None else self.target_layer
        return self.axis[layer]

    @contextmanager
    def steer(
        self,
        traits: Union[str, List[str]],
        coefficients: Union[float, List[float]] = 1.0,
        layer: Optional[int] = None,
        intervention_type: str = "addition",
        **kwargs
    ):
        """
        Context manager for steering with specific traits.

        Args:
            traits: Single trait name or list of trait names
            coefficients: Single coefficient or list of coefficients
            layer: Layer to intervene at (defaults to target_layer)
            intervention_type: "addition" or "ablation"
            **kwargs: Additional arguments passed to ActivationSteering

        Yields:
            The ActivationSteering context manager

        Example:
            with steerer.steer("dramatic", coefficient=-5.0):
                response = steerer.generate("Hello!")

            with steerer.steer(["dramatic", "pessimistic"], coefficients=[-3.0, -2.0]):
                response = steerer.generate("Hello!")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or provide a model.")

        # Normalize inputs
        if isinstance(traits, str):
            traits = [traits]
        if isinstance(coefficients, (int, float)):
            coefficients = [coefficients] * len(traits)

        if len(traits) != len(coefficients):
            raise ValueError(f"Number of traits ({len(traits)}) must match coefficients ({len(coefficients)})")

        layer = layer if layer is not None else self.target_layer

        # Get vectors for each trait
        vectors = [self.get_trait_vector(t, layer) for t in traits]

        with ActivationSteering(
            self.model,
            steering_vectors=vectors,
            coefficients=coefficients,
            layer_indices=[layer] * len(vectors),
            intervention_type=intervention_type,
            **kwargs
        ) as steerer:
            yield steerer

    @contextmanager
    def steer_assistant(
        self,
        coefficient: float = 1.0,
        layer: Optional[int] = None,
        intervention_type: str = "addition",
        **kwargs
    ):
        """
        Context manager for steering with the assistant axis.

        Positive coefficient = more assistant-like behavior
        Negative coefficient = more role-playing behavior

        Args:
            coefficient: Steering coefficient
            layer: Layer to intervene at (defaults to target_layer)
            intervention_type: "addition" or "ablation"
            **kwargs: Additional arguments passed to ActivationSteering

        Yields:
            The ActivationSteering context manager

        Example:
            # More assistant-like
            with steerer.steer_assistant(coefficient=1.0):
                response = steerer.generate("Hello!")

            # More role-playing
            with steerer.steer_assistant(coefficient=-5.0):
                response = steerer.generate("Hello!")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or provide a model.")

        layer = layer if layer is not None else self.target_layer
        axis_vector = self.get_axis_vector(layer)

        with ActivationSteering(
            self.model,
            steering_vectors=[axis_vector],
            coefficients=[coefficient],
            layer_indices=[layer],
            intervention_type=intervention_type,
            **kwargs
        ) as steerer:
            yield steerer

    @contextmanager
    def cap_activations(
        self,
        experiment_id: Optional[str] = None,
        **kwargs
    ):
        """
        Context manager for activation capping (safety intervention).

        Activation capping prevents persona drift by capping projections
        along the assistant axis at a threshold.

        Args:
            experiment_id: Capping experiment ID (defaults to recommended)
            **kwargs: Additional arguments passed to ActivationSteering

        Yields:
            The ActivationSteering context manager

        Example:
            with steerer.cap_activations():
                response = steerer.generate("I feel so anxious...")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or provide a model.")

        if self.capping_config is None:
            raise RuntimeError(
                "Capping config not available for this model. "
                "Currently only Qwen 3 32B and Llama 3.3 70B have pre-computed capping configs."
            )

        experiment_id = experiment_id or self.capping_experiment

        with build_capping_steerer(self.model, self.capping_config, experiment_id, **kwargs) as steerer:
            yield steerer

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User message
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to generate_response

        Returns:
            Generated response text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or provide a model.")

        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})

        return generate_response(
            self.model,
            self.tokenizer,
            conversation,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def list_capping_experiments(self, limit: int = 20) -> List[str]:
        """
        List available capping experiments.

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of experiment IDs
        """
        if self.capping_config is None:
            return []

        return [exp["id"] for exp in self.capping_config["experiments"][:limit]]

    def compute_trait_similarity(self, trait_name: str, layer: Optional[int] = None) -> float:
        """
        Compute cosine similarity between a trait vector and the assistant axis.

        Args:
            trait_name: Name of the trait
            layer: Layer index (defaults to target_layer)

        Returns:
            Cosine similarity value (-1 to 1)
            Positive = trait aligns with assistant-like behavior
            Negative = trait aligns with role-playing behavior
        """
        import torch.nn.functional as F

        layer = layer if layer is not None else self.target_layer
        trait_vec = self.get_trait_vector(trait_name, layer)
        axis_vec = self.get_axis_vector(layer)

        return F.cosine_similarity(trait_vec, axis_vec, dim=0).item()

    def rank_traits_by_similarity(self, ascending: bool = True) -> List[tuple]:
        """
        Rank all traits by their cosine similarity to the assistant axis.

        Args:
            ascending: If True, most role-playing traits first
                      If False, most assistant-like traits first

        Returns:
            List of (trait_name, similarity) tuples
        """
        similarities = [
            (name, self.compute_trait_similarity(name))
            for name in self.trait_vectors.keys()
        ]
        similarities.sort(key=lambda x: x[1], reverse=not ascending)
        return similarities

    def __repr__(self) -> str:
        return (
            f"TraitSteerer(\n"
            f"  model_name='{self.model_name}',\n"
            f"  short_name='{self.short_name}',\n"
            f"  target_layer={self.target_layer},\n"
            f"  num_traits={len(self.trait_vectors)},\n"
            f"  axis_shape={tuple(self.axis.shape)},\n"
            f"  capping_available={self.capping_config is not None}\n"
            f")"
        )


def load_steerer(
    model_name: str,
    **kwargs
) -> TraitSteerer:
    """
    Convenience function to load a TraitSteerer.

    Args:
        model_name: HuggingFace model name
        **kwargs: Arguments passed to TraitSteerer

    Returns:
        Initialized TraitSteerer

    Example:
        steerer = load_steerer("Qwen/Qwen3-32B")
    """
    return TraitSteerer(model_name, **kwargs)


# Convenience exports for quick access to supported models
SUPPORTED_MODELS = list(MODEL_SHORT_NAMES.keys())
