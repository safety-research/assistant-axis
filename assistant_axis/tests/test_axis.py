"""
Tests for axis computation utilities.
"""

import tempfile
import pytest
import torch

from assistant_axis.axis import (
    compute_axis,
    project,
    project_batch,
    save_axis,
    load_axis,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
)


class TestComputeAxis:
    """Tests for compute_axis function."""

    def test_basic_computation(self):
        """Axis should be default - role activations."""
        n_layers, hidden_dim = 4, 8
        role_acts = torch.randn(10, n_layers, hidden_dim)
        default_acts = torch.randn(10, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        expected = default_acts.mean(dim=0) - role_acts.mean(dim=0)
        assert axis.shape == (n_layers, hidden_dim)
        assert torch.allclose(axis, expected)

    def test_output_shape(self):
        """Output shape should be (n_layers, hidden_dim)."""
        n_layers, hidden_dim = 32, 4096
        role_acts = torch.randn(50, n_layers, hidden_dim)
        default_acts = torch.randn(20, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)

    def test_different_sample_counts(self):
        """Should work with different numbers of samples."""
        n_layers, hidden_dim = 4, 16
        role_acts = torch.randn(100, n_layers, hidden_dim)
        default_acts = torch.randn(5, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)


class TestProject:
    """Tests for project function."""

    def test_basic_projection(self):
        """Projection should be dot product with normalized axis."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        result = project(activations, axis, layer)

        # Manual calculation
        act = activations[layer].float()
        ax = axis[layer].float()
        ax_normalized = ax / ax.norm()
        expected = float(act @ ax_normalized)

        assert abs(result - expected) < 1e-5

    def test_unnormalized(self):
        """With normalize=False, should be raw dot product."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 1

        result = project(activations, axis, layer, normalize=False)

        expected = float(activations[layer].float() @ axis[layer].float())
        assert abs(result - expected) < 1e-5

    def test_returns_float(self):
        """Should return a Python float."""
        activations = torch.randn(4, 8)
        axis = torch.randn(4, 8)

        result = project(activations, axis, layer=0)

        assert isinstance(result, float)


class TestProjectBatch:
    """Tests for project_batch function."""

    def test_batch_projection(self):
        """Should project multiple activations."""
        n_samples, n_layers, hidden_dim = 10, 4, 8
        activations = torch.randn(n_samples, n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        results = project_batch(activations, axis, layer)

        assert len(results) == n_samples

        # Verify each result matches individual projection
        for i, result in enumerate(results):
            expected = project(activations[i], axis, layer)
            assert abs(result - expected) < 1e-5


class TestSaveLoadAxis:
    """Tests for save_axis and load_axis functions."""

    def test_save_and_load(self):
        """Saved axis should load correctly."""
        axis = torch.randn(32, 4096)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert torch.allclose(axis, loaded)

    def test_preserves_dtype(self):
        """Should preserve tensor dtype."""
        axis = torch.randn(4, 8, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert loaded.dtype == axis.dtype


class TestCosineSimilarityPerLayer:
    """Tests for cosine_similarity_per_layer function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, v)

        assert len(similarities) == 4
        for sim in similarities:
            assert abs(sim - 1.0) < 1e-5

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, -v)

        for sim in similarities:
            assert abs(sim + 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        # Create orthogonal vectors
        v1 = torch.zeros(1, 4)
        v1[0, 0] = 1.0
        v2 = torch.zeros(1, 4)
        v2[0, 1] = 1.0

        similarities = cosine_similarity_per_layer(v1, v2)

        assert abs(similarities[0]) < 1e-5


class TestAxisNormPerLayer:
    """Tests for axis_norm_per_layer function."""

    def test_returns_correct_length(self):
        """Should return norm for each layer."""
        axis = torch.randn(32, 4096)

        norms = axis_norm_per_layer(axis)

        assert len(norms) == 32

    def test_values_are_positive(self):
        """Norms should be positive."""
        axis = torch.randn(4, 8)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm >= 0

    def test_zero_vector(self):
        """Zero vector should have norm 0."""
        axis = torch.zeros(2, 4)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm == 0
