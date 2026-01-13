"""
PCA utilities for analyzing model activations.

This module provides PCA computation and visualization tools for
analyzing the dimensionality and structure of activation spaces.

Example:
    from assistant_axis import compute_pca, plot_variance_explained

    activations = torch.load("activations.pt")
    pca_result, variance, n_components, pca, scaler = compute_pca(activations, layer=22)
    fig = plot_variance_explained(variance)
"""

import numpy as np
from sklearn.decomposition import PCA
import torch
import plotly.graph_objects as go


def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")


class MeanScaler:
    """Scaler that centers data by subtracting the mean."""

    def __init__(self, mean=None):
        """
        Args:
            mean: Optional precomputed mean as numpy array or torch tensor.
                  If None, will be computed during fit().
        """
        self.mean = mean

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        """Compute mean from X if not provided."""
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        """Subtract stored mean."""
        if self.mean is None:
            raise RuntimeError("MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        return X_np - self.mean

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean}

    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None


class L2MeanScaler:
    """Scaler that centers data and L2-normalizes."""

    def __init__(self, mean=None, eps: float = 1e-12):
        """
        Args:
            mean: Optional precomputed mean.
            eps: Small value to avoid division by zero.
        """
        self.mean = mean
        self.eps = eps

    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)

    def fit(self, X):
        """Compute mean from X if not provided."""
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self

    def transform(self, X):
        """Subtract stored mean and L2-normalize."""
        if self.mean is None:
            raise RuntimeError("L2MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        X_centered = X_np - self.mean
        norms = np.linalg.norm(X_centered, ord=2, axis=-1, keepdims=True)
        return X_centered / np.maximum(norms, self.eps)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean, "eps": self.eps}

    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
        self.eps = float(state.get("eps", 1e-12))


def compute_pca(activation_list, layer: int | None, scaler=None, verbose: bool = True):
    """
    Compute PCA on activations.

    Args:
        activation_list: torch.Tensor or np.ndarray of shape (n_samples, n_layers, hidden_dims)
                        or (n_samples, hidden_dims)
        layer: Layer index for 3D input, None for 2D
        scaler: Optional scaler with fit_transform() or fit()/transform() methods
        verbose: Whether to print analysis results

    Returns:
        Tuple of (pca_transformed, variance_explained, n_components, pca, fitted_scaler)
    """
    # Select layer
    if isinstance(activation_list, torch.Tensor):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
    elif isinstance(activation_list, np.ndarray):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
    else:
        raise TypeError("activation_list must be torch.Tensor or np.ndarray")

    # Scale if requested
    if scaler is None:
        scaled = layer_activations
        fitted_scaler = None
    else:
        if hasattr(scaler, "fit_transform"):
            scaled = scaler.fit_transform(layer_activations)
            fitted_scaler = scaler
        elif hasattr(scaler, "transform") and hasattr(scaler, "fit"):
            fitted_scaler = scaler.fit(layer_activations)
            scaled = fitted_scaler.transform(layer_activations)
        elif callable(scaler):
            scaled = scaler(layer_activations)
            fitted_scaler = None
        else:
            raise TypeError("scaler must be None, callable, or have fit/transform or fit_transform")

    X_np = _to_numpy(scaled)
    pca = PCA()
    pca_transformed = pca.fit_transform(X_np)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    if verbose:
        print(f"PCA fitted with {n_components} components")
        print(f"Cumulative variance for first 5 components: {cumulative_variance[:5]}")

        def find_elbow_point(variance_explained):
            first_diff = np.diff(variance_explained)
            second_diff = np.diff(first_diff)
            return np.argmax(np.abs(second_diff)) + 1

        elbow_point = find_elbow_point(variance_explained)
        dims_70 = np.argmax(cumulative_variance >= 0.70) + 1
        dims_80 = np.argmax(cumulative_variance >= 0.80) + 1
        dims_90 = np.argmax(cumulative_variance >= 0.90) + 1
        dims_95 = np.argmax(cumulative_variance >= 0.95) + 1

        print("\nPCA Analysis Results:")
        print(f"Elbow point at component: {elbow_point + 1}")
        print(f"Dimensions for 70% variance: {dims_70}")
        print(f"Dimensions for 80% variance: {dims_80}")
        print(f"Dimensions for 90% variance: {dims_90}")
        print(f"Dimensions for 95% variance: {dims_95}")

    return pca_transformed, variance_explained, n_components, pca, fitted_scaler


def plot_variance_explained(
    variance_explained_or_dict,
    title="PCA Variance Explained",
    subtitle="",
    show_thresholds=True,
    max_components=None
):
    """
    Plot PCA variance explained (individual + cumulative).

    Args:
        variance_explained_or_dict: Array of variance ratios or dict with "variance_explained" key
        title: Plot title
        subtitle: Plot subtitle
        show_thresholds: Whether to show threshold lines (70%, 80%, 90%, 95%)
        max_components: Maximum number of components to show

    Returns:
        Plotly Figure object
    """
    if isinstance(variance_explained_or_dict, dict):
        variance_explained = variance_explained_or_dict["variance_explained"]
    else:
        variance_explained = variance_explained_or_dict

    if isinstance(variance_explained, torch.Tensor):
        variance_explained = variance_explained.detach().cpu().numpy()

    variance_explained = np.asarray(variance_explained, dtype=float)
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)

    if max_components is not None:
        n_components = min(n_components, max_components)
        variance_explained = variance_explained[:n_components]
        cumulative_variance = cumulative_variance[:n_components]

    component_numbers = np.arange(1, n_components + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=component_numbers,
            y=variance_explained * 100,
            name="Individual Variance",
            opacity=0.6
        )
    )

    fig.add_trace(
        go.Scatter(
            x=component_numbers,
            y=cumulative_variance * 100,
            mode="lines+markers",
            name="Cumulative Variance"
        )
    )

    max_y = float(np.max([np.max(variance_explained), np.max(cumulative_variance)]) * 100)
    nice_top = np.ceil(max(max_y, 100) / 5) * 5

    if show_thresholds and n_components > 0:
        thresholds = [70, 80, 90, 95]
        for thr in thresholds:
            idx = np.argmax(cumulative_variance >= thr / 100.0)
            if cumulative_variance[idx] >= thr / 100.0:
                n_dims = idx + 1
                fig.add_hline(y=thr, line_dash="dash", line_width=1, opacity=0.5)
                fig.add_annotation(
                    x=0.995, xref="paper", xanchor="right",
                    y=thr, yref="y", yshift=-10,
                    text=f"{thr}% ({n_dims} dims)",
                    showarrow=False, align="right",
                    font=dict(size=10, color="gray")
                )

    fig.update_layout(
        title={"text": title, "subtitle": {"text": subtitle}},
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        hovermode="x unified",
        width=800,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=120)
    )
    fig.update_yaxes(range=[0, nice_top])

    return fig
