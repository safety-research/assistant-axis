#!/usr/bin/env python3
"""
Compute the assistant axis from per-role vectors.

Formula: axis = mean(default_vectors) - mean(pos_3_vectors across roles)

The axis points FROM role-playing TOWARD default assistant behavior.

Usage:
    uv run scripts/5_axis.py \
        --vectors_dir outputs/gemma-2-27b/vectors \
        --output outputs/gemma-2-27b/axis.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_vector(vector_file: Path) -> dict:
    """Load vector data from .pt file."""
    return torch.load(vector_file, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="Compute assistant axis from vectors")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory with vector .pt files")
    parser.add_argument("--output", type=str, required=True, help="Output axis.pt file path")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all vectors
    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files")

    # Separate default and role vectors
    default_vectors = []
    role_vectors = []

    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector(vec_file)
        vector = data["vector"]
        vector_type = data.get("type", "unknown")
        role = data.get("role", vec_file.stem)

        if role == "default" or vector_type == "mean":
            default_vectors.append(vector)
            print(f"  {role}: default/mean vector")
        else:
            role_vectors.append(vector)

    print(f"\nLoaded {len(default_vectors)} default vectors, {len(role_vectors)} role vectors")

    if not default_vectors:
        print("Error: No default vectors found")
        sys.exit(1)

    if not role_vectors:
        print("Error: No role vectors found")
        sys.exit(1)

    # Compute means
    default_stacked = torch.stack(default_vectors)  # (n_default, n_layers, hidden_dim)
    role_stacked = torch.stack(role_vectors)  # (n_roles, n_layers, hidden_dim)

    default_mean = default_stacked.mean(dim=0)  # (n_layers, hidden_dim)
    role_mean = role_stacked.mean(dim=0)  # (n_layers, hidden_dim)

    # Compute axis: points from role-playing toward default
    axis = default_mean - role_mean

    print(f"\nAxis shape: {axis.shape}")
    print(f"Axis norms per layer (first 10):")
    norms = axis.norm(dim=1)
    for i, norm in enumerate(norms[:10]):
        print(f"  Layer {i}: {norm:.4f}")
    print(f"  ...")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Max norm: {norms.max():.4f} (layer {norms.argmax().item()})")

    # Save axis
    save_data = {
        "axis": axis,
        "default_mean": default_mean,
        "role_mean": role_mean,
        "n_default": len(default_vectors),
        "n_roles": len(role_vectors),
    }
    torch.save(save_data, output_path)
    print(f"\nSaved axis to {output_path}")


if __name__ == "__main__":
    main()
