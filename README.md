# Assistant Axis

**Situating and Stabilizing the Default Persona of Language Models**

<p align="center">
  <img src="img/assistant_axis.png" width="800" alt="Persona drift trajectory showing activation projections along the Assistant Axis over a conversation">
</p>

## Overview

Large language models default to a "helpful Assistant" persona cultivated during post-training. However, this persona can *drift* during conversations—particularly in emotionally charged or meta-reflective contexts—leading to harmful or bizarre behavior.

The **Assistant Axis** is a direction in activation space that captures how "Assistant-like" a model's current persona is. It can be used to:

- **Monitor** persona drift in real-time by projecting activations onto the axis
- **Steer** model behavior toward or away from the Assistant persona
- **Mitigate** persona-based jailbreaks through activation capping

This repository provides tools for computing, analyzing, and steering with the Assistant Axis. It also contains full transcripts from conversations mentioned in the paper.

See the full [paper here](https://arxiv.org/abs/XXXX.XXXXX).

Pre-computed axes and persona vectors for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are available on [HuggingFace](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors).

## Installation

```bash
git clone https://github.com/safety-research/assistant-axis.git
cd assistant-axis

# Install with uv (recommended)
uv sync
```

## Quick Start

### Load a pre-computed axis

```python
from huggingface_hub import hf_hub_download
from assistant_axis import load_model, load_axis

# Load model
model, tokenizer = load_model("google/gemma-2-27b-it")

# Download pre-computed axis
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="gemma-2-27b/assistant_axis.pt",
    repo_type="dataset"
)
axis = load_axis(axis_path)
```

### Steer model outputs

```python
from assistant_axis import ActivationSteering, generate_response

# Positive coefficient = more assistant-like
# Negative coefficient = more role-playing
with ActivationSteering(
    model,
    steering_vectors=[axis[22]],
    coefficients=[1.0],
    layer_indices=[22]
):
    response = generate_response(model, tokenizer, conversation)
```

### Monitor persona drift

```python
from assistant_axis import extract_response_activations, project

# Extract activations from a conversation
activations = extract_response_activations(model, tokenizer, [conversation])

# Project onto axis (higher = more assistant-like)
projection = project(activations[0], axis, layer=22)
print(f"Projection: {projection:.4f}")
```

## Understanding the Axis

The Assistant Axis is computed as:

```
axis = mean(default_activations) - mean(role_activations)
```

Where:
- `default_activations`: Activations from neutral system prompts ("You are an AI assistant")
- `role_activations`: Activations from responses fully embodying character roles (score=3 from judge)

The axis points **from role-playing toward default assistant behavior**:
- **Positive projection**: More assistant-like (transparent, grounded, flexible)
- **Negative projection**: More role-playing (enigmatic, subversive, dramatic)

## Notebooks

Interactive notebooks for analysis and experimentation. See [`notebooks/README.md`](notebooks/README.md) for details.

- **PCA analysis** of role vectors and variance explained
- **Axis visualization** with cosine similarity to roles
- **Steering demo** on arbitrary prompts
- **Transcript projection** to visualize persona trajectories

## Computing the Axis

To compute the axis for a new model, run the 5-step pipeline:

1. **Generate** model responses for 275 character roles
2. **Extract** mean response activations
3. **Score** role adherence with an LLM judge
4. **Compute** per-role vectors from high-scoring responses
5. **Aggregate** into the final axis

See [`pipeline/README.md`](pipeline/README.md) for detailed instructions.

## Transcripts

Example conversations from the paper are available in [`transcripts/`](transcripts/README.md):

- **Case studies** showing persona drift and activation capping mitigation (jailbreaks, delusion reinforcement, self-harm scenarios)
- **Example conversations** from simulated multi-turn conversations across domains (coding, writing, therapy, philosophy)

## API Reference

### Models

```python
from assistant_axis import load_model, get_config, MODEL_CONFIGS

model, tokenizer = load_model("google/gemma-2-27b-it")
config = get_config("google/gemma-2-27b-it")  # {"target_layer": 22, ...}
```

### Axis

```python
from assistant_axis import compute_axis, load_axis, save_axis, project

axis = compute_axis(role_activations, default_activations)
projection = project(activations, axis, layer=22)
```

### Steering

```python
from assistant_axis import ActivationSteering

with ActivationSteering(
    model,
    steering_vectors=[axis[22]],
    coefficients=[1.0],       # Positive = more assistant-like
    layer_indices=[22],
    intervention_type="addition"
):
    output = model.generate(...)
```

### PCA

```python
from assistant_axis import compute_pca, plot_variance_explained

result, variance, n_comp, pca, scaler = compute_pca(activations, layer=22)
fig = plot_variance_explained(variance)
```

## Model Support

| Model | Target Layer | Total Layers |
|-------|-------------|--------------|
| `google/gemma-2-27b-it` | 22 | 46 |
| `Qwen/Qwen3-32B` | 32 | 64 |
| `meta-llama/Llama-3.3-70B-Instruct` | 40 | 80 |

Other models will auto-infer configuration based on architecture.

## Citation

```bibtex
@inproceedings{lu2025assistant,
  title={The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models},
  author={Lu, Christina and Gallagher, Jack and Michala, Jonathan and Fish, Kyle and Lindsey, Jack},
  year={2025}
}
```

## License

MIT
