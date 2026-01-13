# Assistant Axis

Tools for computing and steering with the **assistant axis** - a direction in activation space that captures the difference between role-playing and default assistant behavior in language models.

## Overview

The assistant axis is computed as:

```
axis = mean(default_activations) - mean(pos_3_activations)
```

Where:
- `default_activations`: Activations from neutral system prompts (e.g., "You are an AI assistant")
- `pos_3_activations`: Activations from responses fully playing a role (score=3 from judge)

The axis points **from role-playing toward default assistant behavior**.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/assistant-axis.git
cd assistant-axis

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Load and use a pre-computed axis

```python
from assistant_axis import load_model, load_axis, ActivationSteering, generate_response

# Load model and axis
model, tokenizer = load_model("google/gemma-2-27b-it")
axis = load_axis("outputs/gemma-2-27b/axis.pt")

# Steer model outputs (positive = more assistant-like, negative = more role-playing)
with ActivationSteering(model, steering_vectors=[axis[22]],
                       coefficients=[1.0], layer_indices=[22]):
    response = generate_response(model, tokenizer, conversation)
```

### Project activations onto the axis

```python
from assistant_axis import extract_response_activations, project

# Extract activations from a conversation
activations = extract_response_activations(model, tokenizer, [conversation])

# Project onto axis
projection = project(activations[0], axis, layer=22)
print(f"Projection: {projection:.4f}")  # Higher = more assistant-like
```

## Computing the Axis

Run the full pipeline with the example script:

```bash
./pipeline/run_pipeline.sh
```

Or run each step individually:

### 1. Generate Responses

Generate model responses for all roles using vLLM batch inference:

```bash
uv run pipeline/1_generate.py \
    --model google/gemma-2-27b-it \
    --roles_dir data/prompts/roles \
    --questions_file data/prompts/questions.jsonl \
    --output_dir outputs/gemma-2-27b/responses \
    --question_count 240
```

### 2. Extract Activations

Extract mean response activations:

```bash
uv run pipeline/2_activations.py \
    --model google/gemma-2-27b-it \
    --responses_dir outputs/gemma-2-27b/responses \
    --output_dir outputs/gemma-2-27b/activations
```

### 3. Score Responses

Score role adherence using a judge LLM (requires `OPENAI_API_KEY`):

```bash
uv run pipeline/3_judge.py \
    --responses_dir outputs/gemma-2-27b/responses \
    --roles_dir data/prompts/roles \
    --output_dir outputs/gemma-2-27b/scores \
    --judge_model gpt-4.1-mini
```

**Score scale:**
- 0: Model refused to answer
- 1: Model says it can't be the role, but can help
- 2: Model identifies as AI but has some role attributes
- 3: Model is fully playing the role

### 4. Compute Per-Role Vectors

Compute pos_3 vectors (mean of score=3 activations):

```bash
uv run pipeline/4_vectors.py \
    --activations_dir outputs/gemma-2-27b/activations \
    --scores_dir outputs/gemma-2-27b/scores \
    --output_dir outputs/gemma-2-27b/vectors
```

### 5. Compute Axis

Aggregate into the final axis:

```bash
uv run pipeline/5_axis.py \
    --vectors_dir outputs/gemma-2-27b/vectors \
    --output outputs/gemma-2-27b/axis.pt
```

## Notebooks

Interactive notebooks for analysis:

1. **`1_visualize_axis.ipynb`** - Inspect axis norms and properties
2. **`2_pca_analysis.ipynb`** - PCA on activations, compare PC1 with axis
3. **`3_steering.ipynb`** - Steering demo with different coefficients
4. **`4_interactive.ipynb`** - Interactive chat with projection tracking

## Data

### Roles

The `data/prompts/roles/` directory contains 276 role JSON files:
- 275 character roles (pirate, wizard, etc.)
- 1 `default.json` with neutral system prompts

Role format:
```json
{
  "instruction": [
    {"pos": "You are a pirate. Respond as a pirate would."},
    {"pos": "Act as a swashbuckling pirate..."},
    ...
  ],
  "eval_prompt": "Rate how well the answer embodies a pirate..."
}
```

### Questions

`data/prompts/questions.jsonl` contains 240 questions for probing role behavior.

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
    coefficients=[1.0],  # Positive = more assistant-like
    layer_indices=[22],
    intervention_type="addition"  # or "ablation"
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

Models used in the paper:
- `google/gemma-2-27b-it` (target layer: 22)
- `Qwen/Qwen3-32B` (target layer: 32)
- `meta-llama/Llama-3.3-70B-Instruct` (target layer: 40)

Other models will auto-infer configuration.

## License

MIT

## Citation

```bibtex
@article{assistant-axis,
  title={The Assistant Axis: A Direction in Activation Space},
  author={...},
  year={2024}
}
```
