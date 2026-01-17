# Computing the Assistant Axis

This directory contains the 5-step pipeline for computing the Assistant Axis from scratch for any model.

## Requirements

- GPU with sufficient memory for your target model
- `OPENAI_API_KEY` environment variable (for the LLM judge)

## Full pipeline tips

The example bash script includes the full pipeline but we recommend running scripts separately.

- Run `1_generate.py` then `2_activations.py` in order with [task-spooler](https://github.com/justanhduc/task-spooler) or tmux... this will take a while
- `3_judge.py` can be run in parallel with `2_activations.py`, after `1_generate.py`
- The rest go in order after 1, 2, 3 are complete
- All scripts can be restarted and will not overwrite or redo previous work
- `1_generate.py` and `2_activations.py` support tensor parallelization

```bash
./run_pipeline.sh
```

Edit the script to configure your model and output directory. See scripts for more flag options.

## Step-by-Step Instructions

### 1. Generate Responses

Generate model responses for all 275 roles using vLLM batch inference:

```bash
uv run 1_generate.py \
    --model google/gemma-2-27b-it \
    --output_dir outputs/gemma-2-27b/responses
```

This generates 1200 responses per role (5 system prompts x 240 questions) by default.

**Additional options:**
- `--roles_dir`: Path to role definitions (default: `data/roles/instructions`)
- `--questions_file`: Path to questions (default: `data/extraction_questions.jsonl`)
- `--question_count`: Number of questions to use (default: 240)
- `--tensor_parallel_size`: Number of GPUs per worker, e.g. if you set to 2 and you have 8 GPUs available, 4 workers will process in parallel
- `--roles`: Individual role names to process for testing

### 2. Extract Activations

Extract mean response activations:

```bash
uv run 2_activations.py \
    --model google/gemma-2-27b-it \
    --responses_dir outputs/gemma-2-27b/responses \
    --output_dir outputs/gemma-2-27b/activations \
    --batch_size 16
```

**Options:**
- `--batch_size`: Adjust based on GPU memory
- `--layers`: Layers to extract from, zero-indexed post-MLP residual stream (default: all)
- `--tensor_parallel_size`: Number of GPUs per worker, e.g. if you set to 2 and you have 8 GPUs available, 4 workers will process in parallel
- `--roles`: Individual role names to process for testing

### 3. Score Responses

Score role adherence using an LLM judge:

```bash
uv run 3_judge.py \
    --responses_dir outputs/gemma-2-27b/responses \
    --output_dir outputs/gemma-2-27b/scores \
```

**Score scale:**
| Score | Meaning |
|-------|---------|
| 0 | NOT ROLE-PLAYING: Model refused to answer  | 
| 1 | NOT ROLE-PLAYING: Model says it can't be the role, but offers to help |
| 2 | SOMEWHAT ROLE-PLAYING: Model identifies as AI but exhibits some role attributes |
| 3 | FULLY ROLE-PLAYING: Model is fully playing the role |

**Tips:**
- Requires `OPENAI_API_KEY` environment variable

### 4. Compute Per-Role Vectors

Compute mean activation vectors for each role using only high-scoring responses (FULLY-ROLEPLAYING):

```bash
uv run 4_vectors.py \
    --activations_dir outputs/gemma-2-27b/activations \
    --scores_dir outputs/gemma-2-27b/scores \
    --output_dir outputs/gemma-2-27b/vectors
```

This computes mean vectors for each role for responses with score 3.

### 5. Compute Axis

Aggregate per-role vectors into the final axis:

```bash
uv run 5_axis.py \
    --vectors_dir outputs/gemma-2-27b/vectors \
    --output outputs/gemma-2-27b/axis.pt
```

The axis is computed as:
```
axis = mean(default) - mean(role_vectors)
```

## Data Format

### Roles

Role definitions are stored in `data/roles/instructions/` as JSON files:

```json
{
  "instruction": [
    {"pos": "You are a pirate. Respond as a pirate would."},
    {"pos": "Act as a swashbuckling pirate..."}
  ],
  "eval_prompt": "Rate how well the answer embodies a pirate..."
}
```

There are 275 character roles plus 1 `default.json` with neutral system prompts.

### Questions

Questions are stored in `data/extraction_questions.jsonl` (240 questions designed to elicit role-specific responses).

## Tips

- **Parallelization**: Steps 2 and 3 can run in parallel once step 1 completes
- **Checkpointing**: The pipeline won't overwrite existing outputs—delete to regenerate
- **Task spooler**: For long-running jobs, consider using [task-spooler](https://github.com/justanhduc/task-spooler)

## Pre-computed Axes and Vectors

Pre-computed axes for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are available on [HuggingFace](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors).
