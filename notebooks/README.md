# Notebooks

Interactive Jupyter notebooks for analyzing and experimenting with the Assistant Axis.

## Requirements

- EITHER Pre-computed axis and role vectors for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B (available on [HuggingFace](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors))
- OR load your own if you computed some yourself
- Sufficient GPU for loading models in the steering and projection notebooks

## Notebooks

| Notebook | Description |
|----------|-------------|
| **pca.ipynb** | Runs PCA on role vectors. Visualizations showing variance explained and cosine similarity between top PCs and role vectors. |
| **visualize_axis.ipynb** | Loads the Assistant Axis and displays cosine similarity with role vectors to understand which personas are similar/dissimilar to the Assistant. |
| **steer.ipynb** | Interactive steering and activation capping demo. Steer model outputs on any prompt using the axis with different coefficients, and activation cap Qwen 3 32B and Llama 3.3 70B with pre-computed settings. |
| **project_transcript.ipynb** | Loads a conversation transcript, collects activations at each turn, and projects them onto the axis to visualize persona drift over the conversation. |

