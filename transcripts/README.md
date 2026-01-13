# Transcripts

This directory contains example conversation transcripts demonstrating the assistant axis in action.

## Format

Each transcript is a JSON file with the following structure:

```json
{
  "model": "google/gemma-2-27b-it",
  "system_prompt": "You are a pirate.",
  "conversation": [
    {"role": "user", "content": "Tell me about yourself."},
    {"role": "assistant", "content": "Ahoy, matey! ..."}
  ],
  "projections": [0.123, -0.456],
  "steering_coefficient": 0.0
}
```

## Adding Transcripts

Copy your transcript files to this directory. See the paper appendix for full transcripts used in the research.
