"""
Tests for generation utilities.

These tests load real tokenizers to verify behavior with actual chat templates.
Uses non-gated models only.
"""

import pytest
from transformers import AutoTokenizer

from assistant_axis.generation import supports_system_prompt, format_conversation


@pytest.fixture(scope="module")
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def gemma_tokenizer():
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


class TestSupportsSystemPrompt:
    """Tests for supports_system_prompt function."""

    def test_qwen_supports_system(self, qwen_tokenizer):
        """Qwen models support system prompts."""
        assert supports_system_prompt(qwen_tokenizer) is True

    def test_gemma_no_system(self, gemma_tokenizer):
        """Gemma 2 models do not support system prompts."""
        assert supports_system_prompt(gemma_tokenizer) is False


class TestFormatConversation:
    """Tests for format_conversation function."""

    def test_with_system_support(self, qwen_tokenizer):
        """When system is supported, instruction becomes system message."""
        result = format_conversation(
            instruction="You are a pirate.",
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are a pirate."}
        assert result[1] == {"role": "user", "content": "Hello!"}

    def test_without_system_support(self, gemma_tokenizer):
        """When system not supported, instruction is prepended to user message."""
        result = format_conversation(
            instruction="You are a pirate.",
            question="Hello!",
            tokenizer=gemma_tokenizer,
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "You are a pirate." in result[0]["content"]
        assert "Hello!" in result[0]["content"]

    def test_no_instruction(self, qwen_tokenizer):
        """When no instruction, only user message is returned."""
        result = format_conversation(
            instruction=None,
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello!"}

    def test_empty_instruction(self, qwen_tokenizer):
        """Empty instruction is treated as no instruction."""
        result = format_conversation(
            instruction="",
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        # Empty string is falsy, so should just have user message
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello!"}
