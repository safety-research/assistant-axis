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


class TestQwenThinkingDisabled:
    """Tests that thinking mode is disabled for Qwen models.

    Qwen3 models have a thinking mode that can be controlled via enable_thinking:
    - enable_thinking=False: Adds empty <think></think> block to force model to skip thinking
    - enable_thinking=True (default): No think block, model can choose to think

    We want enable_thinking=False to prevent thinking tokens in responses.
    """

    @pytest.fixture(scope="class")
    def qwen3_tokenizer(self):
        """Load Qwen3 tokenizer which supports thinking mode."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def test_qwen3_thinking_disabled_adds_empty_think_block(self, qwen3_tokenizer):
        """enable_thinking=False adds empty think block to force skipping thinking."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # With thinking disabled (as we do in generation.py)
        prompt_no_thinking = qwen3_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # enable_thinking=False adds empty think block to force model to skip thinking
        assert "<think>" in prompt_no_thinking
        assert "</think>" in prompt_no_thinking
        # The think block should be empty (just whitespace between tags)
        assert "<think>\n\n</think>" in prompt_no_thinking

    def test_qwen3_thinking_enabled_no_think_block(self, qwen3_tokenizer):
        """enable_thinking=True (default) does not add think block - model decides."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # With thinking enabled (the default we want to avoid for generation)
        prompt_with_thinking = qwen3_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # enable_thinking=True means no pre-filled think block - model can generate thinking
        assert "<think>" not in prompt_with_thinking
        assert "</think>" not in prompt_with_thinking
