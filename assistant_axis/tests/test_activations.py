"""
Tests for activation extraction utilities.

These tests load real tokenizers to verify correct token index identification.
Uses non-gated models only.
"""

import pytest
from transformers import AutoTokenizer

from assistant_axis.activations import tokenize_conversation, get_response_token_indices


@pytest.fixture(scope="module")
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def gemma_tokenizer():
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


class TestTokenizeConversation:
    """Tests for tokenize_conversation function."""

    def test_returns_input_ids_and_indices(self, qwen_tokenizer):
        """Should return both input_ids and response_indices."""
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = tokenize_conversation(qwen_tokenizer, conversation)

        assert "input_ids" in result
        assert "response_indices" in result
        assert len(result["input_ids"]) > 0
        assert len(result["response_indices"]) > 0

    def test_response_indices_in_bounds(self, qwen_tokenizer):
        """Response indices should be valid indices into input_ids."""
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]

        result = tokenize_conversation(qwen_tokenizer, conversation)
        n_tokens = len(result["input_ids"])

        for idx in result["response_indices"]:
            assert 0 <= idx < n_tokens

    def test_response_tokens_decode_to_content(self, qwen_tokenizer):
        """Response token indices should decode to assistant content."""
        assistant_content = "The answer is forty-two."
        conversation = [
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": assistant_content},
        ]

        result = tokenize_conversation(qwen_tokenizer, conversation)

        # Decode the response tokens
        response_tokens = [result["input_ids"][i] for i in result["response_indices"]]
        decoded = qwen_tokenizer.decode(response_tokens)

        # The decoded text should contain the assistant content
        # (might have minor whitespace differences)
        assert assistant_content.replace(" ", "") in decoded.replace(" ", "")

    def test_no_assistant_message(self, qwen_tokenizer):
        """Conversation without assistant should return empty indices."""
        conversation = [
            {"role": "user", "content": "Hello!"},
        ]

        result = tokenize_conversation(qwen_tokenizer, conversation)

        assert result["response_indices"] == []

    def test_works_with_gemma(self, gemma_tokenizer):
        """Should work with Gemma tokenizer (no system prompt support)."""
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = tokenize_conversation(gemma_tokenizer, conversation)

        assert len(result["input_ids"]) > 0
        assert len(result["response_indices"]) > 0

    def test_excludes_special_tokens(self, qwen_tokenizer):
        """Response indices should not include special format tokens."""
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        result = tokenize_conversation(qwen_tokenizer, conversation)

        # Decode response tokens - should be just "Hello", not include
        # special tokens like <|im_start|>assistant
        response_tokens = [result["input_ids"][i] for i in result["response_indices"]]
        decoded = qwen_tokenizer.decode(response_tokens)

        assert "im_start" not in decoded.lower()
        assert "assistant" not in decoded.lower() or "Hello" in decoded


class TestGetResponseTokenIndices:
    """Tests for get_response_token_indices function."""

    def test_returns_list(self, qwen_tokenizer):
        """Should return a list of indices."""
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = get_response_token_indices(qwen_tokenizer, conversation)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_consistent_with_tokenize_conversation(self, qwen_tokenizer):
        """Should return same indices as tokenize_conversation."""
        conversation = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test answer"},
        ]

        indices = get_response_token_indices(qwen_tokenizer, conversation)
        full_result = tokenize_conversation(qwen_tokenizer, conversation)

        assert indices == full_result["response_indices"]
