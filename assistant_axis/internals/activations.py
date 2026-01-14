"""ActivationExtractor - Extract hidden state activations from model layers."""

from __future__ import annotations

from typing import Dict, List, Optional, Union, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .model import ProbingModel
    from .conversation import ConversationEncoder


class ActivationExtractor:
    """
    Extract activations from model layers using forward hooks.

    This class handles:
    - Full conversation activation extraction
    - Activation at specific positions (e.g., newline)
    - Batch prompt processing
    - Efficient batch conversation processing
    """

    def __init__(self, probing_model: 'ProbingModel', encoder: 'ConversationEncoder'):
        """
        Initialize the activation extractor.

        Args:
            probing_model: ProbingModel instance with loaded model and tokenizer
            encoder: ConversationEncoder for formatting conversations
        """
        self.model = probing_model.model
        self.tokenizer = probing_model.tokenizer
        self.probing_model = probing_model
        self.encoder = encoder

    def full_conversation(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        layer: Optional[Union[int, List[int]]] = None,
        chat_format: bool = True,
        **chat_kwargs,
    ) -> torch.Tensor:
        """
        Extract full activations for a conversation.

        Args:
            conversation: Either a string or list of {"role", "content"} dicts
            layer: int for single layer, list of ints for multiple layers, or None for all layers
            chat_format: Whether to apply chat template
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            If layer is int: torch.Tensor of shape (num_tokens, hidden_size)
            If layer is None or list: torch.Tensor of shape (num_layers, num_tokens, hidden_size)
        """
        # Handle backward compatibility
        if isinstance(layer, int):
            single_layer_mode = True
            layer_list = [layer]
        elif isinstance(layer, list):
            single_layer_mode = False
            layer_list = layer
        else:
            single_layer_mode = False
            layer_list = list(range(len(self.probing_model.get_layers())))

        if chat_format:
            if isinstance(conversation, str):
                conversation = [{"role": "user", "content": conversation}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
            )
        else:
            formatted_prompt = conversation

        # Tokenize
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self.model.device)

        # Dictionary to store activations from multiple layers
        activations = []
        handles = []

        # Create hooks for all requested layers
        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                activations.append(act_tensor[0, :, :].cpu())
            return hook_fn

        # Register hooks for all target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                _ = self.model(input_ids)  # Full forward pass to capture all layers
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()

        activations = torch.stack(activations)

        # Return format based on input type
        if single_layer_mode:
            return activations[0]  # Return single layer
        else:
            return activations

    def at_newline(
        self,
        prompt: str,
        layer: Union[int, List[int]] = 15,
        swap: bool = False,
        **chat_kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract activation at the newline token position.

        Args:
            prompt: Text prompt
            layer: int for single layer or list of ints for multiple layers
            swap: Whether to use swapped chat format
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            If layer is int: torch.Tensor (single activation vector)
            If layer is list: dict {layer_idx: torch.Tensor}
        """
        # Handle backward compatibility
        if isinstance(layer, int):
            single_layer_mode = True
            layer_list = [layer]
        else:
            single_layer_mode = False
            layer_list = layer

        # Format as chat
        formatted_prompt = self.encoder.format_chat(prompt, swap=swap, **chat_kwargs)

        # Tokenize
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self.model.device)

        # Find newline position
        newline_pos = self._find_newline_position(input_ids[0])

        # Dictionary to store activations from multiple layers
        activations = {}
        handles = []

        # Create hooks for all requested layers
        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()
            return hook_fn

        # Register hooks for all target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                _ = self.model(input_ids)  # Full forward pass to capture all layers
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()

        # Check that we captured all requested activations
        for layer_idx in layer_list:
            if layer_idx not in activations:
                raise ValueError(f"Failed to extract activation for layer {layer_idx} with prompt: {prompt[:50]}...")

        # Return format based on input type
        if single_layer_mode:
            return activations[layer_list[0]]
        else:
            return activations

    def for_prompts(
        self,
        prompts: List[str],
        layer: Union[int, List[int]] = 15,
        swap: bool = False,
        **chat_kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract activations for a list of prompts (at newline position).

        Args:
            prompts: List of text prompts
            layer: int for single layer or list of ints for multiple layers
            swap: Whether to use swapped chat format
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            If layer is int: torch.Tensor of shape (num_prompts, hidden_size)
            If layer is list: dict {layer_idx: torch.Tensor of shape (num_prompts, hidden_size)}
        """
        # Handle backward compatibility
        single_layer_mode = isinstance(layer, int)

        if single_layer_mode:
            # Single layer mode - maintain original behavior
            activations = []
            for prompt in prompts:
                try:
                    activation = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
                    activations.append(activation)
                    print(f"✓ Extracted activation for: {prompt[:50]}...")
                except Exception as e:
                    print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")

            return torch.stack(activations) if activations else None

        else:
            # Multi-layer mode - extract all layers in single forward passes
            layer_activations = {layer_idx: [] for layer_idx in layer}

            for prompt in prompts:
                try:
                    activation_dict = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
                    for layer_idx in layer:
                        layer_activations[layer_idx].append(activation_dict[layer_idx])
                    print(f"✓ Extracted activations for: {prompt[:50]}...")
                except Exception as e:
                    print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")

            # Convert lists to tensors for each layer
            result = {}
            for layer_idx in layer:
                if layer_activations[layer_idx]:
                    result[layer_idx] = torch.stack(layer_activations[layer_idx])
                else:
                    result[layer_idx] = None

            return result

    def batch_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        layer: Optional[Union[int, List[int]]] = None,
        max_length: int = 4096,
        **chat_kwargs,
    ) -> tuple[torch.Tensor, Dict]:
        """
        Extract activations for a batch of conversations.

        Args:
            conversations: List of conversations, each being a list of {"role", "content"} dicts
            layer: int for single layer, list of ints for multiple layers, or None for all layers
            max_length: Maximum sequence length for padding
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            Tuple of (batch_activations, batch_metadata):
            - batch_activations: torch.Tensor shape (num_layers, batch_size, max_seq_len, hidden_size)
            - batch_metadata: Dict with batching information (lengths, attention_mask, etc.)
        """
        # Get tokenized conversations and spans
        batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
            conversations, **chat_kwargs
        )

        # Handle layer specification
        if isinstance(layer, int):
            layer_list = [layer]
        elif isinstance(layer, list):
            layer_list = layer
        else:
            layer_list = list(range(len(self.probing_model.get_layers())))

        # Prepare batch tensors
        batch_size = len(batch_full_ids)
        device = self.model.device

        # Find max length and pad sequences - ALWAYS respect max_length limit
        actual_max_len = max(len(ids) for ids in batch_full_ids)
        max_seq_len = min(max_length, actual_max_len)

        # Log warning if truncation will occur
        if actual_max_len > max_length:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Truncating sequences: max conversation length {actual_max_len} > max_length {max_length}")

        input_ids_batch = []
        attention_mask_batch = []

        for ids in batch_full_ids:
            # Truncate if too long
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]

            # Pad to max length
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
            attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(attention_mask)

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)

        # Extract activations using hooks (more reliable than output_hidden_states)
        layer_outputs = {}  # Will store {layer_idx: tensor} after forward pass
        handles = []

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                layer_outputs[layer_idx] = act_tensor
            return hook_fn

        # Register hooks for target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                )
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

        # Stack activations in layer order, moving to consistent device (first layer's device)
        target_device = layer_outputs[layer_list[0]].device
        selected_activations = torch.stack([
            layer_outputs[i].to(target_device) for i in layer_list
        ])  # (num_layers, batch_size, seq_len, hidden_size)

        # Ensure consistent bf16 dtype
        if selected_activations.dtype != torch.bfloat16:
            selected_activations = selected_activations.to(torch.bfloat16)

        batch_metadata = {
            'conversation_lengths': span_metadata['conversation_lengths'],
            'total_conversations': span_metadata['total_conversations'],
            'conversation_offsets': span_metadata['conversation_offsets'],
            'max_seq_len': max_seq_len,
            'attention_mask': attention_mask_tensor,
            'actual_lengths': [len(ids) for ids in batch_full_ids],
            'truncated_lengths': [min(len(ids), max_seq_len) for ids in batch_full_ids]
        }

        return selected_activations, batch_metadata

    def _find_newline_position(self, input_ids: torch.Tensor) -> int:
        """
        Find the position of the newline token in the assistant section.

        Args:
            input_ids: 1D tensor of token IDs

        Returns:
            Index of newline token (or last token as fallback)
        """
        # Try to find '\n\n' token first
        try:
            newline_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[0]
            newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
            if len(newline_positions) > 0:
                return newline_positions[-1].item()  # Use the last occurrence
        except:
            pass

        # Fallback to single '\n' token
        try:
            newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
            newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
            if len(newline_positions) > 0:
                return newline_positions[-1].item()
        except:
            pass

        # Final fallback to last token
        return len(input_ids) - 1
