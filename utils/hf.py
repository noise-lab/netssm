"""HuggingFace utilities for loading model configurations and weights.

This module provides helper functions to load model configs and state dicts
from HuggingFace-style checkpoint directories.
"""

import json

import torch

from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    """Load model configuration from HuggingFace cache.

    Args:
        model_name: Name or path of the pretrained model.

    Returns:
        Dict containing model configuration.
    """
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, dtype=None):
    """Load model state dict from HuggingFace cache.

    Args:
        model_name: Name or path of the pretrained model.
        device: Target device for the loaded tensors.
        dtype: Target dtype for the loaded tensors.

    Returns:
        State dict with model weights.
    """
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    state_dict = torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    if device is not None:
        state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
