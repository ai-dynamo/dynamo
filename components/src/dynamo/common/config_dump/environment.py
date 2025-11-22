# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Set

# Default environment variable prefixes to capture
# These cover common ML/GPU/Dynamo-related configurations
DEFAULT_ENV_PREFIXES = [
    "DYN_",  # Dynamo-specific variables
    "CUDA_",  # CUDA configuration
    "NCCL_",  # NVIDIA Collective Communications Library
    "HF_",  # HuggingFace
    "TRANSFORMERS_",  # Transformers library
    "SGLANG_",  # SGLang
    "SGL_",  # SGLang (short prefix)
    "MC_",  # Mooncake
    "VLLM_",  # vLLM
    "TENSORRT_",  # TensorRT
    "TORCH_",  # PyTorch
    "UCX_",  # UCX
    "NIXL_",  # NIXL
    "OMPI_",  # OpenMPI
    "LLM_",  # Misc trtllm variables
    "TLLM_",
    "TRT_LLM_",
    "TRTLLM_",
    "NVIDIA_",
    "NSYS_",
    "GENERATE_CU_",
    "OVERRIDE_",
    "TOKENIZERS_",
    "DISABLE_TORCH_",
    "PYTORCH_",
    "ENABLE_PERFECT_ROUTER",
    "FLA_",
    "NEMOTRON_",
]

# Sensitive variable patterns to redact (case-insensitive)
SENSITIVE_PATTERNS = [
    "TOKEN",
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
]


def get_environment_vars(
    prefixes: Optional[List[str]] = None,
    include_sensitive: bool = False,
    additional_vars: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """
    Get relevant environment variables based on prefixes.

    Args:
        prefixes: List of environment variable prefixes to capture.
                  If None, uses DEFAULT_ENV_PREFIXES.
        include_sensitive: If False, redacts values of potentially sensitive variables.
                          Default is False for security.
        additional_vars: Set of specific variable names to include regardless of prefix.

    Returns:
        Dictionary of environment variable names to values.
        Sensitive values are replaced with "<REDACTED>" unless include_sensitive is True.

    Examples:
        >>> get_environment_vars()  # Uses default prefixes
        >>> get_environment_vars(prefixes=["MY_APP_"])  # Custom prefixes only
        >>> get_environment_vars(additional_vars={"PATH", "HOME"})  # Include specific vars
    """
    if prefixes is None:
        prefixes = DEFAULT_ENV_PREFIXES

    if additional_vars is None:
        additional_vars = set()

    relevant_env_vars = {}

    for key, value in os.environ.items():
        # Check if matches prefix or is in additional_vars
        if any(key.startswith(prefix) for prefix in prefixes) or key in additional_vars:
            # Redact sensitive values unless explicitly requested
            if not include_sensitive and _is_sensitive(key):
                relevant_env_vars[key] = "<REDACTED>"
            else:
                relevant_env_vars[key] = value

    return relevant_env_vars


def _is_sensitive(var_name: str) -> bool:
    """
    Check if an environment variable name suggests it contains sensitive data.

    Args:
        var_name: The environment variable name to check.

    Returns:
        True if the variable name matches sensitive patterns.
    """
    var_name_upper = var_name.upper()
    return any(pattern in var_name_upper for pattern in SENSITIVE_PATTERNS)
