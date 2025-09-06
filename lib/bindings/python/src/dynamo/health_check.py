"""
Simple health check utilities for Dynamo backends.

This module provides helper functions for working with health check payloads.
Most backends should define their payload directly in main.py.
"""

import json
import os
from typing import Any, Dict, Optional


def get_default_health_check_payload(backend: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a simple default health check payload.

    This is a helper function if you want a default payload.

    Args:
        backend: Backend type ('vllm', 'sglang', 'trtllm', etc.)

    Returns:
        A simple health check payload suitable for the backend.
    """
    if not backend:
        backend = "vllm"

    backend = backend.lower()

    if backend == "vllm":
        return {
            "prompt": "1",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    elif backend == "sglang":
        return {
            "token_ids": [1],
            "sampling_options": {
                "temperature": 0.0,
                "max_tokens": 1,
            },
            "stop_conditions": {
                "max_tokens": 1,
            },
        }
    elif backend in ["trtllm", "trt", "tensorrt"]:
        return {
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        # Default to vLLM format
        return {
            "prompt": "1",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }


def load_health_check_from_env() -> Optional[Dict[str, Any]]:
    """
    Load health check payload from DYN_HEALTH_CHECK_PAYLOAD environment variable.

    Supports two formats:
    1. JSON string: export DYN_HEALTH_CHECK_PAYLOAD='{"prompt": "test", "max_tokens": 1}'
    2. File path: export DYN_HEALTH_CHECK_PAYLOAD='@/path/to/health_check.json'

    Returns:
        Dict containing the health check payload, or None if not set.
    """
    env_value = os.environ.get("DYN_HEALTH_CHECK_PAYLOAD")
    if not env_value:
        return None

    try:
        if env_value.startswith("@"):
            # Load from file
            file_path = env_value[1:]
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            # Parse as JSON
            return json.loads(env_value)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Failed to parse DYN_HEALTH_CHECK_PAYLOAD: {e}")
        return None
