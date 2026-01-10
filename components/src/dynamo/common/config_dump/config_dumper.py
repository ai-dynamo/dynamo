# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import dataclasses
import functools
import json
import logging
import pathlib
from enum import Enum
from typing import Any, Dict, Optional

from .environment import get_environment_vars
from .system_info import (
    get_gpu_info,
    get_package_info,
    get_runtime_info,
    get_system_info,
)

logger = logging.getLogger(__name__)


def _get_sglang_version() -> Optional[str]:
    """Get SGLang version if available.

    Returns:
        Version string if SGLang is installed, None otherwise.
    """
    try:
        import sglang as sgl

        return sgl.__version__
    except ImportError:
        logger.debug("SGLang not available")
        return None
    except AttributeError:
        logger.warning("SGLang installed but version not available")
        return None


def _get_trtllm_version() -> Optional[str]:
    """Get TensorRT-LLM version if available.

    Returns:
        Version string if TensorRT-LLM is installed, None otherwise.
    """
    try:
        import tensorrt_llm

        return tensorrt_llm.__version__
    except ImportError:
        logger.debug("TensorRT-LLM not available")
        return None
    except AttributeError:
        logger.warning("TensorRT-LLM installed but version not available")
        return None


def _get_vllm_version() -> Optional[str]:
    """Get vLLM version if available.

    Returns:
        Version string if vLLM is installed, None otherwise.
    """
    try:
        import vllm

        return vllm.__version__
    except ImportError:
        logger.debug("vLLM not available")
        return None
    except AttributeError:
        logger.warning("vLLM installed but version not available")
        return None


def _get_dynamo_version() -> str:
    """Get Dynamo version."""
    try:
        from dynamo.common import __version__
    except Exception:
        __version__ = "0.0.0+unknown"

    return __version__


def dump_config(dump_config_to: Optional[str], config: Any) -> None:
    """
    Dump the configuration to a file or stdout.

    If dump_config_to is not provided, the config will be logged to stdout at VERBOSE level.

    Args:
        dump_config_to: Optional path to dump the config to. If None, logs to stdout.
        config: The configuration object to dump (must be JSON-serializable).

    Raises:
        Logs errors but does not raise exceptions to ensure graceful degradation.
    """

    if dump_config_to:
        config_dump_payload = get_config_dump(config)
        try:
            dump_path = pathlib.Path(dump_config_to)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_path.resolve(), "w", encoding="utf-8") as f:
                f.write(config_dump_payload)
            logger.info(f"Dumped config to {dump_path.resolve()}")
        except (OSError, IOError):
            logger.exception(
                f"Failed to dump config to {dump_config_to}, dropping to stdout"
            )
            logger.info(f"CONFIG_DUMP: {config_dump_payload}")
        except Exception:
            logger.exception("Unexpected error dumping config, dropping to stdout")
            logger.info(f"CONFIG_DUMP: {config_dump_payload}")
    elif logger.getEffectiveLevel() <= logging.DEBUG:
        # only collect/dump config if the logger is at DEBUG level or lower
        config_dump_payload = get_config_dump(config)
        logger.debug(f"CONFIG_DUMP: {config_dump_payload}")


def get_config_dump(config: Any, extra_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Collect comprehensive config information about a backend instance.

    Args:
        config: Any JSON-serializable object containing the backend configuration.
        extra_info: Optional dict of additional information to include in the dump.

    Returns:
        JSON string containing comprehensive information.

    Note:
        Returns error information if collection fails, ensuring some diagnostic data is always available.
    """
    if extra_info is None:
        extra_info = {}
    try:
        config_dump = {
            "system_info": get_system_info(),
            "environment": get_environment_vars(),
            "config": config,
            "runtime_info": get_runtime_info(),
            "dynamo_version": _get_dynamo_version(),
            "gpu_info": get_gpu_info(),
            "installed_packages": get_package_info(),
        }

        # Add common versions
        if ver := _get_sglang_version():
            config_dump["sglang_version"] = ver
        if ver := _get_trtllm_version():
            config_dump["trtllm_version"] = ver
        if ver := _get_vllm_version():
            config_dump["vllm_version"] = ver

        # Add any extra information provided by the caller
        if extra_info:
            config_dump.update(extra_info)

        return canonical_json_encoder.encode(config_dump)

    except Exception as e:
        logger.error(f"Error collecting config dump: {e}")
        # Return a basic error response with at least system info
        error_info = {
            "error": f"Failed to collect config dump: {str(e)}",
            "system_info": get_system_info(),  # Always try to include basic system info
        }
        return canonical_json_encoder.encode(error_info)


def add_config_dump_args(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser to dump the config to a file.

    Args:
        parser: The parser to add the arguments to
    """
    parser.add_argument(
        "--dump-config-to",
        type=str,
        default=None,
        help="Dump config to the specified file path. If not specified, the config will be dumped to stdout at INFO level.",
    )


try:
    # trtllm uses pydantic, but it's not a hard dependency
    import pydantic

    def try_process_pydantic(obj: Any) -> Optional[dict]:
        if isinstance(obj, pydantic.BaseModel):
            return obj.model_dump()
        return None

except ImportError:

    def try_process_pydantic(obj: Any) -> Optional[dict]:
        return None


@functools.singledispatch
def _preprocess_for_encode(obj: object) -> object:
    """
    Single dispatch function for preprocessing objects before JSON encoding.

    This function should be extended using @register_encoder decorator
    for backend-specific types.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    if (result := try_process_pydantic(obj)) is not None:
        return result

    logger.warning(f"Unknown type {type(obj)}, using __dict__ or str(obj)")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def register_encoder(type_class):
    """
    Decorator to register custom encoders for specific types.

    Usage:
        @register_encoder(MyClass)
        def encode_my_class(obj: MyClass):
            return {"field": obj.field}
    """
    logger.debug(f"Registering encoder for {type_class}")
    return _preprocess_for_encode.register(type_class)


@register_encoder(set)
def _preprocess_for_encode_set(
    obj: set,
) -> list:  # pyright: ignore[reportUnusedFunction]
    return sorted(list(obj))


@register_encoder(argparse.Namespace)
def _preprocess_for_encode_namespace(
    obj: argparse.Namespace,
) -> dict:  # pyright: ignore[reportUnusedFunction]
    return obj.__dict__


@register_encoder(Enum)
def _preprocess_for_encode_enum(
    obj: Enum,
) -> str:  # pyright: ignore[reportUnusedFunction]
    return str(obj)


# Create a canonical JSON encoder with consistent formatting
canonical_json_encoder = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    allow_nan=False,
    sort_keys=True,
    indent=None,
    default=_preprocess_for_encode,
)
