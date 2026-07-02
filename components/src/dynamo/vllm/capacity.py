# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY = "reasoning_aware_guided_decoding"

# vLLM 0.24 registers Granite as a response reasoning parser, but that class
# does not implement any token-boundary method used by structured output to
# detect the end of reasoning. Advertising delayed grammar for it would leave
# required/named guidance permanently inactive. Fail closed until the native
# parser exposes a real boundary signal.
_PARSERS_WITHOUT_GUIDED_REASONING_BOUNDARY = frozenset({"granite"})

_DYNAMO_TO_VLLM_REASONING_PARSER = {
    "basic": "qwen3",
    "deepseek_r1": "deepseek_r1",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v3_1": "deepseek_v3",
    "deepseek_v3_2": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "deepseek-v4": "deepseek_v4",
    "deepseekv4": "deepseek_v4",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "glm45": "glm45",
    "gpt_oss": "openai_gptoss",
    "granite": "granite",
    "kimi_k25": "kimi_k2",
    "minimax_append_think": "minimax_m2_append_think",
    "minimax_m3": "minimax_m3",
    "minimax-m3": "minimax_m3",
    "mistral": "mistral",
    "nemotron_deci": "qwen3",
    "nemotron_nano": "nemotron_v3",
    "nemotron3": "nemotron_v3",
    "nemotron_v3": "nemotron_v3",
    "qwen3": "qwen3",
    "step3": "step3",
}


_DYNAMO_TO_VLLM_TOOL_PARSER = {
    # Dynamo's Qwen 2.5 parser uses Hermes' <tool_call> JSON wire format.
    "qwen25": "hermes",
    # Dynamo's Harmony parser is vLLM's GPT-OSS/OpenAI parser.
    "harmony": "openai",
    "deepseek_v3_1": "deepseek_v31",
    "deepseek_v3_2": "deepseek_v32",
    "deepseek-v4": "deepseek_v4",
    "deepseekv4": "deepseek_v4",
    "gemma-4": "gemma4",
    "minimax_m3": "minimax_m3",
    "minimax-m3": "minimax_m3",
    "minimax_m3_nom": "minimax_m3",
    "minimax-m3-nom": "minimax_m3",
    # Nemotron Nano emits Qwen3-Coder XML tool calls.
    "nemotron_nano": "qwen3_coder",
    "phi4": "phi4_mini_json",
}


def normalize_vllm_tool_parser(parser: str | None) -> str | None:
    if not parser:
        return parser
    normalized = parser.strip().lower()
    return _DYNAMO_TO_VLLM_TOOL_PARSER.get(normalized, normalized)


def normalize_vllm_reasoning_parser(parser: str | None) -> str | None:
    if not parser:
        return parser
    return _DYNAMO_TO_VLLM_REASONING_PARSER.get(parser.strip().lower(), parser.strip())


def reasoning_parsers_compatible(
    native_parser: str | None, dynamo_parser: str | None
) -> bool:
    if not native_parser or not dynamo_parser:
        return True
    expected = normalize_vllm_reasoning_parser(dynamo_parser)
    return expected is not None and native_parser.strip().lower() == expected


def supports_reasoning_aware_guided_decoding(
    vllm_config: Any, dynamo_reasoning_parser: str | None = None
) -> bool:
    """Whether vLLM defers guided decoding until native reasoning ends."""
    if not dynamo_reasoning_parser:
        return False
    structured_outputs_config = getattr(vllm_config, "structured_outputs_config", None)
    native_parser = getattr(structured_outputs_config, "reasoning_parser", "")
    return (
        bool(native_parser)
        and native_parser.strip().lower()
        not in _PARSERS_WITHOUT_GUIDED_REASONING_BOUNDARY
        and reasoning_parsers_compatible(native_parser, dynamo_reasoning_parser)
        and getattr(structured_outputs_config, "enable_in_reasoning", None) is False
    )


def reasoning_aware_guided_decoding_runtime_data(
    vllm_config: Any, dynamo_reasoning_parser: str | None = None
) -> dict[str, bool] | None:
    if not supports_reasoning_aware_guided_decoding(
        vllm_config, dynamo_reasoning_parser
    ):
        return None
    return {REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True}


def publish_reasoning_aware_guided_decoding(
    runtime_config: Any,
    vllm_config: Any,
    dynamo_reasoning_parser: str | None = None,
) -> bool:
    """Publish the capability through the JSON-string Python binding."""
    runtime_data = reasoning_aware_guided_decoding_runtime_data(
        vllm_config, dynamo_reasoning_parser
    )
    if runtime_data is None:
        return False
    runtime_config.set_engine_specific(
        REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY, json.dumps(True)
    )
    return True


def per_rank_kv_blocks(
    total_kv_blocks: int | None, data_parallel_size: int
) -> int | None:
    if total_kv_blocks is None:
        return None

    if data_parallel_size <= 1 or total_kv_blocks <= 0:
        return total_kv_blocks

    per_rank = total_kv_blocks // data_parallel_size
    if per_rank == 0:
        logger.warning(
            "vLLM reported fewer total KV blocks (%s) than DP ranks (%s); "
            "publishing 1 block per rank",
            total_kv_blocks,
            data_parallel_size,
        )
        return 1

    remainder = total_kv_blocks % data_parallel_size
    if remainder:
        logger.warning(
            "vLLM reported aggregate KV blocks (%s) not divisible by DP ranks (%s); "
            "publishing floor per-rank capacity %s",
            total_kv_blocks,
            data_parallel_size,
            per_rank,
        )

    return per_rank


def get_metrics_model_name(config: Any) -> str:
    return str(
        getattr(config, "served_model_name", None) or getattr(config, "model", "")
    )


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def get_spec_decode_runtime_data(
    config: Any, vllm_config: Any
) -> dict[str, Any] | None:
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        engine_args = getattr(config, "engine_args", None)
        spec_config = getattr(engine_args, "speculative_config", None)
    spec = _as_mapping(spec_config)
    if not spec:
        return None

    try:
        nextn = int(spec.get("num_speculative_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if nextn <= 0:
        return None

    data: dict[str, Any] = {"nextn": nextn, "source": "backend_config"}
    method = spec.get("method")
    if method:
        data["method"] = str(method)
    return data
