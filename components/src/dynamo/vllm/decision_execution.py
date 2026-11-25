# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advertise the qualified vLLM causal decision execution profile."""

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .args import Config

logger = logging.getLogger(__name__)
VLLM_CAUSAL_DECISION_CAPABILITY = "vllm_causal_decision_v1"
VLLM_POOLING_DECISION_CAPABILITY = "vllm_pooling_decision_v1"
_POOLING_INLINE_TEMPLATE = "joint-question-inline-readout-v1"
_POOLING_CHUNKED_TEMPLATE = "joint-chunked-readout-v1"
_MAX_POOLING_CLASSES = 255
_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Gemma4ForConditionalGeneration",
    }
)


def _unqualified_runtime_condition(runtime: "VllmConfig") -> str | None:
    model = runtime.model_config
    scheduler = runtime.scheduler_config
    expected_scheduler = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if scheduler.async_scheduling
        else "vllm.v1.core.sched.scheduler.Scheduler"
    )
    checks = (
        (
            f"unsupported model architecture {model.architecture!r}",
            model.architecture in _ARCHITECTURES,
        ),
        ("requires runner_type=generate", model.runner_type == "generate"),
        ("requires logprobs_mode=raw_logprobs", model.logprobs_mode == "raw_logprobs"),
        ("requires is_encoder_decoder=False", model.is_encoder_decoder is False),
        ("requires is_diffusion=False", model.is_diffusion is False),
        ("requires model_impl=auto or vllm", model.model_impl in ("auto", "vllm")),
        ("requires no model_class_overrides", not model.model_class_overrides),
        ("requires convert_type=none", model.convert_type == "none"),
        (
            "requires max_model_len integer >= 2",
            type(model.max_model_len) is int and model.max_model_len >= 2,
        ),
        (
            "requires max_num_seqs integer = 1",
            type(scheduler.max_num_seqs) is int and scheduler.max_num_seqs == 1,
        ),
        ("requires async_scheduling boolean", type(scheduler.async_scheduling) is bool),
        (
            "requires standard scheduler_cls",
            scheduler.scheduler_cls in (None, expected_scheduler),
        ),
        (
            "requires data_parallel_size integer = 1",
            type(runtime.parallel_config.data_parallel_size) is int
            and runtime.parallel_config.data_parallel_size == 1,
        ),
        ("requires lora_config=None", runtime.lora_config is None),
        ("requires speculative_config=None", runtime.speculative_config is None),
        ("requires kv_transfer_config=None", runtime.kv_transfer_config is None),
        ("requires ec_transfer_config=None", runtime.ec_transfer_config is None),
    )
    return next((name for name, qualified in checks if not qualified), None)


def qualify_vllm_decision_runtime(vllm_config: "VllmConfig", config: "Config") -> None:
    """Reject an unqualified resolved configuration before the engine starts."""
    if config.systemone_serial is not True:
        return
    try:
        condition = _unqualified_runtime_condition(vllm_config)
    except AttributeError as exc:
        raise ValueError(
            "--systemone-serial cannot qualify missing runtime metadata"
        ) from exc
    if condition is not None:
        raise ValueError(
            f"--systemone-serial: runtime qualification failed: {condition}"
        )


def publish_vllm_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    worker_type: WorkerType,
    config: "Config",
    *,
    native_generate_supported: bool,
    needs: list[list[WorkerType]] | None = None,
) -> bool:
    """Publish a prequalified runtime's serialized raw candidate-read contract.

    The frontend owns question rendering and answer projection. Execution uses
    one generated token per question; this flag does not establish model quality.
    """
    if config.systemone_serial is not True:
        return False
    if (
        native_generate_supported is not True
        or model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or any(needs or ())
    ):
        raise ValueError(
            "--systemone-serial requires an aggregated token-input worker with "
            "native generate support and no downstream workers"
        )
    runtime_config.set_engine_specific(VLLM_CAUSAL_DECISION_CAPABILITY, "true")
    logger.info("Published vLLM causal decision capability (serialized raw scoring)")
    return True


_POOLING_METADATA_FILES = ("decision_config.json", "joint_config.json")


def _resolve_pooling_metadata_files(
    config: "Config",
    vllm_config: "VllmConfig",
) -> tuple[Path, Path] | None:
    """Locate pooling decision metadata locally or in the HuggingFace cache."""
    model_cfg = getattr(vllm_config, "model_config", None)
    candidates = (
        getattr(model_cfg, "model", None),
        getattr(getattr(config, "engine_args", None), "model", None),
        getattr(config, "model", None),
    )
    for candidate in candidates:
        if isinstance(candidate, (str, Path)) and str(candidate):
            model_dir = Path(candidate)
            if model_dir.is_dir():
                paths = tuple(model_dir / name for name in _POOLING_METADATA_FILES)
                return paths if all(p.is_file() for p in paths) else None

    repo_id = next((c for c in candidates if isinstance(c, str) and c), None)
    if not repo_id:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.errors import HFValidationError

        revision = getattr(model_cfg, "revision", None)
        cached = [
            try_to_load_from_cache(repo_id=repo_id, filename=name, revision=revision)
            for name in _POOLING_METADATA_FILES
        ]
    except (ImportError, HFValidationError, OSError) as exc:
        logger.debug(
            "HF cache lookup for pooling metadata (%s) failed: %s", repo_id, exc
        )
        return None
    if all(isinstance(p, str) for p in cached):
        paths = (Path(cached[0]), Path(cached[1]))
        if all(p.is_file() for p in paths):
            return paths
    return None


def _optional_positive_int(obj: dict, key: str) -> int | None:
    val = obj.get(key)
    if val is not None and (type(val) is not int or val < 1):
        raise ValueError(f"joint_config.{key} must be a positive integer or null")
    return val


def _load_pooling_decision_descriptor(decision_path: Path, joint_path: Path) -> dict:
    """Load and validate a pooling-classifier readout descriptor."""
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    if not isinstance(decision, dict) or not isinstance(joint, dict):
        raise ValueError(
            "decision_config.json and joint_config.json must be JSON objects"
        )
    if decision.get("format_version") != 2:
        raise ValueError(
            f"unsupported decision format_version: {decision.get('format_version')!r}"
        )
    if decision.get("attention_mode") != "noncausal_full_attention":
        raise ValueError(
            f"unsupported decision attention_mode: {decision.get('attention_mode')!r}"
        )

    joint_version, template = joint.get("version"), joint.get("template")
    if (joint_version, template) == (3, _POOLING_INLINE_TEMPLATE):
        separator = None
    elif (joint_version, template) == (4, _POOLING_CHUNKED_TEMPLATE):
        separator = joint.get("separator")
        if not isinstance(separator, str) or not separator:
            raise ValueError(
                "joint-chunked-readout-v1 requires a non-empty string separator"
            )
    else:
        raise ValueError(
            f"unsupported joint_config version/template: {joint_version!r}/{template!r}"
        )

    marker = joint.get("marker")
    if not isinstance(marker, str) or not marker:
        raise ValueError("joint_config.marker must be a non-empty string")
    readout_offset = joint.get("readout_offset", 0)
    if type(readout_offset) is not int or not (-(2**31) <= readout_offset < 2**31):
        raise ValueError("joint_config.readout_offset must be a 32-bit integer")

    codes = decision.get("codes")
    if (
        not isinstance(codes, list)
        or not (2 <= len(codes) <= _MAX_POOLING_CLASSES)
        or any(not isinstance(c, str) or not c for c in codes)
        or len(set(codes)) != len(codes)
    ):
        raise ValueError(
            f"decision_config.codes must be 2..{_MAX_POOLING_CLASSES} unique strings"
        )

    temperature = decision.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("decision_config.temperature must be a finite positive number")

    descriptor = {
        "template": template,
        "marker": marker,
        "readout_offset": readout_offset,
        "codes": list(codes),
        "temperature": float(temperature),
        "max_questions": _optional_positive_int(joint, "max_questions"),
        "max_length": _optional_positive_int(joint, "max_length"),
    }
    if separator is not None:
        descriptor["separator"] = separator
    return descriptor


def publish_vllm_pooling_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_type: ModelType,
    worker_type: WorkerType,
    config: "Config",
    vllm_config: "VllmConfig",
    *,
    needs: list[list[WorkerType]] | None = None,
) -> bool:
    """Publish `vllm_pooling_decision_v1` for qualified classify/pooling decision workers."""
    if (
        not getattr(config, "classify_worker", False)
        or not model_type.supports_pooling()
    ):
        return False

    metadata_files = _resolve_pooling_metadata_files(config, vllm_config)
    if metadata_files is None:
        return False
    decision_path, joint_path = metadata_files
    model_label = getattr(config, "model", decision_path.parent)
    try:
        descriptor = _load_pooling_decision_descriptor(decision_path, joint_path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Skipping pooling decision capability for %s: %s", model_label, exc
        )
        return False

    model_cfg = vllm_config.model_config
    pooler_cfg = getattr(model_cfg, "pooler_config", None)
    dp_size = getattr(
        getattr(vllm_config, "parallel_config", None), "data_parallel_size", None
    )
    uses_last_classify = (
        descriptor["template"] == _POOLING_CHUNKED_TEMPLATE
        and descriptor["readout_offset"] == 0
    )
    checks = (
        (
            "requires runner_type=pooling",
            getattr(model_cfg, "runner_type", None) == "pooling",
        ),
        (
            "requires convert_type=classify",
            getattr(model_cfg, "convert_type", None) == "classify",
        ),
        (
            "requires pooler_config.use_activation=False",
            getattr(pooler_cfg, "use_activation", None) is False,
        ),
        (
            "requires pooler_config.pooling_type=LAST",
            not uses_last_classify
            or getattr(pooler_cfg, "pooling_type", None) in (None, "LAST", "last"),
        ),
        ("requires aggregated worker", worker_type == WorkerType.Aggregated),
        ("requires no downstream workers", not any(needs or ())),
        ("requires data_parallel_size=1", dp_size == 1),
        (
            "requires lora_config=None",
            getattr(vllm_config, "lora_config", None) is None,
        ),
    )
    if failed := next((reason for reason, ok in checks if not ok), None):
        logger.warning(
            "Skipping pooling decision capability for %s: %s", model_label, failed
        )
        return False

    runtime_config.set_engine_specific(
        VLLM_POOLING_DECISION_CAPABILITY, json.dumps(descriptor)
    )
    logger.info(
        "Published vLLM pooling decision capability (%d classes, marker=%r)",
        len(descriptor["codes"]),
        descriptor["marker"],
    )
    return True
