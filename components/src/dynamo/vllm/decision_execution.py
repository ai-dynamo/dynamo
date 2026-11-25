# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advertise the qualified vLLM 0.33 causal decision execution profile."""

import json
import logging
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType

from .decision_cache_provenance import verify_cache_sources

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


def _released_runtime() -> bool:
    try:
        installed = version("vllm")
        return isinstance(installed, str) and Version(installed).public == "0.33.0"
    except (PackageNotFoundError, InvalidVersion):
        return False


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
    """Reject an unqualified resolved configuration before the engine starts.

    Cache fields alone do not establish measurement provenance, so their installed
    source owners must match the qualified release before loading the model.
    """
    if config.systemone_serial is not True:
        return
    if not _released_runtime():
        raise ValueError("--systemone-serial requires qualified vLLM 0.33.0")
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
    if not verify_cache_sources():
        raise ValueError("--systemone-serial requires qualified vLLM cache sources")


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


def _resolve_pooling_metadata_files(
    config: "Config",
    vllm_config: "VllmConfig",
) -> tuple[Path, Path] | None:
    """Locate ``decision_config.json`` and ``joint_config.json`` for a model.

    Checks resolved local directories first (``vllm_config.model_config.model``,
    ``config.engine_args.model``, ``config.model``), then falls back to the
    HuggingFace cache for repo IDs using ``vllm_config.model_config.revision``.
    """
    model_cfg = getattr(vllm_config, "model_config", None)
    engine_args = getattr(config, "engine_args", None)
    candidates = (
        getattr(model_cfg, "model", None),
        getattr(engine_args, "model", None),
        getattr(config, "model", None),
    )
    for candidate in candidates:
        if isinstance(candidate, (str, Path)) and str(candidate):
            model_dir = Path(candidate)
            if model_dir.is_dir():
                decision_path = model_dir / "decision_config.json"
                joint_path = model_dir / "joint_config.json"
                if decision_path.is_file() and joint_path.is_file():
                    return decision_path, joint_path
                return None

    repo_id = next(
        (str(c) for c in candidates if isinstance(c, str) and c),
        None,
    )
    if not repo_id:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.errors import HFValidationError
    except ImportError:
        return None
    revision = getattr(model_cfg, "revision", None)
    try:
        decision_cached = try_to_load_from_cache(
            repo_id=repo_id, filename="decision_config.json", revision=revision
        )
        joint_cached = try_to_load_from_cache(
            repo_id=repo_id, filename="joint_config.json", revision=revision
        )
    except (HFValidationError, OSError) as exc:
        logger.debug(
            "HF cache lookup for pooling decision metadata (%s) failed: %s",
            repo_id,
            exc,
        )
        return None
    if isinstance(decision_cached, str) and isinstance(joint_cached, str):
        decision_path = Path(decision_cached)
        joint_path = Path(joint_cached)
        if decision_path.is_file() and joint_path.is_file():
            return decision_path, joint_path
    return None


def _load_pooling_decision_descriptor(decision_path: Path, joint_path: Path) -> dict:
    """Load and validate a generic joint-inline-readout pooling descriptor.

    Raises ``ValueError`` when the metadata files are invalid or unsupported.
    """
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
    joint_version = joint.get("version")
    template = joint.get("template")
    if joint_version == 3 and template == _POOLING_INLINE_TEMPLATE:
        separator = None
    elif joint_version == 4 and template == _POOLING_CHUNKED_TEMPLATE:
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
        or any(not isinstance(code, str) or not code for code in codes)
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

    max_questions = joint.get("max_questions")
    if max_questions is not None and (
        type(max_questions) is not int or max_questions < 1
    ):
        raise ValueError(
            "joint_config.max_questions must be a positive integer or null"
        )
    max_length = joint.get("max_length")
    if max_length is not None and (type(max_length) is not int or max_length < 1):
        raise ValueError("joint_config.max_length must be a positive integer or null")

    descriptor = {
        "template": template,
        "marker": marker,
        "readout_offset": readout_offset,
        "codes": list(codes),
        "temperature": float(temperature),
        "max_questions": max_questions,
        "max_length": max_length,
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
    """Publish a pooling-classifier decision contract when a classify worker serves
    a checkpoint with valid ``decision_config.json`` + ``joint_config.json`` metadata.
    """
    if (
        not getattr(config, "classify_worker", False)
        or not model_type.supports_pooling()
    ):
        return False

    metadata_files = _resolve_pooling_metadata_files(config, vllm_config)
    if metadata_files is None:
        return False
    decision_path, joint_path = metadata_files
    try:
        descriptor = _load_pooling_decision_descriptor(decision_path, joint_path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Skipping pooling decision capability for %s: %s",
            getattr(config, "model", decision_path.parent),
            exc,
        )
        return False

    model_cfg = vllm_config.model_config
    pooler_cfg = getattr(model_cfg, "pooler_config", None)
    use_activation = getattr(pooler_cfg, "use_activation", None)
    pooling_type = getattr(pooler_cfg, "pooling_type", None)
    dp_size = getattr(
        getattr(vllm_config, "parallel_config", None), "data_parallel_size", None
    )
    uses_last_classify = (
        descriptor["template"] == _POOLING_CHUNKED_TEMPLATE
        and descriptor["readout_offset"] == 0
    )
    pooling_type_ok = not uses_last_classify or pooling_type in (None, "LAST", "last")
    checks = (
        (
            "requires runner_type=pooling",
            getattr(model_cfg, "runner_type", None) == "pooling",
        ),
        (
            "requires convert_type=classify",
            getattr(model_cfg, "convert_type", None) == "classify",
        ),
        ("requires pooler_config.use_activation=False", use_activation is False),
        ("requires pooler_config.pooling_type=LAST", pooling_type_ok),
        ("requires aggregated worker", worker_type == WorkerType.Aggregated),
        ("requires no downstream workers", not any(needs or ())),
        ("requires data_parallel_size=1", dp_size == 1),
        (
            "requires lora_config=None",
            getattr(vllm_config, "lora_config", None) is None,
        ),
    )
    failed = next((reason for reason, ok in checks if not ok), None)
    if failed is not None:
        logger.warning(
            "Skipping pooling decision capability for %s: %s",
            getattr(config, "model", decision_path.parent),
            failed,
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
