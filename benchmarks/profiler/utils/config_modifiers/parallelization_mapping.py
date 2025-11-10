# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from dataclasses import dataclass
from enum import Enum

from benchmarks.profiler.utils.defaults import PREFILL_MAX_NUM_TOKENS, EngineType
from benchmarks.profiler.utils.model_info import ModelInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ParallelizationStrategy(Enum):
    """Enum for parallelization strategy types."""

    TP = "TP"
    TEP = "TEP"
    DEP = "DEP"


@dataclass(frozen=True)
class ParallelizationMapping:
    """
    Represents parallelization mapping of configs
    """

    tp: int | None = None
    tep: int | None = None
    dep: int | None = None

    def label(self) -> str:
        if self.tp is not None:
            return f"{ParallelizationStrategy.TP.value}={self.tp}"
        if self.tep is not None:
            return f"{ParallelizationStrategy.TEP.value}={self.tep}"
        if self.dep is not None:
            return f"{ParallelizationStrategy.DEP.value}={self.dep}"
        return "default"

    def get_tp_size(self) -> int:
        """
        Get the effective TP size for KV heads splitting.
        Both TP and TEP split KV heads, DEP doesn't (returns 1).
        """
        if self.tp is not None:
            return self.tp
        if self.tep is not None:
            return self.tep
        return 1  # DEP has TP split of 1

    def get_expert_split(self) -> int:
        """
        Get the effective expert split size.
        Both TEP and DEP split experts, TP doesn't (returns 1).
        """
        if self.tep is not None:
            return self.tep
        if self.dep is not None:
            return self.dep
        return 1  # TP has expert split of 1

    def get_attn_dp_size(self) -> int:
        """
        Get the attention data parallelism size.
        DEP uses data parallelism for attention (returns dep size).
        TP and TEP don't use data parallelism for attention (returns 1).

        Args:
            None

        Returns:
            The attention data parallelism size
        """
        return self.dep if self.dep is not None else 1  # TP and TEP → 1


def _check_divisibility(
    value: int | None,
    divisor: int,
    value_name: str,
    divisor_name: str,
    mapping_label: str,
) -> bool:
    """
    Check if value is divisible by divisor.
    Returns True if valid (or value is None), False if invalid.

    Args:
        value: The value to check (e.g., num_kv_heads, num_experts)
        divisor: The divisor to check against
        value_name: Name of the value for error messages
        divisor_name: Name of the divisor for error messages (e.g., "tp_size", "expert_split")
        mapping_label: Label of the mapping for error messages
    """
    if value is None:
        logger.warning(
            f"Skipping {value_name} divisibility check for {mapping_label}: {value_name} is unknown"
        )
        return True

    if divisor > 1 and int(value) % divisor != 0:
        logger.warning(
            f"Invalid mapping {mapping_label}: {value_name}={value} not divisible by {divisor_name}={divisor}"
        )
        return False

    return True


def _validate_intermediate_size(
    mapping: ParallelizationMapping,
    intermediate_size: int | None,
    quant_block: int | None,
) -> bool:
    """
    Validate intermediate size and quantization block for TP and TEP strategies.
    Checks:
    - intermediate_size % tp_size == 0
    - (intermediate_size // tp_size) divides quant_block (if quant_block is known)
    """
    tp_size = mapping.get_tp_size()

    # Check basic divisibility
    if not _check_divisibility(
        intermediate_size, tp_size, "intermediate_size", "tp_size", mapping.label()
    ):
        return False

    # Additional check for quantization block constraint
    if intermediate_size is not None and quant_block is not None and tp_size > 1:
        per_shard = int(intermediate_size) // tp_size
        if not _check_divisibility(
            per_shard, quant_block, "per_shard", "quant_block", mapping.label()
        ):
            return False

    return True


def get_candidate_parallel_mappings(
    num_gpus: int, model_info: ModelInfo, phase: str
) -> list[ParallelizationMapping]:
    """
    Return a list of candidate parallelization mappings for a given GPU count and phase,
    verified against model properties.

    Verification rules:
    - TP and TEP must divide num_kv_heads (if available)
    - TEP and DEP must divide num_experts (if available)
    """
    is_moe = bool(model_info.is_moe)
    num_kv_heads = model_info.num_kv_heads
    num_experts = model_info.num_experts
    intermediate_size = model_info.intermediate_size
    quant_block = model_info.quantization_block_size

    candidates: list[ParallelizationMapping] = []
    if is_moe:
        if phase == EngineType.PREFILL:
            candidates = [
                ParallelizationMapping(tep=num_gpus),
                ParallelizationMapping(dep=num_gpus),
            ]
        elif phase == EngineType.DECODE:
            candidates = [
                ParallelizationMapping(dep=num_gpus),
                ParallelizationMapping(tep=num_gpus),
            ]
    else:
        candidates = [ParallelizationMapping(tp=num_gpus)]

    # Verify candidates against model constraints
    verified: list[ParallelizationMapping] = []
    for m in candidates:
        # Check KV heads divisibility
        if not _check_divisibility(
            num_kv_heads, m.get_tp_size(), "num_kv_heads", "tp_size", m.label()
        ):
            continue

        # Check experts divisibility
        if not _check_divisibility(
            num_experts, m.get_expert_split(), "num_experts", "expert_split", m.label()
        ):
            continue

        # Check intermediate size and quantization block
        if not _validate_intermediate_size(m, intermediate_size, quant_block):
            continue

        verified.append(m)

    return verified


def apply_parallel_mapping_to_config(
    base_config: dict,
    mapping: ParallelizationMapping,
    phase: str,
    config_modifier,
    num_gpus_per_node: int | None,
) -> dict:
    cfg = copy.deepcopy(base_config)
    if mapping.tp is not None:
        cfg = config_modifier.set_config_tp_size(cfg, mapping.tp)
    elif mapping.tep is not None:
        cfg = config_modifier.set_config_tep_size(cfg, mapping.tep, num_gpus_per_node)
    elif mapping.dep is not None:
        cfg = config_modifier.set_config_dep_size(cfg, mapping.dep, num_gpus_per_node)
    else:
        raise ValueError(f"Invalid mapping: {mapping.label()}")

    # for prefill,set batch size to attention_dp_size
    # (this assume prompt is long enough to saturate the GPU, which is usually valid in disagg)
    if phase == EngineType.PREFILL:
        cfg = config_modifier.set_prefill_config(
            cfg,
            max_batch_size=mapping.get_attn_dp_size(),
            max_num_tokens=PREFILL_MAX_NUM_TOKENS,
        )
    return cfg
