# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ParallelizationMapping:
    """
    Represents parallelization mapping of configs
    """

    tp: int | None = None
    tep: int | None = None
    dep: int | None = None

    def label(self, phase: str) -> str:
        if self.tp is not None:
            return f"TP={self.tp}"
        if phase == "prefill" and self.tep is not None:
            return f"TEP={self.tep}"
        if phase == "decode" and self.dep is not None:
            return f"DEP={self.dep}"
        return "default"


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
        if phase == "prefill":
            candidates = [ParallelizationMapping(tep=num_gpus)]
        else:
            candidates = [ParallelizationMapping(dep=num_gpus)]
    else:
        candidates = [ParallelizationMapping(tp=num_gpus)]

    # now verify if the candidates are valid
    verified: list[ParallelizationMapping] = []
    for m in candidates:

        # 1) KV heads divisibility checks
        if m.tp is not None:
            if num_kv_heads is None:
                logger.warning(
                    f"Skipping KV heads divisibility check for TP={m.tp}: num_kv_heads is unknown"
                )
            else:
                if int(num_kv_heads) % int(m.tp) != 0:
                    logger.warning(
                        f"Invalid mapping TP={m.tp}: num_kv_heads={num_kv_heads} not divisible by TP"
                    )
                    continue

        if m.tep is not None:
            if num_kv_heads is None:
                logger.warning(
                    f"Skipping KV heads divisibility check for TEP={m.tep}: num_kv_heads is unknown"
                )
            else:
                if int(num_kv_heads) % int(m.tep) != 0:
                    logger.warning(
                        f"Invalid mapping TEP={m.tep}: num_kv_heads={num_kv_heads} not divisible by TEP"
                    )
                    continue

        # 2) Experts divisibility checks (for MoE)
        if m.tep is not None:
            if num_experts is None:
                logger.warning(
                    f"Skipping experts divisibility check for TEP={m.tep}: num_experts is unknown"
                )
            else:
                if int(num_experts) % int(m.tep) != 0:
                    logger.warning(
                        f"Invalid mapping TEP={m.tep}: num_experts={num_experts} not divisible by TEP"
                    )
                    continue

        if m.dep is not None:
            if num_experts is None:
                logger.warning(
                    f"Skipping experts divisibility check for DEP={m.dep}: num_experts is unknown"
                )
            else:
                if int(num_experts) % int(m.dep) != 0:
                    logger.warning(
                        f"Invalid mapping DEP={m.dep}: num_experts={num_experts} not divisible by DEP"
                    )
                    continue

        # 3) Intermediate size vs quantization block checks
        # Always check: intermediate_size % parallel_size == 0 when intermediate_size is known
        # Additionally (if quant_block known): (intermediate_size // parallel_size) divides quant_block if quant_block is known
        # Applies to TP and TEP only
        if intermediate_size is not None:
            parallel_size = None
            tag = None
            if m.tp is not None:
                parallel_size = int(m.tp)
                tag = "TP"
            elif m.tep is not None:
                parallel_size = int(m.tep)
                tag = "TEP"

            if parallel_size is not None and parallel_size > 0:
                I = int(intermediate_size)
                if I % parallel_size != 0:
                    logger.warning(
                        f"Invalid mapping {tag}={parallel_size}: intermediate_size={I} not divisible by {tag}"
                    )
                    continue
                if quant_block is not None:
                    per_shard = I // parallel_size
                    Q = int(quant_block)
                    if Q % per_shard != 0:
                        logger.warning(
                            f"Invalid mapping {tag}={parallel_size}: (intermediate_size // {tag})={per_shard} does not divide quantization block {Q}"
                        )
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
    cfg = base_config
    if mapping.tp is not None:
        cfg = config_modifier.set_config_tp_size(cfg, mapping.tp)
    elif phase == "prefill" and mapping.tep is not None:
        cfg = config_modifier.set_config_tep_size(cfg, mapping.tep, num_gpus_per_node)
    elif phase == "decode" and mapping.dep is not None:
        cfg = config_modifier.set_config_dep_size(cfg, mapping.dep, num_gpus_per_node)
    else:
        pass
    return cfg


