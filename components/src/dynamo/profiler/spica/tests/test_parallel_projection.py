# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.profiler.spica.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from dynamo.profiler.spica.parallel_projection import (
    AGG_ATTENTION_MODE,
    AGG_FFN_MODE,
    AGG_GPUS_PER_ENGINE,
    DECODE_ATTENTION_MODE,
    DECODE_FFN_MODE,
    DECODE_GPUS_PER_ENGINE,
    PREFILL_ATTENTION_MODE,
    PREFILL_FFN_MODE,
    PREFILL_GPU_SHARE,
    PREFILL_GPUS_PER_ENGINE,
    USED_GPU_RATIO,
    ParallelConfigProjector,
)
from dynamo.profiler.spica.search_space import BranchSpace


def _role(
    *, gpus: int, attention: str, ffn: str, replicas: int
) -> ReplicaParallelConfig:
    tp, dp = (gpus, 1) if attention == "tp" else (1, gpus)
    moe_tp, moe_ep = (gpus, 1) if ffn == "tp" else (1, gpus)
    return ReplicaParallelConfig(
        shape=ParallelShape(tp=tp, dp=dp, moe_tp=moe_tp, moe_ep=moe_ep),
        replicas=replicas,
    )


def _branch(mode, configs, support, *, budget=32):
    return BranchSpace(
        deployment_mode=mode,
        parallel_configs=tuple(configs),
        supported_backends=support,
        knob_choices={"backend": sorted(set().union(*support.values()))},
        gpu_budget=budget,
    )


def test_agg_parameters_have_structural_defaults():
    configs = [
        _role(gpus=1, attention="tp", ffn="tp", replicas=8),
        _role(gpus=4, attention="tp", ffn="ep", replicas=4),
        _role(gpus=16, attention="dp", ffn="ep", replicas=2),
    ]
    branch = _branch(
        "agg", configs, {config: frozenset({"vllm"}) for config in configs}
    )

    projector = ParallelConfigProjector(branch)
    parameters = {parameter.name: parameter for parameter in projector.parameters}

    assert parameters[USED_GPU_RATIO].default == 1.0
    assert parameters[AGG_GPUS_PER_ENGINE].values == (1.0, 4.0, 16.0)
    assert parameters[AGG_GPUS_PER_ENGINE].default == 4.0
    assert parameters[AGG_GPUS_PER_ENGINE].log_scale
    assert parameters[AGG_ATTENTION_MODE].default == "tp"
    assert parameters[AGG_FFN_MODE].default == "ep"


def test_agg_exact_valid_point_projects_to_itself():
    tep4 = _role(gpus=4, attention="tp", ffn="ep", replicas=4)
    dep8 = _role(gpus=8, attention="dp", ffn="ep", replicas=2)
    branch = _branch(
        "agg", [tep4, dep8], {tep4: frozenset({"vllm"}), dep8: frozenset({"vllm"})}
    )
    projector = ParallelConfigProjector(branch)

    projection = projector.project(
        {
            USED_GPU_RATIO: 0.5,
            AGG_GPUS_PER_ENGINE: 8,
            AGG_ATTENTION_MODE: "dp",
            AGG_FFN_MODE: "ep",
        },
        "vllm",
    )

    assert projection.config == dep8
    assert projection.distance == 0.0
    assert not projection.mode_projected


def test_projection_hard_filters_backend_then_snaps_worker_size():
    vllm4 = _role(gpus=4, attention="tp", ffn="ep", replicas=4)
    sglang8 = _role(gpus=8, attention="tp", ffn="ep", replicas=2)
    branch = _branch(
        "agg",
        [vllm4, sglang8],
        {vllm4: frozenset({"vllm"}), sglang8: frozenset({"sglang"})},
    )
    projector = ParallelConfigProjector(branch)

    projection = projector.project(
        {
            USED_GPU_RATIO: 0.5,
            AGG_GPUS_PER_ENGINE: 8,
            AGG_ATTENTION_MODE: "tp",
            AGG_FFN_MODE: "ep",
        },
        "vllm",
    )

    assert projection.config == vllm4
    assert not projection.mode_projected


def test_projection_falls_back_when_requested_mode_has_no_valid_config():
    dep8 = _role(gpus=8, attention="dp", ffn="ep", replicas=2)
    branch = _branch("agg", [dep8], {dep8: frozenset({"vllm"})})
    projector = ParallelConfigProjector(branch)

    projection = projector.project(
        {
            USED_GPU_RATIO: 0.5,
            AGG_GPUS_PER_ENGINE: 8,
            AGG_ATTENTION_MODE: "tp",
            AGG_FFN_MODE: "ep",
        },
        "vllm",
    )

    assert projection.config == dep8
    assert projection.mode_projected


def test_disagg_projection_uses_role_features_and_joint_gpu_budget():
    balanced = DisaggParallelConfig(
        prefill=_role(gpus=4, attention="tp", ffn="ep", replicas=2),
        decode=_role(gpus=8, attention="dp", ffn="ep", replicas=1),
    )
    decode_heavy = DisaggParallelConfig(
        prefill=_role(gpus=4, attention="tp", ffn="ep", replicas=1),
        decode=_role(gpus=8, attention="dp", ffn="tp", replicas=3),
    )
    configs = [balanced, decode_heavy]
    branch = _branch(
        "disagg", configs, {config: frozenset({"sglang"}) for config in configs}
    )
    projector = ParallelConfigProjector(branch)

    projection = projector.project(
        {
            USED_GPU_RATIO: 1.0,
            PREFILL_GPU_SHARE: 0.125,
            PREFILL_GPUS_PER_ENGINE: 4,
            DECODE_GPUS_PER_ENGINE: 8,
            PREFILL_ATTENTION_MODE: "tp",
            DECODE_ATTENTION_MODE: "dp",
            PREFILL_FFN_MODE: "ep",
            DECODE_FFN_MODE: "tp",
        },
        "sglang",
    )

    assert projection.config == decode_heavy
    assert projection.config.total_gpus == 28 <= branch.gpu_budget
    assert projection.actual_features[PREFILL_GPU_SHARE] == 1 / 7


def test_single_config_makes_every_parallel_parameter_constant():
    pinned = _role(gpus=4, attention="tp", ffn="ep", replicas=8)
    branch = _branch("agg", [pinned], {pinned: frozenset({"vllm"})})

    projector = ParallelConfigProjector(branch)

    assert all(parameter.is_constant for parameter in projector.parameters)
    assert projector.project({}, "vllm").config == pinned


def test_dense_pool_does_not_expose_ffn_mode():
    dense = ReplicaParallelConfig(
        shape=ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=1),
        replicas=4,
    )
    branch = _branch("agg", [dense], {dense: frozenset({"vllm"})})

    projector = ParallelConfigProjector(branch)

    assert AGG_FFN_MODE not in {parameter.name for parameter in projector.parameters}
