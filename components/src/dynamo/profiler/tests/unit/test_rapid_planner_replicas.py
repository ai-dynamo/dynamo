# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests pinning the autoscale-replica contract in `_generate_dgd_from_pick`.

Background
----------
`docs/components/profiler/profiler-guide.md` documents the "Autoscale" picking
mode as picking prefill / decode engines "each with 1 replica" and letting the
planner scale at runtime. The picker (`aiconfigurator.sdk.picking.pick_autoscale`)
honours this by hardcoding `(p)workers=1` / `(d)workers=1` on every returned
row. However, AIC's generator (`module_bridge.task_config_to_generator_config`)
then rescales workers by `total_gpus // gpus_per_replica` whenever `total_gpus`
is truthy, which would silently produce N initial replicas.

`_generate_dgd_from_pick` suppresses that rescale by zeroing `task_config.total_gpus`
on the autoscale path, so the picker's `workers=1` flows through to the DGD
unchanged. These tests pin that behaviour. They mock `task_config_to_generator_config` and `generate_backend_artifacts` so the test
exercises the rescale-suppression logic without touching AIC's full stack.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from aiconfigurator.sdk.task_v2 import Task

from dynamo.profiler.rapid import _generate_dgd_from_pick
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    HardwareSpec,
    ModelCacheSpec,
    SLASpec,
    WorkloadSpec,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.aiconfigurator,
]


def _make_dgdr() -> DynamoGraphDeploymentRequestSpec:
    """Build a minimal autoscale DGDR (`totalGpus=8`, Qwen3-30B-A3B on B200)."""
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-30B-A3B",
        backend="vllm",
        image="nvcr.io/nvidia/ai-dynamo/dynamo-frontend:latest",
        hardware=HardwareSpec(gpuSku="b200_sxm", totalGpus=8, numGpusPerNode=8),
        workload=WorkloadSpec(isl=4096, osl=1024),
        sla=SLASpec(ttft=3000.0, itl=35.0),
        modelCache=ModelCacheSpec(),
    )


def _make_picked_row(
    workers: int = 1, with_total_gpus_needed: bool = False
) -> pd.DataFrame:
    """Row shaped like a picker output. Autoscale picker always emits workers=1."""
    row: dict = {
        "(p)workers": workers,
        "(p)tp": 2,
        "(p)pp": 1,
        "(p)dp": 1,
        "(p)moe_tp": 2,
        "(p)moe_ep": 1,
        "(p)bs": 1,
        "(d)workers": workers,
        "(d)tp": 2,
        "(d)pp": 1,
        "(d)dp": 1,
        "(d)moe_tp": 1,
        "(d)moe_ep": 2,
        "(d)bs": 256,
        "ttft": 2800.0,
        "tpot": 16.7,
    }
    if with_total_gpus_needed:
        row["total_gpus_needed"] = 4
    return pd.DataFrame([row])


def _make_task() -> Task:
    """Build a disagg `Task` with `total_gpus=8` matching `_make_dgdr()`.

    v2 disagg forbids top-level worker fields; fan the shared worker spec out to
    both prefill_* and decode_* roles.
    """
    return Task(
        serving_mode="disagg",
        prefill_model_path="Qwen/Qwen3-30B-A3B",
        decode_model_path="Qwen/Qwen3-30B-A3B",
        prefill_system_name="b200_sxm",
        decode_system_name="b200_sxm",
        prefill_backend_name="vllm",
        decode_backend_name="vllm",
        total_gpus=8,
        isl=4096,
        osl=1024,
        ttft=3000.0,
        tpot=16.7,
    )


def _capture_aic_call(dgdr, row, picking_mode, chosen_exp="disagg") -> dict:
    """Drive `_generate_dgd_from_pick` and capture what AIC actually saw."""
    captured: dict = {}
    task = _make_task()

    def fake_task_config_to_generator_config(
        task_config, result_df, generator_overrides=None, **_kw
    ):
        """Mock for AIC's generator entrypoint — records the inputs AIC would have seen."""
        captured["total_gpus"] = task_config.total_gpus
        captured["overrides"] = generator_overrides
        return {}

    with (
        patch(
            "dynamo.profiler.rapid.task_config_to_generator_config",
            side_effect=fake_task_config_to_generator_config,
        ),
        patch(
            "dynamo.profiler.rapid.generate_backend_artifacts",
            return_value={"k8s_deploy.yaml": ""},
        ),
    ):
        _generate_dgd_from_pick(dgdr, row, chosen_exp, {chosen_exp: task}, picking_mode)
    captured["restored_total_gpus"] = task.total_gpus
    return captured


class TestAutoscaleSuppressesRescale:
    """Autoscale path must zero total_gpus so AIC's budget-fill rescale is a no-op."""

    def test_autoscale_zeros_total_gpus_inside_aic_call(self):
        """Primary assertion: AIC sees `task_config.total_gpus == 0` on the autoscale path."""
        captured = _capture_aic_call(_make_dgdr(), _make_picked_row(), "autoscale")
        assert captured["total_gpus"] == 0

    def test_autoscale_restores_total_gpus_after_call(self):
        """The try/finally must restore tc.total_gpus so the Task is reusable."""
        captured = _capture_aic_call(_make_dgdr(), _make_picked_row(), "autoscale")
        assert captured["restored_total_gpus"] == 8

    def test_load_match_preserves_total_gpus(self):
        """Non-autoscale picker keeps total_gpus → AIC's existing rescale runs."""
        captured = _capture_aic_call(
            _make_dgdr(),
            _make_picked_row(with_total_gpus_needed=True),
            "load_match",
        )
        # clamp_total_gpus_to_budget caps at the DGDR budget (8); a needed=4 row
        # is below budget so total_gpus becomes 4.
        assert captured["total_gpus"] == 4

    def test_default_picking_mode_omitted_is_non_autoscale(self):
        """Regression: `picking_mode` carries a non-autoscale default, so direct
        callers may omit it without a TypeError. The default must NOT take the
        autoscale path — total_gpus is preserved (clamped to budget), never zeroed.
        """
        dgdr = _make_dgdr()
        row = _make_picked_row(with_total_gpus_needed=True)
        task = _make_task()
        captured: dict = {}

        def fake_task_config_to_generator_config(
            task_config, result_df, generator_overrides=None, **_kw
        ):
            captured["total_gpus"] = task_config.total_gpus
            return {}

        with (
            patch(
                "dynamo.profiler.rapid.task_config_to_generator_config",
                side_effect=fake_task_config_to_generator_config,
            ),
            patch(
                "dynamo.profiler.rapid.generate_backend_artifacts",
                return_value={"k8s_deploy.yaml": ""},
            ),
        ):
            # Omit picking_mode entirely to exercise the default.
            _generate_dgd_from_pick(dgdr, row, "disagg", {"disagg": task})

        # needed=4 ≤ budget 8 → clamp is a no-op, total_gpus becomes 4 (not 0).
        assert captured["total_gpus"] == 4

    def test_autoscale_keeps_k8s_overrides(self):
        """Suppressing total_gpus must not drop K8sConfig overrides."""
        dgdr = _make_dgdr()
        dgdr.modelCache = ModelCacheSpec(
            pvcName="qwen3-30b-a3b",
            pvcMountPath="/opt/models",
            pvcModelPath="hub/models--Qwen--Qwen3-30B-A3B/snapshots/abc",
        )
        captured = _capture_aic_call(dgdr, _make_picked_row(), "autoscale")
        overrides = captured["overrides"]
        assert overrides is not None
        assert overrides.get("K8sConfig", {}).get("k8s_pvc_name") == "qwen3-30b-a3b"
        assert captured["total_gpus"] == 0
