# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests pinning backend consistency in `_generate_dgd_from_pick`.

A generated deployment names a backend twice: once as the container image
derived in `_build_k8s_overrides`, and once as the ``backend`` argument that
selects which command and argument generator `generate_backend_artifacts`
runs. Both must name the backend that the winning configuration row reports,
otherwise the deployment pulls one runtime image and runs another runtime's
command line.
"""

import logging
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from aiconfigurator.sdk.task_v2 import Task

    from dynamo.profiler.rapid import _generate_dgd_from_pick, _generated_backend
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        HardwareSpec,
        SLASpec,
        WorkloadSpec,
    )
except ImportError as e:  # pragma: no cover - environment guard
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


_MODEL = "Qwen/Qwen3-32B"
_REGISTRY = "nvcr.io/nvidia/ai-dynamo"
_REQUEST_IMAGE = f"{_REGISTRY}/dynamo-planner:1.2.3"
_RAPID_LOGGER = "dynamo.profiler.rapid"

# A non-empty artifact: _generate_dgd_from_pick returns None for both an empty
# artifact and a refusal.
_ARTIFACT_YAML = "kind: DynamoGraphDeployment\nmetadata:\n  name: generated\n"


def _make_dgdr(
    image: str = _REQUEST_IMAGE, backend: str = "auto"
) -> DynamoGraphDeploymentRequestSpec:
    return DynamoGraphDeploymentRequestSpec(
        model=_MODEL,
        backend=backend,
        image=image,
        hardware=HardwareSpec(gpuSku="h200_sxm", totalGpus=8, numGpusPerNode=8),
        workload=WorkloadSpec(isl=4000, osl=1000),
        sla=SLASpec(ttft=2000.0, itl=50.0),
    )


def _make_task(backend: str, serving_mode: str = "agg") -> Task:
    """A real AIC Task for one backend, matching `_make_dgdr()`."""
    shared = {"total_gpus": 8, "isl": 4000, "osl": 1000, "ttft": 2000.0, "tpot": 50.0}
    if serving_mode == "agg":
        return Task(
            serving_mode="agg",
            model_path=_MODEL,
            system_name="h200_sxm",
            backend_name=backend,
            **shared,
        )
    return Task(
        serving_mode="disagg",
        prefill_model_path=_MODEL,
        decode_model_path=_MODEL,
        prefill_system_name="h200_sxm",
        decode_system_name="h200_sxm",
        prefill_backend_name=backend,
        decode_backend_name=backend,
        **shared,
    )


def _make_row(**columns) -> pd.DataFrame:
    """A rank-1 picker row. `backend` and `_task_key` are the columns AIC adds
    when it merges per-backend experiments into one bucket."""
    row: dict = {"tp": 1}
    row.update(columns)
    return pd.DataFrame([row])


def _drive(dgdr, best_config_df, chosen_exp, task_configs) -> dict:
    """Run `_generate_dgd_from_pick` with AIC's generator entry points patched
    and report the image and the backend each one was handed."""
    seen: dict = {"k8s_image": None, "generator_backend": None, "generator_calls": 0}

    def fake_bridge(task_config, result_df, generator_overrides=None, **_kw):
        overrides = generator_overrides or {}
        seen["k8s_image"] = overrides.get("K8sConfig", {}).get("k8s_image")
        return {}

    def fake_artifacts(params, backend, **_kw):
        seen["generator_backend"] = backend
        seen["generator_calls"] += 1
        return {"k8s_deploy.yaml": _ARTIFACT_YAML}

    with (
        patch(
            "dynamo.profiler.rapid.task_config_to_generator_config",
            side_effect=fake_bridge,
        ),
        patch(
            "dynamo.profiler.rapid.generate_backend_artifacts",
            side_effect=fake_artifacts,
        ),
    ):
        seen["dgd_config"] = _generate_dgd_from_pick(
            dgdr, best_config_df, chosen_exp, task_configs
        )
    return seen


def _messages(caplog, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


class TestWinningBackendIsHonoured:
    """The emitted backend equals the winning row's backend, or nothing is
    emitted."""

    def test_task_disagreeing_with_winning_row_emits_nothing(self, caplog):
        """A TensorRT-LLM task under the plain `agg` key while the winning row
        says SGLang must not produce a deployment: it would be a TensorRT-LLM
        image running TensorRT-LLM arguments while the run summary named
        SGLang.
        """
        dgdr = _make_dgdr()
        task_configs = {"agg": _make_task("trtllm")}

        with caplog.at_level(logging.ERROR, logger=_RAPID_LOGGER):
            seen = _drive(dgdr, _make_row(backend="sglang"), "agg", task_configs)

        assert seen["dgd_config"] is None
        assert seen["generator_calls"] == 0
        errors = _messages(caplog, logging.ERROR)
        assert any("sglang" in m and "trtllm" in m for m in errors), errors

    def test_unresolvable_task_is_reported(self, caplog):
        """No task for the chosen experiment must be an error naming the
        experiment and the keys that were available, not a silent `None`."""
        dgdr = _make_dgdr()
        task_configs = {"disagg_vllm": _make_task("vllm", "disagg")}

        with caplog.at_level(logging.ERROR, logger=_RAPID_LOGGER):
            seen = _drive(dgdr, _make_row(backend="sglang"), "agg", task_configs)

        assert seen["dgd_config"] is None
        assert seen["generator_calls"] == 0
        errors = _messages(caplog, logging.ERROR)
        assert any("agg" in m and "disagg_vllm" in m for m in errors), errors


class TestSiblingImageIsAnnounced:
    """The derived image can name a repository the operator never supplied."""

    def test_warns_when_derived_repository_differs_from_the_request(self, caplog):
        """The request supplies the SGLang runtime image but the winning
        backend is TensorRT-LLM, so the deployment depends on a sibling image
        that may not exist in the operator's registry."""
        dgdr = _make_dgdr(image="myregistry.io/sglang-runtime:1.2.3")
        task_configs = {"agg": _make_task("trtllm")}

        with caplog.at_level(logging.WARNING, logger=_RAPID_LOGGER):
            seen = _drive(dgdr, _make_row(), "agg", task_configs)

        assert seen["k8s_image"] == "myregistry.io/tensorrtllm-runtime:1.2.3"
        warnings = _messages(caplog, logging.WARNING)
        assert any(
            "sglang-runtime" in m and "tensorrtllm-runtime" in m for m in warnings
        ), warnings

    def test_no_warning_when_the_request_already_names_that_repository(self, caplog):
        dgdr = _make_dgdr(image="myregistry.io/sglang-runtime:1.2.3")
        task_configs = {"agg": _make_task("sglang")}

        with caplog.at_level(logging.WARNING, logger=_RAPID_LOGGER):
            seen = _drive(dgdr, _make_row(), "agg", task_configs)

        assert seen["k8s_image"] == "myregistry.io/sglang-runtime:1.2.3"
        assert _messages(caplog, logging.WARNING) == []


class TestMergedAutoBackendPath:
    """Control: with per-backend task keys the image and the generator already
    agree, and must keep agreeing."""

    def test_image_and_generator_both_follow_the_winning_backend(self):
        dgdr = _make_dgdr()
        task_configs = {
            "agg_trtllm": _make_task("trtllm"),
            "agg_sglang": _make_task("sglang"),
        }

        seen = _drive(dgdr, _make_row(backend="sglang"), "agg", task_configs)

        assert seen["k8s_image"] == f"{_REGISTRY}/sglang-runtime:1.2.3"
        assert seen["generator_backend"] == "sglang"
        assert seen["dgd_config"] == {
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "generated"},
        }

    def test_published_task_key_resolves_the_task(self):
        """AIC publishes each merged row's originating per-backend key as
        `_task_key`, which resolves the task even when the row carries no
        `backend` column to rebuild that key from."""
        dgdr = _make_dgdr()
        task_configs = {"agg_sglang": _make_task("sglang")}
        row = _make_row(_task_key="agg_sglang")

        seen = _drive(dgdr, row, "agg", task_configs)

        assert seen["k8s_image"] == f"{_REGISTRY}/sglang-runtime:1.2.3"
        assert seen["generator_backend"] == "sglang"
        assert _generated_backend(row, "agg", task_configs) == "sglang"


class TestConcreteBackendPath:
    """Control: a request naming one backend keeps that backend end to end."""

    def test_plain_key_without_a_backend_column(self):
        dgdr = _make_dgdr(backend="sglang")
        task_configs = {"disagg": _make_task("sglang", "disagg")}

        seen = _drive(dgdr, _make_row(), "disagg", task_configs)

        assert seen["k8s_image"] == f"{_REGISTRY}/sglang-runtime:1.2.3"
        assert seen["generator_backend"] == "sglang"
