# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary detects rank-pause hangs across {trtllm, vllm, sglang} x {agg, disagg-prefill, disagg-decode}.

To also validate that canary adds signal (vs. always failing), set
DYN_HEALTH_CHECK_ENABLED=false and re-run one scenario locally: /health
should stay 200 after SIGSTOP.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any

import psutil
import pytest
import requests

from tests.serve.test_sglang import sglang_configs
from tests.serve.test_trtllm import trtllm_configs
from tests.serve.test_vllm import vllm_configs
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)

PAUSE_DETECT_BUDGET_S = 45
RESUME_RECOVER_BUDGET_S = 30
STARTUP_READY_BUDGET_S = 120

_RANK_PATTERNS: dict[str, tuple[str, ...]] = {
    "trtllm": ("mpi4py.futures.server", "TRTLLM:EngineCore", "tensorrt_llm"),
    "vllm": ("VLLM::EngineCore", "EngineCoreProc"),
    "sglang": ("sglang::scheduler",),
}

_CONFIGS_BY_BACKEND = {
    "trtllm": trtllm_configs,
    "vllm": vllm_configs,
    "sglang": sglang_configs,
}


@dataclass
class RankPauseScenario:
    label: str
    backend: str
    base_config_key: str
    system_port_index: int


def _scenarios_for(backend: str) -> list:
    mark = getattr(pytest.mark, backend)
    return [
        pytest.param(
            RankPauseScenario(f"{backend}-agg", backend, "aggregated", 0),
            marks=mark,
            id=f"{backend}-agg",
        ),
        pytest.param(
            RankPauseScenario(
                f"{backend}-disagg-prefill", backend, "disaggregated_same_gpu", 0
            ),
            marks=mark,
            id=f"{backend}-disagg-prefill",
        ),
        pytest.param(
            RankPauseScenario(
                f"{backend}-disagg-decode", backend, "disaggregated_same_gpu", 1
            ),
            marks=mark,
            id=f"{backend}-disagg-decode",
        ),
    ]


SCENARIOS = [
    p for backend in ("trtllm", "vllm", "sglang") for p in _scenarios_for(backend)
]


def _find_engine_rank_pid(
    parent_pid: int,
    patterns: tuple[str, ...],
    target_port: int,
    timeout_s: float = 30.0,
) -> int:
    """Find the rank process for the worker whose DYN_SYSTEM_PORT matches target_port.

    Same-GPU disagg has multiple Python processes sharing env (frontend + both
    workers). We match the candidate that both carries target_port in its env
    AND has a rank descendant matching `patterns`.
    """
    target = str(target_port)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            root = psutil.Process(parent_pid)
            for child in root.children(recursive=True):
                try:
                    env = child.environ()
                except psutil.Error:
                    continue
                if env.get("DYN_SYSTEM_PORT") != target:
                    continue
                for desc in child.children(recursive=True):
                    try:
                        dcmd = " ".join(desc.cmdline())
                    except psutil.Error:
                        continue
                    if any(p in dcmd for p in patterns):
                        return desc.pid
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"no engine-rank matching {patterns} with DYN_SYSTEM_PORT={target_port} "
        f"under pid={parent_pid}"
    )


def _health_status(url: str, timeout: float = 2.0) -> int:
    try:
        return requests.get(url, timeout=timeout).status_code
    except requests.exceptions.RequestException:
        return 0


def _wait_for_status(url: str, target: int, deadline_s: float) -> int:
    deadline = time.monotonic() + deadline_s
    last = -1
    while time.monotonic() < deadline:
        last = _health_status(url)
        if last == target:
            return last
        time.sleep(1.0)
    return last


@pytest.mark.fault_tolerance
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.timeout(300)
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_canary_detects_rank_pause(
    scenario: RankPauseScenario,
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ANN001
    dynamo_dynamic_ports,  # noqa: ANN001
    num_system_ports,  # noqa: ANN001
    predownload_models,  # noqa: ANN001
) -> None:
    base = _CONFIGS_BY_BACKEND[scenario.backend][scenario.base_config_key]
    # Fresh env dict — dataclasses.replace shallow-copies, so reusing base.env
    # would mutate the shared module-level config across parametrized runs.
    config = dataclasses.replace(
        base,
        frontend_port=dynamo_dynamic_ports.frontend_port,
        env={
            **base.env,
            "MODEL_PATH": base.model,
            "SERVED_MODEL_NAME": base.model,
            "DYN_HEALTH_CHECK_ENABLED": "true",
        },
    )

    system_ports = [int(p) for p in dynamo_dynamic_ports.system_ports]
    target_port = system_ports[scenario.system_port_index]
    health_url = f"http://localhost:{target_port}/health"

    extra_env: dict[str, str] = {
        f"DYN_SYSTEM_PORT{i}": str(p) for i, p in enumerate(system_ports, start=1)
    }
    extra_env["DYN_SYSTEM_PORT"] = str(system_ports[0])
    extra_env["DYN_HTTP_PORT"] = str(dynamo_dynamic_ports.frontend_port)
    extra_env["DYN_HEALTH_CHECK_ENABLED"] = "true"

    with EngineProcess.from_script(config, request, extra_env=extra_env) as proc:
        assert (
            _wait_for_status(health_url, 200, STARTUP_READY_BUDGET_S) == 200
        ), f"[{scenario.label}] worker never Ready"

        rank_pid = _find_engine_rank_pid(
            proc.proc.pid, _RANK_PATTERNS[scenario.backend], target_port
        )
        logger.info("[%s] SIGSTOP rank pid=%d", scenario.label, rank_pid)
        os.kill(rank_pid, signal.SIGSTOP)
        try:
            assert (
                _wait_for_status(health_url, 503, PAUSE_DETECT_BUDGET_S) == 503
            ), f"[{scenario.label}] canary did not flip /health to 503"
        finally:
            try:
                os.kill(rank_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

        assert (
            _wait_for_status(health_url, 200, RESUME_RECOVER_BUDGET_S) == 200
        ), f"[{scenario.label}] /health did not recover after SIGCONT"
