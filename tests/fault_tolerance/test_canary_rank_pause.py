# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-pause test: prove the canary detects real engine hangs.

We launch a real engine worker, wait for it to be Ready, then SIGSTOP the
engine-rank child process to simulate a hang (the engine can't produce
tokens, but the Python dispatch layer is still alive). We then poll
``/health`` on the worker's system port and assert:

  * When canary is ENABLED AND the endpoint has a registered payload,
    ``/health`` flips to 503 — the canary probe times out against the
    paused engine and marks the endpoint NotReady. This is the positive
    case the user asked for: *"canary can detect the issues we have seen."*

  * When canary is DISABLED (or the endpoint opts out by registering no
    payload, as trtllm disagg decode does today), ``/health`` STAYS at
    200 — there is no active liveness probe. This is the negative
    control the user asked for: it proves the harness distinguishes the
    "canary working" path from the "no canary" path, rather than
    always reporting success.

After each case we ``SIGCONT`` the rank and verify ``/health`` returns to
Ready, both as a cleanup and as a round-trip sanity check.

Style mirrors ``tests/serve/test_trtllm.py::test_deployment`` — same
``EngineProcess.from_script`` launcher, same ``/health`` polling. We do
not use the ``tests/fault_tolerance/hardware/fault_injection_service``
helpers; just POSIX signals.
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

from tests.serve.test_trtllm import trtllm_configs
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)

# Seconds: how long to wait after SIGSTOP for the canary to notice.
# Canary interval in the runtime is typically 10 s; give it generous slack.
PAUSE_DETECT_BUDGET_S = 45

# Seconds: after SIGCONT, how long until /health is back.
RESUME_RECOVER_BUDGET_S = 30

# Seconds: how long to wait for /health to initially come up Ready.
STARTUP_READY_BUDGET_S = 120


@dataclass
class RankPauseScenario:
    """One row in the test matrix."""

    # Label used in pytest param IDs.
    label: str
    # Which trtllm_configs entry to launch (must already be defined in tests/serve).
    base_config_key: str
    # Which system-port index to monitor (0 = prefill / single worker, 1 = decode).
    system_port_index: int
    # Whether to enable canary in the worker env.
    canary_enabled: bool
    # "detect" → /health must flip to 503 after SIGSTOP.
    # "miss"   → /health must stay 200 after SIGSTOP.
    expected: str


# NB: only trtllm scenarios for now. vllm + sglang added in follow-ups once
# we verify the engine-rank process discovery pattern for each backend.
SCENARIOS: list[RankPauseScenario] = [
    # Positive case: agg worker, canary on. Canary should catch the pause.
    RankPauseScenario(
        label="trtllm-agg-canary-on",
        base_config_key="aggregated",
        system_port_index=0,
        canary_enabled=True,
        expected="detect",
    ),
    # Negative control: same worker, canary off. Harness must NOT false-positive.
    RankPauseScenario(
        label="trtllm-agg-canary-off",
        base_config_key="aggregated",
        system_port_index=0,
        canary_enabled=False,
        expected="miss",
    ),
    # Disagg decode: canary opts out by design (trtllm handler rejects generic
    # probes). Documents the current trade-off — will flip to "detect" once
    # the probe-canary follow-up lands.
    RankPauseScenario(
        label="trtllm-disagg-decode-canary-on",
        base_config_key="disaggregated_same_gpu",
        system_port_index=1,  # DYN_SYSTEM_PORT2 = decode worker
        canary_enabled=True,
        expected="miss",
    ),
]


def _find_engine_rank_pid(parent_pid: int, timeout_s: float = 30.0) -> int:
    """Locate the TRT-LLM engine-rank subprocess under the test's worker parent.

    The trtllm Python worker spawns a child process with a command line
    containing ``EngineCore`` (see ``stragglers`` in TRTLLMConfig). We wait
    up to ``timeout_s`` for it to appear so callers don't race startup.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            parent = psutil.Process(parent_pid)
            for child in parent.children(recursive=True):
                try:
                    cmd = " ".join(child.cmdline())
                except psutil.Error:
                    continue
                if "EngineCore" in cmd or "tensorrt_llm" in cmd:
                    return child.pid
        except psutil.NoSuchProcess as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(
        f"Could not find engine-rank child of pid={parent_pid} within {timeout_s}s. "
        f"Last error: {last_err}"
    )


def _health_status(url: str, timeout: float = 2.0) -> int:
    """Return HTTP status from /health, or 0 on connection error."""
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code
    except requests.exceptions.RequestException:
        return 0


def _wait_for_status(url: str, target: int, deadline_s: float) -> int:
    """Poll /health until status == target or deadline reached. Return last status."""
    deadline = time.monotonic() + deadline_s
    last = -1
    while time.monotonic() < deadline:
        last = _health_status(url)
        if last == target:
            return last
        time.sleep(1.0)
    return last


@pytest.mark.e2e
@pytest.mark.trtllm
@pytest.mark.gpu_1
@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda s: s.label if isinstance(s, RankPauseScenario) else str(s),
)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_canary_detects_rank_pause(
    scenario: RankPauseScenario,
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ANN001
    dynamo_dynamic_ports,  # noqa: ANN001
    num_system_ports,  # noqa: ANN001
    predownload_models,  # noqa: ANN001
) -> None:
    base = trtllm_configs[scenario.base_config_key]
    config = dataclasses.replace(base, frontend_port=dynamo_dynamic_ports.frontend_port)
    config.env.update(
        {
            "MODEL_PATH": config.model,
            "SERVED_MODEL_NAME": config.model,
            "DYN_HEALTH_CHECK_ENABLED": "true" if scenario.canary_enabled else "false",
        }
    )

    system_ports = [int(p) for p in dynamo_dynamic_ports.system_ports]
    assert len(system_ports) > scenario.system_port_index, (
        f"scenario wants system_port_index={scenario.system_port_index} "
        f"but only {len(system_ports)} ports are allocated"
    )
    target_port = system_ports[scenario.system_port_index]
    health_url = f"http://localhost:{target_port}/health"
    logger.info(
        "[%s] health_url=%s canary=%s expected=%s",
        scenario.label,
        health_url,
        scenario.canary_enabled,
        scenario.expected,
    )

    # Build env + launch the worker using the exact same machinery as
    # test_deployment so the scenario matches real CI conditions.
    extra_env: dict[str, str] = {}
    for i, p in enumerate(system_ports, start=1):
        extra_env[f"DYN_SYSTEM_PORT{i}"] = str(p)
    extra_env["DYN_SYSTEM_PORT"] = str(system_ports[0])
    extra_env["DYN_HTTP_PORT"] = str(dynamo_dynamic_ports.frontend_port)
    extra_env["DYN_HEALTH_CHECK_ENABLED"] = (
        "true" if scenario.canary_enabled else "false"
    )

    with EngineProcess.from_script(config, request, extra_env=extra_env) as proc:
        # 1. Wait for the worker to become healthy (Ready) before we pause.
        status = _wait_for_status(health_url, 200, STARTUP_READY_BUDGET_S)
        assert status == 200, (
            f"[{scenario.label}] worker never became healthy "
            f"(last status={status}) within {STARTUP_READY_BUDGET_S}s"
        )

        # 2. Find the engine rank subprocess and SIGSTOP it.
        rank_pid = _find_engine_rank_pid(proc.proc.pid)
        logger.info("[%s] pausing engine rank pid=%d", scenario.label, rank_pid)
        os.kill(rank_pid, signal.SIGSTOP)
        try:
            # 3. Give the canary (or lack thereof) time to notice.
            if scenario.expected == "detect":
                status = _wait_for_status(health_url, 503, PAUSE_DETECT_BUDGET_S)
                assert status == 503, (
                    f"[{scenario.label}] canary FAILED to detect rank pause: "
                    f"/health returned {status} (expected 503) within "
                    f"{PAUSE_DETECT_BUDGET_S}s"
                )
            else:  # miss
                # Wait the full budget; /health must stay 200 throughout. We
                # sample periodically rather than trusting a single poll so a
                # transient glitch doesn't masquerade as "miss".
                deadline = time.monotonic() + PAUSE_DETECT_BUDGET_S
                flips = 0
                while time.monotonic() < deadline:
                    s = _health_status(health_url)
                    if s != 200:
                        flips += 1
                    time.sleep(2.0)
                assert flips == 0, (
                    f"[{scenario.label}] expected canary to MISS rank pause "
                    f"(no active probe), but /health flipped away from 200 "
                    f"{flips} time(s). Harness may be false-detecting."
                )
        finally:
            # 4. Always resume the rank so teardown can complete cleanly.
            try:
                os.kill(rank_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

        # 5. After resume, for the detect case, confirm /health returns to 200.
        if scenario.expected == "detect":
            status = _wait_for_status(health_url, 200, RESUME_RECOVER_BUDGET_S)
            assert status == 200, (
                f"[{scenario.label}] /health did not recover after SIGCONT "
                f"(last status={status}) within {RESUME_RECOVER_BUDGET_S}s"
            )
