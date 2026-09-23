# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression for the manual GPU probe's failure cleanup."""

import signal
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from tests.runtime import validate_gpu_shutdown as probe

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_probe_restores_alarm_before_process_cleanup(monkeypatch, tmp_path):
    # Regression: an assertion left SIGALRM armed during process cleanup,
    # masking the original failure and interrupting resource reclamation.
    previous_handler = signal.getsignal(signal.SIGALRM)
    cleanup_state = []

    @contextmanager
    def managed_process(**kwargs):
        try:
            yield SimpleNamespace(
                proc=SimpleNamespace(pid=123, wait=lambda **kwargs: 0)
            )
        finally:
            cleanup_state.append(
                (
                    signal.getitimer(signal.ITIMER_REAL)[0],
                    signal.getsignal(signal.SIGALRM),
                )
            )

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        iter_lines=lambda **kwargs: iter(
            [
                b'data: {"choices":[{"text":"done","finish_reason":"length"}]}',
                b"data: [DONE]",
            ]
        ),
    )
    monkeypatch.setattr(probe, "ManagedProcess", managed_process)
    monkeypatch.setattr(probe.requests, "post", lambda *a, **kw: nullcontext(response))
    monkeypatch.setattr(
        probe.psutil, "Process", lambda pid: SimpleNamespace(children=lambda **kw: [])
    )
    monkeypatch.setattr(probe.os, "kill", lambda *args: None)
    args = SimpleNamespace(
        backend="vllm",
        model="unused",
        output=tmp_path,
        total_timeout=30,
        cleanup_timeout=5,
        expect_interrupted=True,
    )

    with pytest.raises(AssertionError, match="expiry was not exercised"):
        probe.validate(args)

    assert cleanup_state == [(0.0, previous_handler), (0.0, previous_handler)]
