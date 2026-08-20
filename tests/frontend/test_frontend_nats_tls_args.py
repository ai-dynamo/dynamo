# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: frontend NATS TLS CLI flags must reach the Rust runtime.

The Rust side snapshots NATS TLS settings once when DistributedRuntime is
constructed, so propagating them via os.environ afterwards is too late. The
frontend must pass them through explicit DistributedRuntime parameters.
"""

import asyncio
import os
import sys
from unittest import mock

import pytest

from dynamo.frontend import main as frontend_main

_STOP = "stop after runtime construction"


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
class TestFrontendNatsTlsArgs:
    def _run_until_runtime_init(self, monkeypatch, argv):
        """Run async_main up to DistributedRuntime construction.

        The patched constructor raises to abort startup before any servers or
        signal handlers are set up. Returns the DistributedRuntime mock.
        """
        monkeypatch.setattr(sys, "argv", argv)
        with mock.patch.object(frontend_main, "DistributedRuntime") as mock_rt:
            mock_rt.side_effect = RuntimeError(_STOP)
            with pytest.raises(RuntimeError, match=_STOP):
                asyncio.run(frontend_main.async_main())
        mock_rt.assert_called_once()
        return mock_rt

    def test_cli_flags_passed_to_distributed_runtime(self, monkeypatch):
        mock_rt = self._run_until_runtime_init(
            monkeypatch,
            [
                "dynamo.frontend",
                "--nats-tls-ca-cert-path",
                "/etc/certs/internal-ca.pem",
                "--nats-tls-insecure",
            ],
        )
        _, kwargs = mock_rt.call_args
        assert kwargs["nats_tls_ca_cert_path"] == "/etc/certs/internal-ca.pem"
        assert kwargs["nats_tls_insecure"] is True

    def test_no_flags_passes_defaults_without_env_writeback(self, monkeypatch):
        monkeypatch.delenv("NATS_TLS_CA_CERT_PATH", raising=False)
        monkeypatch.delenv("NATS_TLS_INSECURE", raising=False)
        mock_rt = self._run_until_runtime_init(monkeypatch, ["dynamo.frontend"])
        _, kwargs = mock_rt.call_args
        assert kwargs["nats_tls_ca_cert_path"] is None
        assert kwargs["nats_tls_insecure"] is False
        # Parsed config must not leak back into the process environment.
        assert "NATS_TLS_CA_CERT_PATH" not in os.environ
        assert "NATS_TLS_INSECURE" not in os.environ
