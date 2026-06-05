# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancel-path teardown tests for ManagedDeployment.

A SIGTERM / Ctrl-C / runner-cancel must tear the deployment down via
``__aexit__`` (extract logs, delete the DGD, close the ApiClient) instead of
stranding the deployment + leaking an "Unclosed connector". These tests stub
the kubernetes I/O so no real cluster is touched, then assert the cleanup path
runs under cancellation.
"""
import asyncio
import logging
import signal
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.utils.managed_deployment import ManagedDeployment

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


def _make_managed_deployment(tmp_path):
    """A ManagedDeployment with every cluster-touching method stubbed."""
    spec = types.SimpleNamespace(name="test-dgd", namespace="test-ns")
    md = ManagedDeployment(
        namespace="test-ns",
        log_dir=str(tmp_path),
        deployment_spec=spec,
        skip_service_restart=True,
    )
    for name in (
        "_init_kubernetes",
        "_scrub_namespace",
        "_restart_etcd",
        "_restart_nats",
        "_create_log_collection_pvc",
        "_prefetch_models",
        "_create_deployment",
        "_wait_for_ready",
        "_capture_metrics",
        "_cleanup_orphaned_jobs",
        "_cleanup_log_collection_pvc",
        "_extract_logs_from_pvc",
        "_delete_deployment",
    ):
        setattr(md, name, AsyncMock(name=name))
    md._get_service_logs = MagicMock(name="_get_service_logs")
    md._logger = logging.getLogger("test-md")
    # _init_kubernetes is stubbed, so plant the ApiClient ourselves to verify
    # __aexit__ closes it (the "Unclosed connector" fix).
    md._api_client = AsyncMock(name="api_client")
    return md


async def test_cancel_during_body_runs_full_cleanup(tmp_path):
    """Cancelling the task in the ``async with`` body must still extract logs,
    delete the DGD, and close the client — not strand the run."""
    md = _make_managed_deployment(tmp_path)
    api_client = md._api_client
    entered = asyncio.Event()

    async def body():
        async with md:
            entered.set()
            await asyncio.sleep(60)  # block until cancelled

    task = asyncio.create_task(body())
    await asyncio.wait_for(entered.wait(), timeout=5)

    # What loop.add_signal_handler(SIGTERM, ...) would invoke on a real kill.
    md._on_cancel_signal(signal.SIGTERM)

    with pytest.raises(asyncio.CancelledError):
        await task

    md._extract_logs_from_pvc.assert_awaited()  # logs preserved before delete
    md._delete_deployment.assert_awaited()  # DGD removed
    api_client.close.assert_awaited()  # no leaked aiohttp connector


async def test_repeat_signal_is_one_shot(tmp_path):
    """A second signal during teardown must be a no-op: cancel exactly once,
    never re-cancel or fall through to the default terminate action."""
    md = _make_managed_deployment(tmp_path)
    md._main_task = MagicMock()
    md._cancelling = False

    md._on_cancel_signal(signal.SIGTERM)
    md._on_cancel_signal(signal.SIGTERM)
    md._on_cancel_signal(signal.SIGINT)

    md._main_task.cancel.assert_called_once()


async def test_normal_exit_still_cleans_up(tmp_path):
    """The happy path (no cancel) must still extract + delete + close."""
    md = _make_managed_deployment(tmp_path)
    api_client = md._api_client

    async with md:
        pass

    md._extract_logs_from_pvc.assert_awaited()
    md._delete_deployment.assert_awaited()
    api_client.close.assert_awaited()
