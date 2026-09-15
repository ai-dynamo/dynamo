# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.timeout(30),
]

MODEL = "runtime-tests/cached"
REVISION = "0" * 40
LOG_SENTINEL = "fetch-model-runtime-bridge-child-ok"
OLD_DISTRIBUTED_RUNTIME_WARNING = (
    "the pyo3 async bridge built its own tokio runtime before this "
    "DistributedRuntime was created"
)
CHILD = textwrap.dedent(
    f"""\
    import asyncio
    import os
    import socket
    import sys
    from pathlib import Path

    import dynamo._core as core

    MODEL = {MODEL!r}
    LOG_SENTINEL = {LOG_SENTINEL!r}
    scenario = sys.argv[1]
    expected_snapshot = Path(sys.argv[2])
    system_port = int(sys.argv[3])

    async def fetch_cached_models(count=1):
        for _ in range(count):
            fetched = Path(await core.fetch_model(MODEL, True))
            assert fetched.resolve() == expected_snapshot.resolve(), (
                fetched,
                expected_snapshot,
            )

    async def trigger_context_bridge():
        context = core.Context("runtime-bridge-test")
        waiter = context.async_killed_or_stopped()
        context.stop_generating()
        assert await asyncio.wait_for(waiter, timeout=1)

    def create_backend_worker(engine):
        runtime_config = core.backend.RuntimeConfig("mem", "tcp", "zmq")
        worker_config = core.backend.WorkerConfig(
            "runtime-bridge-test",
            enable_kv_routing=False,
            runtime=runtime_config,
        )
        return core.backend.Worker(
            engine, worker_config, asyncio.get_running_loop()
        )

    async def exercise_backend_startup_and_cleanup():
        class FailingEngine:
            cleanup_count = 0

            async def start(self, _worker_id):
                # Reaching the engine proves explicit mem/tcp/zmq settings won
                # over the deliberately conflicting environment, without infra.
                _, writer = await asyncio.open_connection("127.0.0.1", system_port)
                writer.close()
                await writer.wait_closed()
                raise RuntimeError("runtime-bridge-engine-start-failed")

            async def cleanup(self):
                self.cleanup_count += 1

        engine = FailingEngine()
        worker = create_backend_worker(engine)
        try:
            await asyncio.wait_for(worker.run(), timeout=5)
        except Exception as error:
            assert "runtime-bridge-engine-start-failed" in str(error), error
        else:
            raise AssertionError("engine startup unexpectedly succeeded")
        assert engine.cleanup_count == 1, engine.cleanup_count

        # shutdown() schedules cancellation; wait for observable socket release,
        # rather than assuming cleanup completed when run() returned its error.
        async def wait_for_listener_release():
            while True:
                with socket.socket() as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        listener.bind(("127.0.0.1", system_port))
                    except OSError:
                        await asyncio.sleep(0.01)
                    else:
                        return

        await asyncio.wait_for(wait_for_listener_release(), timeout=5)

    no_loop_error = None
    if scenario == "no_loop_then_fetch":
        # The missing-loop error must win over invalid runtime settings and
        # leave runtime initialization retryable once those settings are fixed.
        try:
            core.fetch_model(MODEL, True)
        except RuntimeError as error:
            no_loop_error = str(error)
        else:
            raise AssertionError("fetch_model returned an awaitable without a running loop")
        assert "no running event loop" in no_loop_error.lower(), no_loop_error
        os.environ["DYN_RUNTIME_NUM_WORKER_THREADS"] = "1"

    async def main():
        if scenario == "fetch_first_distributed_runtime":
            await fetch_cached_models()
            runtime = core.DistributedRuntime(
                asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
            )
            runtime.shutdown()
        elif scenario == "context_first":
            await trigger_context_bridge()
            await fetch_cached_models(2)
        elif scenario == "detached_then_fetch":
            runtime = core.DistributedRuntime.detached()
            try:
                await fetch_cached_models()
                # detached() does not register the bridge or emit the old
                # mismatch warning. The ordinary constructor observes whether
                # fetching correctly registered the already-existing runtime.
                observer = core.DistributedRuntime(
                    asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
                )
                observer.shutdown()
            finally:
                runtime.shutdown()
        elif scenario == "fetch_then_backend_worker":
            await fetch_cached_models()
            await exercise_backend_startup_and_cleanup()
            await fetch_cached_models()
        elif scenario == "no_loop_then_fetch":
            await fetch_cached_models()
        elif scenario == "invalid_config_then_fetch":
            try:
                pending = core.fetch_model(MODEL, True)
            except Exception as error:
                assert "num_worker_threads" in str(error), error
            else:
                pending.cancel()
                raise AssertionError("invalid runtime config returned an awaitable")

            os.environ["DYN_RUNTIME_NUM_WORKER_THREADS"] = "1"
            await fetch_cached_models()
        else:
            raise AssertionError(f"unknown scenario: {{scenario}}")

        core.log_message(
            "warn",
            f"{{LOG_SENTINEL}}:{{scenario}}",
            "fetch_model_runtime_bridge_test",
            "fetch_model_runtime_bridge_child.py",
            1,
        )

    asyncio.run(main())
    """
)


SCENARIOS = [
    pytest.param("fetch_first_distributed_runtime", False, id="fetch-first"),
    pytest.param("context_first", True, id="context-first"),
    pytest.param("detached_then_fetch", False, id="detached-then-fetch"),
    pytest.param("fetch_then_backend_worker", False, id="backend-after-fetch"),
    pytest.param(
        "no_loop_then_fetch",
        False,
        id="no-loop-then-fetch",
    ),
    pytest.param(
        "invalid_config_then_fetch",
        False,
        id="invalid-config-then-fetch",
    ),
]


def _build_cached_model(cache: Path) -> Path:
    repository = cache / "models--runtime-tests--cached"
    snapshot = repository / "snapshots" / REVISION
    refs = repository / "refs"
    snapshot.mkdir(parents=True)
    refs.mkdir()
    (refs / "main").write_text(REVISION, encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    return snapshot


def _isolated_child_env(cache: Path, scenario: str, system_port: int) -> dict[str, str]:
    isolated_prefixes = (
        "DYN_",
        "DYNAMO_",
        "HF_",
        "MODEL_EXPRESS_",
        "NATS_",
        "OTEL_",
        "TRANSFORMERS_",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(isolated_prefixes)
    }
    env.update(
        {
            "DYN_RUNTIME_NUM_WORKER_THREADS": (
                "0"
                if scenario in ("invalid_config_then_fetch", "no_loop_then_fetch")
                else "1"
            ),
            "DYN_RUNTIME_MAX_BLOCKING_THREADS": "1",
            "DYN_COMPUTE_THREADS": "1",
            "DYN_SYSTEM_PORT": "-1",
            "DYN_DISCOVERY_BACKEND": "mem",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_USE_KV_EVENTS": "false",
            "DYN_HEALTH_CHECK_ENABLED": "false",
            "DYN_LOG": "warn",
            "DYN_LOGGING_CONSOLE_FORMAT": "jsonl",
            "HF_HUB_CACHE": str(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OTEL_EXPORT_ENABLED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if scenario == "fetch_then_backend_worker":
        env.update(
            DYN_DISCOVERY_BACKEND="etcd",
            DYN_REQUEST_PLANE="nats",
            DYN_EVENT_PLANE="nats",
            DYN_SYSTEM_HOST="127.0.0.1",
            DYN_SYSTEM_PORT=str(system_port),
        )
    return env


def _parse_jsonl_logs(output: str) -> list[dict[str, Any]]:
    records = []
    for raw_line in output.splitlines():
        json_start = raw_line.find("{")
        if json_start < 0:
            continue
        try:
            record = json.loads(raw_line[json_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@pytest.mark.parametrize(("scenario", "expect_mismatch"), SCENARIOS)
def test_fetch_model_runtime_bridge_orders(
    tmp_path: Path, scenario: str, expect_mismatch: bool, dynamo_dynamic_ports
) -> None:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    snapshot = _build_cached_model(cache)
    system_port = dynamo_dynamic_ports.system_ports[0]
    result = subprocess.run(
        [sys.executable, "-c", CHILD, scenario, str(snapshot), str(system_port)],
        env=_isolated_child_env(cache, scenario, system_port),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    diagnostic = (
        f"scenario={scenario} returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, diagnostic

    records = _parse_jsonl_logs(result.stderr)
    assert any(
        record.get("message") == f"{LOG_SENTINEL}:{scenario}" for record in records
    ), f"logging sentinel missing\n{diagnostic}"

    mismatches = [
        record
        for record in records
        if record.get("operation") == "fetch_model"
        and record.get("runtime_bridge_mismatch") is True
    ]
    if expect_mismatch:
        assert (
            len(mismatches) == 1
        ), f"expected one runtime mismatch event: {mismatches}\n{diagnostic}"
        mismatch = mismatches[0]
        pyo3_runtime_id = mismatch.get("pyo3_runtime_id")
        dynamo_runtime_id = mismatch.get("dynamo_runtime_id")
        assert pyo3_runtime_id is not None, f"pyo3 runtime ID missing\n{diagnostic}"
        assert dynamo_runtime_id is not None, f"Dynamo runtime ID missing\n{diagnostic}"
        assert (
            pyo3_runtime_id != dynamo_runtime_id
        ), f"mismatch event reported equal runtime IDs: {mismatch}\n{diagnostic}"
    else:
        assert (
            not mismatches
        ), f"unexpected runtime mismatch events: {mismatches}\n{diagnostic}"

    if scenario in ("fetch_first_distributed_runtime", "detached_then_fetch"):
        assert OLD_DISTRIBUTED_RUNTIME_WARNING not in result.stderr, diagnostic
