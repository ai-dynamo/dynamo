# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Long-running cascade driver. Holds a 4P:2D:2F DGD open and listens on a
# Unix socket for commands sent by the cascade_inject CLI. Lets us
# deploy ONCE and explore the cascade interactively — adjust load,
# inject faults, snapshot state, watch via Grafana / TUI / spike_view —
# without paying the FSx + model-load tax for every iteration.
#
# Run from inside the dev container:
#
#   pytest tests/fault_tolerance/deploy/test_cascade_console.py \
#     -s --namespace neelays-test --log-pvc qwen3-30b-logs \
#     --model-pvc shared-model-cache --baseline-concurrency 8
#
# In another terminal (or window), drive it:
#
#   python -m tests.utils.cascade_inject status
#   python -m tests.utils.cascade_inject load 16
#   python -m tests.utils.cascade_inject stall VllmDecodeWorker --duration 30s
#   python -m tests.utils.cascade_inject dump
#   python -m tests.utils.cascade_inject quit
#
# `quit` triggers a clean teardown (DGD delete, log PVC kept).

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.fault_tolerance.deploy.events import StallProcess, WaitForModelReady
from tests.fault_tolerance.deploy.scenario import RuntimeEnv, ScenarioContext
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment
from tests.utils.managed_load import LoadConfig, ManagedLoad

logger = logging.getLogger(__name__)

# Socket path is per-DGD so multiple consoles don't collide.
_SOCKET_FMT = "/tmp/cascade-inject-{dgd}.sock"


# ────────────────────────────────────────────────────────────────────────────
# IPC server


class CascadeConsole:
    """Long-running command dispatcher for an open DGD."""

    def __init__(
        self,
        ctx: ScenarioContext,
        dgd_name: str,
        baseline_concurrency: int,
    ):
        self.ctx = ctx
        self.dgd_name = dgd_name
        self.socket_path = Path(_SOCKET_FMT.format(dgd=dgd_name))
        self.baseline_concurrency = baseline_concurrency
        self.current_load: Optional[ManagedLoad] = None
        self.active_stalls: dict[str, StallProcess] = {}  # target → event
        self.quit_requested = asyncio.Event()
        self.event_log_path = Path("/tmp/cascade_events.log")

    async def start_load(self, concurrency: int) -> None:
        """Stop existing load (if any) + start a fresh one at given concurrency."""
        if self.current_load:
            self.ctx.logger.info(
                f"console: stopping existing load (concurrency was "
                f"{self.current_load.load_config.concurrency})"
            )
            try:
                await self.current_load.terminate()
            except Exception as e:
                self.ctx.logger.warning(f"terminate failed (continuing): {e}")
            try:
                await self.current_load._cleanup()
            except Exception as e:
                self.ctx.logger.warning(f"cleanup failed (continuing): {e}")
            self.current_load = None

        served_model = self.ctx.deployment.deployment_spec["VllmDecodeWorker"].model
        # Auto-discover worker /metrics URLs.
        spec = self.ctx.deployment.deployment_spec
        urls: list[str] = []
        for service in spec.services:
            if service.component_type == "frontend":
                continue
            pods_by_service = await asyncio.to_thread(
                self.ctx.deployment.get_pods, [service.name]
            )
            for pod in pods_by_service.get(service.name) or []:
                pod_ip = pod.raw.get("status", {}).get("podIP")
                if pod_ip:
                    urls.append(f"http://{pod_ip}:{spec.system_port}/metrics")

        cfg = LoadConfig(
            model_name=served_model,
            tokenizer=served_model,
            input_tokens_mean=7000,
            input_tokens_stddev=600,
            output_tokens_mean=100,
            output_tokens_stddev=40,
            concurrency=concurrency,
            duration_minutes=60.0,
            request_timeout_seconds=300,
            streaming=True,
            ignore_eos=True,
            warmup_requests=0,
            extra_server_metrics_urls=urls or None,
        )
        load = ManagedLoad(
            namespace=self.ctx.namespace,
            load_config=cfg,
            pvc_name=self.ctx.deployment.get_log_pvc_name(),
            endpoint_url=self.ctx.deployment.deployment_spec.get_in_cluster_frontend_url(
                self.ctx.namespace
            ),
            log_dir=self.ctx.log_dir,
            job_name=f"load-c{concurrency}-{secrets.token_hex(4)}",
        )
        await load._init_kubernetes()
        await load.run(wait_for_completion=False)
        self.current_load = load
        self._append_event(f"load c={concurrency} started (job={load.job_name})")
        self.ctx.logger.info(
            f"console: started load at concurrency={concurrency}, "
            f"job={load.job_name}"
        )

    async def stall(
        self,
        target: str,
        duration_s: float,
        process_name: str = "dynamo.vllm",
    ) -> None:
        evt = StallProcess(
            services=[target],
            process_name=process_name,
            duration=duration_s,
        )
        await evt.timed_execute(self.ctx)
        self.active_stalls[target] = evt
        self._append_event(f"stall {target}/{process_name} for {duration_s}s")

    async def unstall(self, target: str) -> None:
        evt = self.active_stalls.pop(target, None)
        if evt:
            try:
                await evt.stop(self.ctx)
            except Exception as e:
                self.ctx.logger.warning(f"unstall failed: {e}")
            self._append_event(f"unstall {target}")
        else:
            self.ctx.logger.warning(f"unstall: no active stall on {target}")

    async def status(self) -> dict[str, Any]:
        pods = await asyncio.to_thread(self.ctx.deployment.get_pods, None)
        pod_summary = []
        for service, pod_list in pods.items():
            for pod in pod_list:
                phase = pod.raw.get("status", {}).get("phase", "?")
                ready = False
                for cs in pod.raw.get("status", {}).get("containerStatuses") or []:
                    if cs.get("name") == "main" and cs.get("ready"):
                        ready = True
                        break
                pod_summary.append(
                    {
                        "service": service,
                        "pod": pod.raw["metadata"]["name"],
                        "phase": phase,
                        "ready": ready,
                    }
                )
        return {
            "dgd": self.dgd_name,
            "load": (
                {
                    "concurrency": self.current_load.load_config.concurrency,
                    "job": self.current_load.job_name,
                }
                if self.current_load
                else None
            ),
            "active_stalls": list(self.active_stalls),
            "pods": pod_summary,
        }

    async def dump_snapshot(self) -> dict[str, Any]:
        """Snapshot all pod manifests + last 200 lines of logs to <log_dir>/snapshots/<ts>/."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = Path(self.ctx.log_dir) / "snapshots" / ts
        out.mkdir(parents=True, exist_ok=True)
        pods = await asyncio.to_thread(self.ctx.deployment.get_pods, None)
        n_pods = 0
        for service, pod_list in pods.items():
            for pod in pod_list:
                pod_name = pod.raw["metadata"]["name"]
                (out / f"{service}_{pod_name}.manifest.json").write_text(
                    json.dumps(pod.raw, indent=2, default=str)
                )
                try:
                    log_text = await asyncio.to_thread(
                        pod.logs, container="main", tail_lines=200
                    )
                    if isinstance(log_text, list):
                        log_text = "\n".join(log_text)
                    (out / f"{service}_{pod_name}.log").write_text(log_text or "")
                except Exception as e:
                    (out / f"{service}_{pod_name}.log.error").write_text(str(e))
                n_pods += 1
        self._append_event(f"dump → {out} ({n_pods} pods)")
        return {"path": str(out), "pod_count": n_pods}

    def _append_event(self, text: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {text}\n"
        try:
            with self.event_log_path.open("a") as f:
                f.write(line)
        except OSError:
            pass

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                cmd_payload = json.loads(line.decode())
            except json.JSONDecodeError as e:
                response = {"ok": False, "error": f"bad JSON: {e}"}
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
                return

            cmd = cmd_payload.get("cmd")
            response: dict[str, Any] = {"ok": True}

            try:
                if cmd == "status":
                    response.update(await self.status())
                elif cmd == "load":
                    n = int(cmd_payload["concurrency"])
                    await self.start_load(n)
                    response["concurrency"] = n
                elif cmd == "stall":
                    await self.stall(
                        cmd_payload["target"],
                        float(cmd_payload.get("duration_s", 30.0)),
                        cmd_payload.get("process_name", "dynamo.vllm"),
                    )
                    response["target"] = cmd_payload["target"]
                elif cmd == "unstall":
                    await self.unstall(cmd_payload["target"])
                    response["target"] = cmd_payload["target"]
                elif cmd == "dump":
                    response.update(await self.dump_snapshot())
                elif cmd == "quit":
                    self._append_event("quit (driver exiting)")
                    self.quit_requested.set()
                    response["msg"] = "exiting"
                else:
                    response["ok"] = False
                    response["error"] = f"unknown cmd: {cmd}"
            except Exception as e:
                response = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.ctx.logger.exception(f"cmd '{cmd}' failed")

            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def serve(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.socket_path)
        )
        os.chmod(str(self.socket_path), 0o660)
        self.ctx.logger.info(
            f"console: listening on {self.socket_path} "
            f"(use `python -m tests.utils.cascade_inject ...`)"
        )
        async with server:
            await self.quit_requested.wait()
            self.ctx.logger.info("console: quit received, shutting down")


# ────────────────────────────────────────────────────────────────────────────
# pytest entrypoint


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.release  # Long-running interactive driver; not for CI.
@pytest.mark.timeout(0)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_cascade_console(runtime_env: RuntimeEnv, request):
    """Long-running driver; teardown only on `cascade-inject quit` or Ctrl-C."""
    baseline = int(request.config.getoption("--baseline-concurrency", default=8))

    spec = DeploymentSpec(
        "/workspace/tests/fault_tolerance/deploy/templates/vllm/"
        "disagg_qwen3_30b_4p2d_2f.yaml"
    )
    spec.set_logging(True, "info")

    log_collection_kwargs: dict[str, Any] = {
        "pvc_size": "500Mi",
        "container_log_dir": "/tmp/service_logs",
    }
    if runtime_env.storage_class:
        log_collection_kwargs["storage_class"] = runtime_env.storage_class
    if runtime_env.log_pvc:
        log_collection_kwargs["pvc_name"] = runtime_env.log_pvc
    spec.enable_log_collection(**log_collection_kwargs)
    if runtime_env.model_pvc:
        spec.enable_model_cache(runtime_env.model_pvc)

    from tests.utils.test_output import resolve_test_output_path

    log_dir = resolve_test_output_path("test_cascade_console")
    test_logger = logging.getLogger("test_cascade_console")

    test_logger.info("=" * 60)
    test_logger.info("CASCADE CONSOLE — long-running driver")
    test_logger.info("=" * 60)

    async with ManagedDeployment(
        namespace=runtime_env.namespace,
        log_dir=log_dir,
        deployment_spec=spec,
        skip_service_restart=runtime_env.skip_service_restart,
        reuse_log_pvc=bool(runtime_env.log_pvc),
        model_pvc=runtime_env.model_pvc if runtime_env.prefetch_model else None,
    ) as deployment:
        ctx = ScenarioContext(
            deployment=deployment,
            events=[],
            checks=[],
            reports=[],
            logger=test_logger,
            namespace=runtime_env.namespace,
            log_dir=log_dir,
            resource_history=[],
        )

        await WaitForModelReady(timeout=1500).timed_execute(ctx)

        console = CascadeConsole(
            ctx=ctx,
            dgd_name=spec.name,
            baseline_concurrency=baseline,
        )
        await console.start_load(baseline)

        test_logger.info(
            f"=== console ready at {console.socket_path} ===\n"
            f"=== `python -m tests.utils.cascade_inject status` to drive it ===\n"
            f"=== `python -m tests.utils.cascade_inject quit` (or Ctrl-C) to stop ===\n"
        )

        await console.serve()

        if console.current_load:
            try:
                await console.current_load.terminate()
                await console.current_load._cleanup()
            except Exception as e:
                test_logger.warning(f"drain load failed: {e}")
