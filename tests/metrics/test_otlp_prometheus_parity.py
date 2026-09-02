# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/metrics` and OTLP must show the same metrics in a real run.

The two surfaces are fed by independent callbacks -- the scrape appends the
engine's exposition text, the export takes the same metrics typed -- so nothing
in the type system stops them drifting apart. A metric that reaches one and not
the other is the failure this pins: it looks fine on a dashboard scraping
Prometheus while silently missing from the collector, or the reverse.

Runs against a real worker process rather than a constructed registry, because
drift shows up in wiring (a callback registered on one path only, an exporter
gated behind something that is off by default), not in the mapper.
"""

import contextlib
import logging
import os
import random
import socket
import threading
import time
from concurrent import futures
from typing import Optional

import pytest
import requests

from tests.utils.managed_process import ManagedProcess

# Imported through importorskip so that an environment without the OTLP
# receiver dependencies skips this module instead of failing collection for
# every test in the suite.
_OTLP = "opentelemetry.proto.collector.metrics.v1"
grpc = pytest.importorskip("grpc", reason="OTLP receiver needs grpcio")
metrics_service_pb2 = pytest.importorskip(
    f"{_OTLP}.metrics_service_pb2", reason="OTLP receiver needs opentelemetry-proto"
)
metrics_service_pb2_grpc = pytest.importorskip(
    f"{_OTLP}.metrics_service_pb2_grpc",
    reason="OTLP receiver needs opentelemetry-proto",
)

logger = logging.getLogger(__name__)

WORKER = os.path.join(os.path.dirname(__file__), "parity_worker.py")

# Families that legitimately exist on only one surface. Keep this list short and
# justified: every entry is a place the two surfaces genuinely disagree, and a
# growing list means the contract is eroding.
#
# `target_info` is synthesised by OTLP consumers from resource attributes, not
# collected, so it has no Prometheus counterpart.
OTLP_ONLY = {"target_info"}


def _free_port() -> int:
    """A free port that fits in an i16.

    `DYN_SYSTEM_PORT` is parsed as i16, so the ephemeral range the kernel hands
    out is often too high and the runtime refuses the config.
    """
    for _ in range(200):
        candidate = random.randint(20000, 32000)
        with contextlib.closing(socket.socket()) as s:
            try:
                s.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("no free port below the i16 ceiling")


class _OtlpReceiver(metrics_service_pb2_grpc.MetricsServiceServicer):
    """Minimal OTLP/gRPC metrics collector that records what it is sent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: set[str] = set()
        self._exports = 0

    def Export(self, request, context):  # noqa: N802 - gRPC method name
        with self._lock:
            self._exports += 1
            for resource in request.resource_metrics:
                for scope in resource.scope_metrics:
                    for metric in scope.metrics:
                        self._names.add(metric.name)
        return metrics_service_pb2.ExportMetricsServiceResponse()

    @property
    def exports(self) -> int:
        with self._lock:
            return self._exports

    def names(self) -> set[str]:
        with self._lock:
            return set(self._names)

    def wait_for_export(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.exports > 0:
                return True
            time.sleep(0.25)
        return False


@contextlib.contextmanager
def _running_receiver(port: int):
    receiver = _OtlpReceiver()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(receiver, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    try:
        yield receiver
    finally:
        server.stop(grace=None)


def _prometheus_families(text: str) -> set[str]:
    """Family names as declared by ``# TYPE``.

    Compares families, not sample lines: a histogram renders as ``_bucket`` /
    ``_sum`` / ``_count`` on the scrape but is one metric in OTLP, so sample
    names would report drift that is only a representation difference.
    """
    return {
        line.split()[2]
        for line in text.splitlines()
        if line.startswith("# TYPE ") and len(line.split()) >= 4
    }


class _Worker(ManagedProcess):
    def __init__(self, request, system_port: int, otlp_port: int):
        env = os.environ.copy()
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        # Export far more often than the 60s default so the test does not have
        # to wait a minute for the first payload.
        env["OTEL_METRICS_EXPORTER"] = "otlp"
        env["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] = f"http://127.0.0.1:{otlp_port}"
        env["OTEL_METRIC_EXPORT_INTERVAL"] = "1000"

        super().__init__(
            command=["python3", WORKER],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready)
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["parity_worker.py"],
            log_dir=f"{request.node.name}_otlp_parity",
        )

    @staticmethod
    def _is_ready(response) -> bool:
        try:
            return (response.json() or {}).get("status") == "ready"
        except ValueError:
            return False


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.e2e
def test_otlp_and_prometheus_expose_the_same_metrics(request, runtime_services):
    system_port = _free_port()
    otlp_port = _free_port()

    with _running_receiver(otlp_port) as receiver:
        with _Worker(request, system_port, otlp_port):
            assert receiver.wait_for_export(
                timeout=60
            ), "worker never exported over OTLP; is export still wired to runtime startup?"

            # Scrape after an export so both surfaces describe the same process
            # at roughly the same point in its life. A family registered
            # between the two reads shows up as drift, so allow one retry.
            drift: Optional[str] = None
            for _ in range(3):
                scrape = requests.get(
                    f"http://localhost:{system_port}/metrics", timeout=10
                )
                scrape.raise_for_status()
                prometheus = _prometheus_families(scrape.text)
                exported = receiver.names() - OTLP_ONLY

                assert prometheus, "no families on /metrics; the scrape path is broken"

                missing_from_otlp = prometheus - exported
                missing_from_prometheus = exported - prometheus
                if not missing_from_otlp and not missing_from_prometheus:
                    drift = None
                    break

                drift = (
                    f"on /metrics but not exported: {sorted(missing_from_otlp)}\n"
                    f"exported but not on /metrics: {sorted(missing_from_prometheus)}"
                )
                time.sleep(2)

            assert drift is None, (
                "OTLP and /metrics disagree about which metrics exist.\n"
                "One surface is silently missing metrics the other reports.\n"
                f"{drift}"
            )
