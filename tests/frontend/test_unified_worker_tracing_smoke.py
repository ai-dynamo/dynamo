# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the unified-backend worker tracing subscriber.

A silent regression once removed `logging::init()` from the PyO3
`Worker.run()` path. Workers running under `OTEL_EXPORT_ENABLED=1`
spawned with NO tracing subscriber → zero spans, zero log lines, zero
observability. The frontend stayed functional, so request-flow tests
passed while telemetry was fully broken.

Asserts the `engine.generate` `SPAN_FIRST_ENTRY` event lands in the
worker's JSONL log with its declared attributes. Catches future
regressions that drop the subscriber install or the auto-span schema.
"""

from __future__ import annotations

import time

import pytest

from tests.frontend.conftest import (
    SampleUnifiedWorkerProcess,
    wait_for_http_completions_ready,
)
from tests.frontend.test_request_tracing_logs import (
    _send_chat_completions,
    parse_jsonl_logs,
    read_log_file,
)
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
    pytest.mark.timeout(180),
]


# OTLP endpoint points at an unreachable address — batches drop silently,
# but the worker MUST still install the tracing subscriber and emit JSONL.
# That's the regression we're catching.
#
# DYN_LOGGING_SPAN_EVENTS=1 turns on the `SPAN_FIRST_ENTRY` / `SPAN_CLOSED`
# JSONL events that we assert on. Without it the subscriber installs but
# never emits per-span event lines — defeating the regression check.
_OTEL_ENV = {
    "OTEL_EXPORT_ENABLED": "1",
    "DYN_LOGGING_JSONL": "1",
    "DYN_LOGGING_SPAN_EVENTS": "1",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:1",
}


def _find_engine_generate_span_entries(entries):
    """Return all SPAN_FIRST_ENTRY events for the `engine.generate` span."""
    return [
        e
        for e in entries
        if e.get("message") == "SPAN_FIRST_ENTRY"
        and e.get("span_name") == "engine.generate"
    ]


def test_unified_worker_emits_engine_generate_span(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_tokenizers
):
    """Aggregated unified worker must emit at least one engine.generate
    span event under OTEL_EXPORT_ENABLED=1.
    """
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=_OTEL_ENV,
        terminate_all_matching_process_names=False,
    ):
        with SampleUnifiedWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
            model_name=TEST_MODEL,
            component="sample",
            disaggregation_mode="agg",
            extra_env=_OTEL_ENV,
            worker_id="sample-agg",
        ) as worker:
            wait_for_http_completions_ready(frontend_port=frontend_port, model=TEST_MODEL)

            resp = _send_chat_completions(frontend_port, model=TEST_MODEL, max_tokens=5)
            assert resp.status_code == 200, f"curl failed: {resp.status_code} {resp.text!r}"

            # Poll the JSONL log for the span event — flush timing varies.
            engine_spans = []
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                entries = parse_jsonl_logs(read_log_file(worker))
                engine_spans = _find_engine_generate_span_entries(entries)
                if engine_spans:
                    break
                time.sleep(0.5)
            assert engine_spans, (
                "worker emitted zero `engine.generate` SPAN_FIRST_ENTRY events. "
                "This usually means the tracing subscriber was not installed — "
                "check lib/bindings/python/rust/backend.rs for logging::init() "
                "regression."
            )

            # Verify declared attrs are present on the span (proves
            # EngineAdapter still opens the span with its full schema).
            first = engine_spans[0]
            for required in ("model", "input_tokens", "disagg_role"):
                assert required in first, (
                    f"engine.generate SPAN_FIRST_ENTRY missing `{required}` "
                    f"attribute. Got keys: {sorted(first.keys())}"
                )
            assert first["disagg_role"] == "agg", (
                f"expected disagg_role=agg, got {first['disagg_role']!r}"
            )
