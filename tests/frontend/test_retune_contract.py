# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the semantic Retune contract across real frontend/mocker processes."""

import gzip
import hashlib
import json
import uuid
from pathlib import Path

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN, DynamoPortRange
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.model(QWEN),
    pytest.mark.timeout(180),
]


def test_frontend_acknowledges_retune_and_records_replay_traces(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    tmp_path: Path,
):
    ports = dynamo_dynamic_ports
    namespace = f"retune-{uuid.uuid4().hex}"
    config_hash = hashlib.sha256(namespace.encode()).hexdigest()
    trace_root = tmp_path / "trace"
    with reserved_ports(1, DynamoPortRange.FRONTEND.value) as (admin_port,):
        contract = {
            "version": 1,
            "configHash": config_hash,
            "frontendComponentName": "gateway",
            "adminPort": admin_port,
            "traceStorage": {
                "claimName": "request-traces",
                "mountPath": str(trace_root),
                "segmentsDir": "segments",
                "windowsDir": "windows",
            },
            "router": {
                "profile": "kv-v1",
                "overlapScoreCredit": 1.5,
                "prefillLoadScale": 0.7,
                "temperature": 0.2,
            },
        }
        admin_url = f"http://localhost:{admin_port}/router_policy"
        frontend = DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            router_mode="kv",
            extra_args=["--namespace", namespace],
            extra_env={
                "DYN_FRONTEND_RETUNE_CAPABILITY": json.dumps(contract),
                "DYN_ENABLE_FRONTEND_ADMIN_API": "1",
                "DYN_FRONTEND_ADMIN_HOST": "0.0.0.0",
                "DYN_FRONTEND_ADMIN_PORT": str(admin_port),
                "DYN_REQUEST_TRACE": "1",
                "DYN_REQUEST_TRACE_RECORDS": "request_end",
                "DYN_REQUEST_TRACE_SINKS": "file",
                "DYN_REQUEST_TRACE_FILE_FORMAT": "jsonl_gz",
                "DYN_REQUEST_TRACE_FILE_PATH": str(
                    trace_root / "segments" / "graph" / "gateway" / "req"
                ),
            },
        )
        worker = MockerWorkerProcess(
            request,
            QWEN,
            ports.frontend_port,
            ports.system_ports[0],
            extra_env={"DYN_NAMESPACE": namespace},
        )
        worker.health_check_urls.append(
            (
                admin_url,
                lambda response: response.json().get("retune")
                == {"configHash": config_hash},
            )
        )

        with frontend, worker:
            wait_for_http_completions_ready(
                frontend_port=ports.frontend_port, model=QWEN
            )
            acknowledgment = requests.get(admin_url, timeout=10)
            acknowledgment.raise_for_status()
            payload = acknowledgment.json()
            assert payload["retune"] == {"configHash": config_hash}
            assert payload["routers"]
            for router in payload["routers"]:
                assert router["namespace"] == namespace
                parameters = router["configuration"]["parameters"]
                assert parameters["overlap_score_credit"] == pytest.approx(1.5)
                assert parameters["prefill_load_scale"] == pytest.approx(0.7)
                assert parameters["router_temperature"] == pytest.approx(0.2)

    # Teardown drains and closes the gzip sink before the files are read.
    records = []
    for segment in trace_root.rglob("*.jsonl.gz"):
        with gzip.open(segment, "rt") as source:
            records.extend(json.loads(line)["event"] for line in source)
    completed = [record for record in records if record["event_type"] == "request_end"]
    assert completed, "the real completion must produce replayable request_end capture"
    assert any(
        record["request"]["input_tokens"] > 0
        and record["request"]["output_tokens"] > 0
        and record["request"]["replay"]["trace_block_size"] > 0
        for record in completed
    )
