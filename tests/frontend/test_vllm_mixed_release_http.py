# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, host-Docker mixed-release HTTP test with separately pinned runtimes.

Set DYNAMO_PROTOCOL_RELEASE_MATRIX to a JSON file with current/legacy objects
(image, python_source, wheel, wheel_sha256, runtime_version, engine_version,
source_commit) and model_cache (a host HF hub path). Paths are host paths, not
container paths. The caller must verify source trees against the recorded commits
and dirty-source artifacts before execution. Never substitute current bindings
for the legacy runtime: the purpose is to exercise the actual release boundary.
"""

import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.frontend.test_vllm_native_protocol_http import (
    KV_BYTES,
    MODEL,
    VOCAB_SIZE,
    _assert_native_prompt_count,
    _capture,
    _decoded,
    _engine_args,
    _prompt_count_cases,
)
from tests.utils.http_checks import models_available
from tests.utils.managed_process import ManagedProcess

pytestmark = [
    pytest.mark.vllm,
    # The host orchestrator does not import an engine. Each Docker subprocess
    # verifies its own pinned vLLM installation before it starts the service.
    pytest.mark.framework_agnostic,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.model(MODEL),
    pytest.mark.requested_vllm_kv_cache_bytes(KV_BYTES),
    # Measured across both processors/directions with pinned 1.4/1.5 releases.
    # Each native or worker engine runs separately with 256 MiB KV cache.
    pytest.mark.profiled_vram_gib(2.7),
    pytest.mark.timeout(900),
]


@pytest.fixture
def release_matrix():
    source = os.environ.get("DYNAMO_PROTOCOL_RELEASE_MATRIX")
    if source is None:
        pytest.skip("requires explicitly pinned host-Docker release matrix")
    matrix = json.loads(Path(source).read_text())
    for label in ("current", "legacy"):
        pin = matrix[label]
        if "@sha256:" not in pin["image"]:
            raise ValueError("release images must be pinned by digest")
        wheel = Path(pin["wheel"])
        checksum = hashlib.sha256()
        with wheel.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != pin["wheel_sha256"]:
            raise ValueError(f"{label} wheel hash mismatch")
        if not Path(pin["python_source"], "dynamo", "vllm", "main.py").is_file():
            raise ValueError(f"{label} Python source is missing")
    if matrix["legacy"]["runtime_version"] not in {"1.4.0", "1.5.0"}:
        raise ValueError("unsupported legacy release in test matrix")
    return matrix


@contextmanager
def release_process(pin, args, env, matrix, case_dir, label, ready):
    """Use shared process logging, plus explicit cleanup of the owned container."""
    name = f"protocol-release-{label}-{uuid.uuid4().hex}"
    wheel = Path(pin["wheel"])
    verify = (
        "import importlib.metadata as m; "
        f"assert m.version('ai-dynamo-runtime') == {pin['runtime_version']!r}; "
        f"assert m.version('vllm') == {pin['engine_version']!r}; "
        "print('runtime', m.version('ai-dynamo-runtime'), 'vllm', m.version('vllm'), flush=True)"
    )
    startup = " && ".join(
        [
            shlex.join(
                [
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    "/tmp/release-deps",
                    f"/wheels/{wheel.name}",
                ]
            ),
            shlex.join(
                [
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    "/tmp/release-deps",
                    "kubernetes==32.0.1",
                ]
            ),
            shlex.join(["python3", "-c", verify]),
            "exec " + shlex.join(["python3", *args]),
        ]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--runtime=nvidia",
        "--gpus",
        "device=0",
        "--shm-size=2g",
        "-v",
        f"{pin['python_source']}:/source:ro",
        "-v",
        f"{wheel.parent}:/wheels:ro",
        "-v",
        f"{matrix['model_cache']}:/model-cache:ro",
        "-v",
        f"{case_dir}:/case",
        "-e",
        "PYTHONPATH=/tmp/release-deps:/source",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_HUB_CACHE=/model-cache",
    ]
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(["--entrypoint", "/bin/bash", pin["image"], "-lc", startup])
    try:
        with ManagedProcess(
            command=command,
            health_check_urls=ready,
            timeout=300,
            terminate_all_matching_process_names=False,
            display_name=label,
            log_dir=str(case_dir / label),
        ):
            yield
    finally:
        # A killed Docker client need not stop its container. The exact unique
        # name belongs to this invocation, including when startup fails.
        subprocess.run(
            ["docker", "stop", "--time", "10", name],
            check=False,
            capture_output=True,
            timeout=30,
        )
        _wait_for_container_removal(name)


def _wait_for_container_removal(name):
    """Docker stop can return before --rm finishes; require observed removal."""
    deadline = time.monotonic() + 30
    while True:
        remaining = subprocess.run(
            ["docker", "inspect", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if remaining.returncode != 0:
            diagnostic = remaining.stderr.lower()
            if "no such object:" in diagnostic or "no such container:" in diagnostic:
                return
            # A daemon/access failure is not proof that the container vanished.
            raise RuntimeError(f"cannot verify test container removal: {name}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"test container was not removed: {name}")
        time.sleep(0.1)


def release_cases():
    for endpoint in ("chat/completions", "completions"):
        for stream in (False, True):
            base = {"model": MODEL, "max_tokens": 3, "temperature": 0, "stream": stream}
            if endpoint == "chat/completions":
                base["messages"] = [{"role": "user", "content": "Say hello."}]
            else:
                base["prompt"] = "Say hello."
            for extension in (False, True):
                payload = dict(base)
                if extension:
                    payload["allowed_token_ids"] = [0]
                yield (
                    f"{endpoint}-stream-{stream}-extension-{extension}",
                    endpoint,
                    payload,
                )


def response_text(record, endpoint):
    parts = []
    for item in _decoded(record):
        for choice in item["choices"]:
            if endpoint == "completions":
                parts.append(choice["text"])
            else:
                parts.append(
                    (choice.get("message") or choice.get("delta") or {}).get("content")
                    or ""
                )
    return "".join(parts)


def assert_full_vocab_boundary(record, *, direction):
    """A legacy boundary must reject the sentinel before opening an SSE stream.

    New frontends owe a feature-specific error. Released frontends cannot parse
    the signed public value at all; their parser error is not native parity.
    """
    assert record["status"] == 400, record
    assert record["content_type"].startswith("application/json"), record
    body = json.loads(record["body"])
    # Released frontends keep their flat envelope; current routes use the native
    # outer shape. Do not accept either shape indiscriminately and mask a regression.
    error = body.get("error", {}) if direction == "new-frontend" else body
    assert error.get("message"), record
    if direction == "new-frontend":
        assert set(body) == {"error"}, record
        assert error["code"] == 400 and error["type"] == "BadRequestError", record
        assert error["param"] == "prompt_logprobs", record
        assert "prompt_logprobs" in error["message"], record
    else:
        assert "expected u32" in error["message"] and "-1" in error["message"], record


@pytest.mark.parametrize("contract", ["sampling", "full-vocab"])
@pytest.mark.parametrize("direction", ["new-frontend", "old-frontend"])
@pytest.mark.parametrize("processor", ["dynamo", "vllm"])
def test_mixed_release_http(
    release_matrix, contract, direction, processor, dynamo_dynamic_ports, tmp_path
):
    matrix = release_matrix
    (tmp_path / "matrix.json").write_text(json.dumps(matrix, indent=2))
    frontend = matrix["current" if direction == "new-frontend" else "legacy"]
    worker = matrix["legacy" if direction == "new-frontend" else "current"]
    ports = dynamo_dynamic_ports
    port = ports.frontend_port
    namespace = f"mixed-{uuid.uuid4().hex}"
    env = {
        "DYN_NAMESPACE": namespace,
        "DYN_FILE_KV": "/case/discovery",
        "DYN_SYSTEM_PORT": str(ports.system_ports[0]),
        "DYN_FORWARDPASS_METRIC_PORT": str(ports.fpm_port),
    }
    engine_args = _engine_args()
    if contract == "full-vocab":
        engine_args.extend(["--max-logprobs", "-1"])
    runtime_args = [
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    ready = [(f"http://127.0.0.1:{port}/v1/models", models_available)]
    cases = list(release_cases())
    if contract == "full-vocab":
        cases = []
        for name, endpoint, payload in _prompt_count_cases():
            if payload["prompt_logprobs"] not in (None, -1):
                continue
            cases.append((name, endpoint, payload))
            if payload["prompt_logprobs"] is None:
                omitted = dict(payload)
                omitted.pop("prompt_logprobs")
                cases.append((name.replace("None", "omitted"), endpoint, omitted))
    response_suffix = ".json.gz" if contract == "full-vocab" else ".json"
    with release_process(
        worker,
        ["-m", "vllm.entrypoints.openai.api_server", "--port", str(port), *engine_args],
        env,
        matrix,
        tmp_path,
        "native",
        ready,
    ):
        native = _capture(port, cases, tmp_path / f"native-responses{response_suffix}")
    for name, endpoint, payload in cases:
        record = native[name]
        if contract != "full-vocab":
            assert record["status"] == 200, record
        elif payload["stream"] and payload.get("prompt_logprobs") == -1:
            # Error schemas differ across native releases; require the rejected
            # field and HTTP/JSON boundary, not the 0.30-only error.param shape.
            assert record["status"] == 400, record
            assert record["content_type"].startswith("application/json"), record
            assert "prompt_logprobs" in record["body"], record
        elif "prompt_logprobs" in payload:
            _assert_native_prompt_count(
                record,
                endpoint,
                max_logprobs=-1,
                vocab_size=VOCAB_SIZE,
                expected_prompt_tokens=None,
            )
        else:
            assert record["status"] == 200, record
            _decoded(record)

    # The worker survives frontend restarts, proving the policy lives in the
    # frontend rather than being written back into the worker/card.
    with release_process(
        worker,
        [
            "-m",
            "dynamo.vllm",
            *engine_args,
            *runtime_args,
            "--kv-events-config",
            '{"enable_kv_cache_events":false}',
        ],
        env,
        matrix,
        tmp_path,
        "worker",
        [],
    ):
        scopes = (
            ("exact", "wrong-model", "absent")
            if direction == "new-frontend"
            else ("absent",)
        )
        for scope in scopes:
            args = [
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(port),
                "--dyn-chat-processor",
                processor,
                *runtime_args,
            ]
            if scope != "absent":
                declaration = {
                    "namespace": namespace,
                    "component": "backend",
                    "endpoint": "generate",
                    "model": MODEL if scope == "exact" else "different-model",
                    "worker_type": "aggregated",
                    "dynamo_release": worker["runtime_version"],
                }
                args.extend(["--legacy-vllm-targets", json.dumps([declaration])])
            with release_process(
                frontend, args, env, matrix, tmp_path, f"frontend-{scope}", ready
            ):
                actual = _capture(port, cases, tmp_path / f"{scope}-responses.json")
            for name, endpoint, payload in cases:
                record = actual[name]
                missing_14_policy = (
                    direction == "new-frontend"
                    and worker["runtime_version"] == "1.4.0"
                    and scope != "exact"
                    and "allowed_token_ids" in payload
                )
                if contract == "full-vocab" and payload.get("prompt_logprobs") == -1:
                    assert_full_vocab_boundary(record, direction=direction)
                elif missing_14_policy:
                    assert record["status"] == 400, record
                    assert "allowed_token_ids" in record["body"], record
                else:
                    assert record["status"] == 200, record
                    assert response_text(record, endpoint) == response_text(
                        native[name], endpoint
                    ), record
                    if "allowed_token_ids" in payload:
                        assert response_text(record, endpoint) == "!!!", record
