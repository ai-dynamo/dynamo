# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real vLLM qualification of request-time plugin resolution and recovery."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
import requests
from huggingface_hub import snapshot_download

from tests.serve.runtime_lora_plugin import (
    ADAPTER_REPO,
    BASE_MODEL,
    CHANGED_SOURCE,
    ERROR_SOURCE_PREFIX,
    adapter_source,
)
from tests.utils.managed_process import ManagedProcess

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.model(BASE_MODEL),
    pytest.mark.model(ADAPTER_REPO),
    pytest.mark.profiled_vram_gib(3.8),
    pytest.mark.requested_vllm_kv_cache_bytes(1_073_741_824),
    pytest.mark.timeout(900),
]


@pytest.fixture
def runtime_lora_server(tmp_path, dynamo_dynamic_ports, predownload_models):
    adapter = snapshot_download(repo_id=ADAPTER_REPO, local_files_only=True)
    revision = Path(adapter).name
    base = snapshot_download(repo_id=BASE_MODEL, local_files_only=True)
    repo = Path(__file__).resolve().parents[2]
    base_url = f"http://127.0.0.1:{dynamo_dynamic_ports.frontend_port}"
    cache = tmp_path / "lora-cache"
    cache.mkdir()
    calls = tmp_path / "resolver-calls.jsonl"
    env = {
        **os.environ,
        "DYN_FILE_KV": str(tmp_path / "discovery"),
        "DYN_LORA_ENABLED": "true",
        "DYN_LORA_RUNTIME_LOAD_ENABLED": "true",
        "DYN_LORA_DOWNLOADER_PLUGIN": "tests.serve.runtime_lora_plugin:OfflineResolver",
        "DYN_LORA_ALLOWED_SCHEMES": "wandb-artifact",
        "DYN_LORA_PATH": str(cache),
        "DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS": "1",
        "DYN_LORA_MAX_CONCURRENT_RESOLUTIONS": "2",
        "DYN_LORA_MAX_PENDING_ADMISSIONS": "2",
        "DYN_LORA_RESOLVE_TIMEOUT_SECONDS": "5",
        "DYN_SYSTEM_ENABLED": "true",
        "DYN_SYSTEM_PORT": str(dynamo_dynamic_ports.system_ports[0]),
        "DYNAMO_TEST_LORA_ADAPTER": adapter,
        "DYNAMO_TEST_LORA_REVISION": revision,
        "DYNAMO_TEST_LORA_CALLS": str(calls),
        "HF_HUB_OFFLINE": "1",
        "VLLM_CACHE_ROOT": str(tmp_path / "vllm-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "torch-cache"),
    }
    env.pop("NATS_SERVER", None)
    planes = [
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]

    def registered():
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=3)
            return response.ok and any(
                model["id"] == BASE_MODEL for model in response.json().get("data", [])
            )
        except (requests.RequestException, ValueError):
            return False

    with ExitStack() as stack:
        stack.enter_context(
            ManagedProcess(
                command=[
                    sys.executable,
                    "-m",
                    "dynamo.frontend",
                    "--http-port",
                    str(dynamo_dynamic_ports.frontend_port),
                    "--router-mode",
                    "round-robin",
                    *planes,
                ],
                env=env,
                working_dir=str(repo),
                health_check_ports=[dynamo_dynamic_ports.frontend_port],
                terminate_all_matching_process_names=False,
                timeout=180,
                log_dir=str(tmp_path / "logs"),
                display_name="runtime-lora-frontend",
            )
        )
        stack.enter_context(
            ManagedProcess(
                command=[
                    sys.executable,
                    "-m",
                    "dynamo.vllm",
                    "--model",
                    base,
                    "--served-model-name",
                    BASE_MODEL,
                    "--enforce-eager",
                    "--max-model-len",
                    "1024",
                    "--max-num-seqs",
                    "2",
                    "--kv-cache-memory-bytes",
                    "1073741824",
                    "--gpu-memory-utilization",
                    "0.05",
                    "--enable-lora",
                    "--max-lora-rank",
                    "64",
                    "--max-loras",
                    "1",
                    "--max-cpu-loras",
                    "1",
                    *planes,
                ],
                env=env,
                working_dir=str(repo),
                health_check_funcs=[registered],
                terminate_all_matching_process_names=False,
                timeout=420,
                log_dir=str(tmp_path / "logs"),
                display_name="runtime-lora-worker",
            )
        )
        yield base_url, calls, revision, Path(base).name


def _request(base_url, model, *, endpoint="chat/completions", request_id=None, **extra):
    payload = {"model": model, "max_tokens": 24, "temperature": 0, **extra}
    if endpoint == "chat/completions":
        payload.update(
            messages=[{"role": "user", "content": "What is two plus two?"}],
            chat_template_kwargs={"enable_thinking": False},
        )
    else:
        payload["prompt"] = "Two plus two is"
    return requests.post(
        f"{base_url}/v1/{endpoint}",
        json=payload,
        headers={"request-id": request_id or str(uuid4())},
        stream=extra.get("stream", False),
        timeout=180,
    )


def _assert_chat(base_url, model, **kwargs):
    with _request(base_url, model, **kwargs) as response:
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == model
        assert body["choices"][0]["message"]["content"]


def _calls(path):
    return Counter(json.loads(line)["source"] for line in path.read_text().splitlines())


def test_runtime_lora_plugin_serving_and_recovery(runtime_lora_server, request):
    base_url, calls, revision, base_revision = runtime_lora_server
    request.node.user_properties = [
        *request.node.user_properties,
        ("base_model", BASE_MODEL),
        ("base_revision", base_revision),
        ("adapter_model", ADAPTER_REPO),
        ("adapter_revision", revision),
        ("serving_mode", "aggregated-dynamo.vllm"),
    ]
    source_a = adapter_source(revision, "replica-a")
    source_b = adapter_source(revision, "replica-b")
    _assert_chat(base_url, BASE_MODEL)
    shared_request_id = str(uuid4())
    barrier = Barrier(2)

    def same_id_request(model):
        barrier.wait(timeout=10)
        _assert_chat(base_url, model, request_id=shared_request_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(same_id_request, model)
            for model in (f"{BASE_MODEL}|{source_a}", source_a)
        ]
        for future in futures:
            future.result(timeout=180)
    assert _calls(calls)[source_a] == 1

    with _request(base_url, source_a, endpoint="completions") as response:
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"]
    with _request(base_url, source_a, stream=True) as response:
        assert response.status_code == 200, response.text
        lines = [line for line in response.iter_lines() if line]
        assert lines[-1] == b"data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines if line.startswith(b"data: {")]
        assert any(
            choice.get("delta", {}).get("content")
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert all(chunk["model"] == source_a for chunk in chunks)
    assert _calls(calls)[source_a] == 1

    _assert_chat(base_url, source_b)
    _assert_chat(base_url, source_a)
    assert _calls(calls)[source_a] == 2
    assert _calls(calls)[source_b] == 1

    for model, status in (
        (f"{BASE_MODEL}|unsupported:///reference", 400),
        (f"unknown/base|{source_a}", 404),
        (ERROR_SOURCE_PREFIX + "missing", 404),
        (ERROR_SOURCE_PREFIX + "unavailable", 503),
        (ERROR_SOURCE_PREFIX + "timeout", 504),
        (ERROR_SOURCE_PREFIX + "malformed", 422),
        (ERROR_SOURCE_PREFIX + "invalid-result", 500),
    ):
        for streaming in (False, True):
            with _request(base_url, model, stream=streaming) as response:
                assert response.status_code == status, response.text
                assert model not in response.text
                assert "test provider private reference" not in response.text
        _assert_chat(base_url, source_a)

    with _request(
        base_url,
        ERROR_SOURCE_PREFIX + "missing",
        endpoint="completions",
        stream=True,
        nvext={"annotations": ["request_id"] * 17},
    ) as response:
        assert response.status_code == 404, response.text

    _assert_chat(base_url, CHANGED_SOURCE)
    _assert_chat(base_url, source_b)
    with _request(base_url, CHANGED_SOURCE) as response:
        assert response.status_code == 500, response.text
        assert CHANGED_SOURCE not in response.text
    _assert_chat(base_url, source_b)
    _assert_chat(base_url, BASE_MODEL)
    request.node.user_properties = [
        *request.node.user_properties,
        ("resolver_invocations", json.dumps(_calls(calls), sort_keys=True)),
    ]
