# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-request smoke for the TokenSpeed runtime image and Dynamo worker."""

import os
import sys
import uuid
from functools import partial

import pytest
import requests

from tests.utils.constants import DynamoPortRange
from tests.utils.http_checks import check_model_registered
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import reserved_ports

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.pre_merge
@pytest.mark.e2e
@pytest.mark.tokenspeed
@pytest.mark.gpu_1
@pytest.mark.model(MODEL)
@pytest.mark.timeout(600)
def test_tokenspeed_single_request(
    request,
    predownload_models,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
):
    """Load one small model on one GPU and complete one HTTP inference request."""
    ports = dynamo_dynamic_ports
    namespace = f"tokenspeed-smoke-{uuid.uuid4().hex}"
    env = {"DYN_NAMESPACE": namespace}
    worker_env = {
        **os.environ,
        **env,
        "DYN_SYSTEM_PORT": str(ports.system_ports[0]),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[
            0
        ],
    }
    base_url = f"http://127.0.0.1:{ports.frontend_port}"

    with reserved_ports(2, DynamoPortRange.SERVE.value) as engine_ports:
        command = [
            sys.executable,
            "-m",
            "dynamo.tokenspeed",
            "--model",
            MODEL,
            "--world-size",
            "1",
            "--port",
            str(engine_ports[0]),
            "--dist-init-addr",
            f"127.0.0.1:{engine_ports[1]}",
            "--max-model-len",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-num-seqs",
            "1",
            "--chunked-prefill-size",
            "512",
            "--max-prefill-tokens",
            "512",
            "--gpu-memory-utilization",
            "0.4",
            "--enforce-eager",
            "--disable-kvstore",
        ]
        with DynamoFrontendProcess(
            request, frontend_port=ports.frontend_port, extra_env=env
        ):
            with ManagedProcess(
                command=command,
                env=worker_env,
                display_name="tokenspeed-worker",
                log_dir=request.node.name,
                display_output=True,
                terminate_all_matching_process_names=False,
                timeout=420,
                # Readiness uses GET only; the POST below is the sole inference.
                health_check_urls=[
                    (
                        f"{base_url}/v1/models",
                        partial(check_model_registered, model=MODEL),
                    )
                ],
            ):
                response = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": MODEL,
                        "prompt": "The capital of France is",
                        "max_tokens": 16,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=120,
                )
                assert response.status_code == 200, response.text
                completion = response.json()
                assert len(completion["choices"]) == 1, completion
                choice = completion["choices"][0]
                assert choice["text"].strip(), completion
                assert choice["finish_reason"] in ("stop", "length"), completion
                assert completion["usage"]["prompt_tokens"] > 0, completion
                assert 0 < completion["usage"]["completion_tokens"] <= 16, completion
