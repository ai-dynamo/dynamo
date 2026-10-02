# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Rust sidecar/engine checks; requests bypass the Dynamo frontend."""

import contextlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from importlib.metadata import version as package_version
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tests.utils.gpu_args import map_cuda_visible_devices
from tests.utils.managed_process import ManagedProcess

MODEL = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.sidecar,
    pytest.mark.sidecar_native,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.post_merge,
    pytest.mark.nightly,
    pytest.mark.gpu_1,
    pytest.mark.model(MODEL),
    pytest.mark.timeout(900),
]


@pytest.mark.parametrize(
    "backend,scenario,engines",
    [
        pytest.param(
            backend,
            scenario,
            engines,
            marks=[
                getattr(pytest.mark, backend),
                pytest.mark.requested_vllm_kv_cache_bytes(1119388000)
                if backend == "vllm"
                else pytest.mark.requested_sglang_kv_tokens(8192 * engines),
                pytest.mark.profiled_vram_gib(
                    sglang_vram_gib if backend == "sglang" else vllm_vram_gib
                ),
            ],
            id=f"{backend}-{name}",
        )
        for backend in ("vllm", "sglang")
        for name, scenario, engines, vllm_vram_gib, sglang_vram_gib in (
            (
                "compatibility",
                "native_logprobs_and_structured_output_are_compatible",
                1,
                3.5,
                4.0,
            ),
            (
                "cancellation",
                "cancellation_and_consumer_drop_release_native_work",
                1,
                3.5,
                3.3,
            ),
            ("handoff", "handoff_transfers_native_kv", 2, 5.7, 6.6),
        )
    ],
)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_native_integration(
    backend, scenario, engines, tmp_path, dynamo_dynamic_ports, predownload_models
):
    binary = Path(os.environ["DYNAMO_SIDECAR_NATIVE_TEST"])
    assert binary.is_file(), f"Missing compiled native_engine test: {binary}"
    native = None
    if backend == "vllm":
        native = shutil.which("vllm-rs") or str(
            distribution("vllm").locate_file("vllm/vllm-rs")
        )
        assert Path(
            native
        ).is_file(), "The pinned vLLM release's vllm-rs executable is required"
        version = subprocess.run(
            [native, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        assert version == f"vllm-rs {package_version('vllm')}", version
    else:
        assert package_version("sglang") == "0.5.19"
    gpu_id = map_cuda_visible_devices([0], os.environ.get("CUDA_VISIBLE_DEVICES"))
    probe = tmp_path / "transfers.jsonl"
    probe.write_text("")
    model = os.environ.get("SIDECAR_NATIVE_MODEL_PATH", MODEL)
    environment = dict(
        os.environ,
        SIDECAR_NATIVE_MODEL="sidecar-native",
        SIDECAR_NATIVE_TRANSFER_PROBE=str(probe),
    )
    structured_tokens = tmp_path / "structured-tokens.json"
    if scenario == "native_logprobs_and_structured_output_are_compatible":
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        environment["SIDECAR_NATIVE_STRUCTURED_PROMPT"] = json.dumps(tokens)
        environment["SIDECAR_NATIVE_STRUCTURED_TOKENS"] = str(structured_tokens)
    with contextlib.ExitStack() as stack:
        for index in range(engines):
            http_port = dynamo_dynamic_ports.system_ports[index]
            grpc_port = dynamo_dynamic_ports.kv_event_ports[index]
            nixl_port = dynamo_dynamic_ports.nixl_side_channel_ports[index]
            if backend == "vllm":
                command = [
                    native,
                    "serve",
                    model,
                    "--served-model-name",
                    "sidecar-native",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(http_port),
                    "--grpc-port",
                    str(grpc_port),
                    "--max-model-len",
                    "8192",
                    "--",
                    "--enforce-eager",
                    "--max-num-seqs",
                    "2",
                    "--kv-cache-memory-bytes",
                    "1119388000",
                    "--gpu-memory-utilization",
                    "0.01",
                ]
                if scenario == "native_logprobs_and_structured_output_are_compatible":
                    separator = command.index("--")
                    command[separator:separator] = ["--reasoning-parser", "none"]
                if engines == 2:
                    command += [
                        "--kv-transfer-config",
                        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                        "--worker-extension-cls",
                        "native_probe.TransferProbe",
                    ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    model,
                    "--served-model-name",
                    "sidecar-native",
                    "--host",
                    "0.0.0.0" if engines == 2 else "127.0.0.1",
                    "--port",
                    str(http_port),
                    "--grpc-port",
                    str(grpc_port),
                    "--context-length",
                    "8192",
                    "--max-running-requests",
                    "2",
                    "--max-total-tokens",
                    "8192",
                    "--mem-fraction-static",
                    "0.2",
                    "--disable-cuda-graph",
                    "--enable-metrics",
                    "--decode-log-interval",
                    "1",
                    "--incremental-streaming-output",
                ]
                if engines == 2:
                    command += [
                        "--disaggregation-mode",
                        "decode" if index == 0 else "prefill",
                        "--disaggregation-transfer-backend",
                        "nixl",
                        "--disaggregation-bootstrap-port",
                        str(nixl_port),
                    ]
            env = dict(
                environment,
                CUDA_VISIBLE_DEVICES=gpu_id,
                VLLM_NIXL_SIDE_CHANNEL_PORT=str(nixl_port),
                VLLM_PLUGINS="",
                PYTHONPATH=os.pathsep.join(
                    [
                        str(ROOT / "lib/sidecar/testkit"),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            )
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    working_dir=str(tmp_path),
                    display_name=f"{backend}-native-{index}",
                    env=env,
                    health_check_urls=[f"http://127.0.0.1:{http_port}/health"],
                    timeout=600,
                    log_dir=str(tmp_path / f"engine-{index}"),
                    terminate_all_matching_process_names=False,
                )
            )
            key = "SIDECAR_NATIVE_GRPC" if index == 0 else "SIDECAR_NATIVE_PREFILL_GRPC"
            environment[key] = f"http://127.0.0.1:{grpc_port}"
            metrics_key = (
                "SIDECAR_NATIVE_METRICS"
                if index == 0
                else "SIDECAR_NATIVE_PREFILL_METRICS"
            )
            environment[metrics_key] = f"http://127.0.0.1:{http_port}/metrics"
        result = subprocess.run(
            [str(binary), f"{backend}_{scenario}", "--exact", "--nocapture"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed; 0 failed; 0 ignored;" in result.stdout, result.stdout
        if (
            backend == "sglang"
            and scenario == "native_logprobs_and_structured_output_are_compatible"
        ):
            output = tokenizer.decode(
                json.loads(structured_tokens.read_text()), skip_special_tokens=True
            )
            assert json.loads(output) == {"ok": True}, output
