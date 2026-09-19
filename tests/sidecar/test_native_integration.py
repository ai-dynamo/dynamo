# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Rust sidecar/engine checks; requests bypass the Dynamo frontend."""

import contextlib
import json
import os
import shutil
import subprocess
from importlib.metadata import distribution
from importlib.metadata import version as package_version
from pathlib import Path

import pytest

from tests.utils.managed_process import ManagedProcess

MODEL = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.vllm,
    pytest.mark.sidecar,
    pytest.mark.sidecar_native,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.post_merge,
    pytest.mark.nightly,
    pytest.mark.model(MODEL),
    pytest.mark.timeout(900),
    pytest.mark.requested_vllm_kv_cache_bytes(1119388000),
]


@pytest.mark.parametrize(
    "scenario,engines",
    [
        pytest.param(
            "vllm_native_logprobs_and_structured_output_are_compatible",
            1,
            marks=[pytest.mark.gpu_1, pytest.mark.profiled_vram_gib(3.5)],
            id="compatibility",
        ),
        pytest.param(
            "vllm_cancellation_and_consumer_drop_release_native_work",
            1,
            marks=[pytest.mark.gpu_1, pytest.mark.profiled_vram_gib(3.5)],
            id="cancellation",
        ),
        pytest.param(
            "vllm_handoff_transfers_native_kv", 2, marks=pytest.mark.gpu_2, id="handoff"
        ),
    ],
)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_native_integration(scenario, engines, tmp_path, dynamo_dynamic_ports):
    binary = Path(os.environ["DYNAMO_SIDECAR_NATIVE_TEST"])
    assert binary.is_file(), f"Missing compiled native_engine test: {binary}"
    native = shutil.which("vllm-rs") or str(
        distribution("vllm").locate_file("vllm/vllm-rs")
    )
    assert Path(
        native
    ).is_file(), "The pinned vLLM release's vllm-rs executable is required"
    version = subprocess.run(
        [native, "--version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()
    assert version == "vllm-rs 0.29.0", version
    assert package_version("vllm") == "0.29.0"
    gpu_ids = os.environ.get(
        "SIDECAR_NATIVE_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    ).split(",")
    assert len(gpu_ids) >= engines and all(
        gpu_ids
    ), "Insufficient assigned GPUs for native integration"
    probe = tmp_path / "transfers.jsonl"
    probe.write_text("")
    model = os.environ.get("SIDECAR_NATIVE_MODEL_PATH", MODEL)
    environment = dict(
        os.environ,
        SIDECAR_NATIVE_MODEL="sidecar-native",
        SIDECAR_NATIVE_TRANSFER_PROBE=str(probe),
    )
    if scenario == "vllm_native_logprobs_and_structured_output_are_compatible":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        environment["SIDECAR_NATIVE_STRUCTURED_PROMPT"] = json.dumps(tokens)
    with contextlib.ExitStack() as stack:
        for index in range(engines):
            http_port = dynamo_dynamic_ports.system_ports[index]
            grpc_port = dynamo_dynamic_ports.kv_event_ports[index]
            nixl_port = dynamo_dynamic_ports.nixl_side_channel_ports[index]
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
            if scenario == "vllm_native_logprobs_and_structured_output_are_compatible":
                separator = command.index("--")
                command[separator:separator] = ["--reasoning-parser", "none"]
            if engines == 2:
                command += [
                    "--kv-transfer-config",
                    '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                    "--worker-extension-cls",
                    "native_probe.TransferProbe",
                ]
            env = dict(
                environment,
                CUDA_VISIBLE_DEVICES=gpu_ids[index],
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
                    display_name=f"vllm-native-{index}",
                    env=env,
                    health_check_urls=[f"http://127.0.0.1:{http_port}/health"],
                    timeout=600,
                    log_dir=str(tmp_path / f"engine-{index}"),
                    terminate_all_matching_process_names=False,
                )
            )
            key = "SIDECAR_NATIVE_GRPC" if index == 0 else "SIDECAR_NATIVE_PREFILL_GRPC"
            environment[key] = f"http://127.0.0.1:{grpc_port}"
            if index == 0:
                environment[
                    "SIDECAR_NATIVE_METRICS"
                ] = f"http://127.0.0.1:{http_port}/metrics"
        result = subprocess.run(
            [str(binary), scenario, "--exact", "--nocapture"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed; 0 failed; 0 ignored;" in result.stdout, result.stdout
