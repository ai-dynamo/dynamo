# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Rust sidecar/engine checks; requests bypass the Dynamo frontend."""

import base64
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from importlib.metadata import version as package_version
from pathlib import Path

import psutil
import pytest
from transformers import AutoTokenizer

from tests.utils.gpu_args import map_cuda_visible_devices
from tests.utils.managed_process import ManagedProcess

MODEL = "Qwen/Qwen3-0.6B"
MM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
LORA = "codelion/Qwen3-0.6B-accuracy-recovery-lora"
ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.sidecar,
    pytest.mark.sidecar_native,
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.nightly,
    pytest.mark.timeout(900),
]


@pytest.mark.parametrize(
    "backend,scenario,engines,devices",
    [
        pytest.param(
            backend,
            scenario,
            engines,
            devices,
            marks=[
                getattr(pytest.mark, backend),
                getattr(pytest.mark, f"gpu_{devices}"),
                (
                    pytest.mark.multimodal
                    if ("raw_image" in scenario or "encoder" in scenario)
                    else pytest.mark.core
                ),
                pytest.mark.model(
                    MM_MODEL
                    if ("raw_image" in scenario or "encoder" in scenario)
                    else MODEL
                ),
                (
                    pytest.mark.requested_vllm_kv_cache_bytes(1119388000)
                    if backend == "vllm"
                    else pytest.mark.requested_sglang_kv_tokens(8192 * engines)
                ),
                *(
                    [
                        pytest.mark.profiled_vram_gib(
                            sglang_vram_gib if backend == "sglang" else vllm_vram_gib
                        )
                    ]
                    if devices == 1
                    and (sglang_vram_gib if backend == "sglang" else vllm_vram_gib)
                    is not None
                    else []
                ),
                *([pytest.mark.model(LORA)] if "lora" in scenario else []),
            ],
            id=f"{backend}-{name}" + ("-two-gpu" if devices == 2 else ""),
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
            ("handoff", "handoff_transfers_native_kv", 2, 6.3, 6.6),
            (
                "lora",
                (
                    "lora_lifecycle_selects_native_adapter"
                    if backend == "vllm"
                    else "preloaded_lora_selects_native_adapter"
                ),
                1,
                3.7,
                4.5,
            ),
            *(
                [
                    (
                        "native-http",
                        "native_http_stream_cancel_and_drop_recover",
                        1,
                        None,
                        3.9,
                    ),
                    (
                        "managed-launch",
                        "managed_sidecar_serves_through_worker_ingress",
                        1,
                        None,
                        3.9,
                    ),
                ]
                if backend == "sglang"
                else []
            ),
            *(
                [
                    (
                        "rl-controls",
                        "native_pause_sleep_and_weight_version_recover",
                        1,
                        3.5,
                        None,
                    ),
                    ("image", "native_raw_image_is_accepted", 1, 7.1, None),
                    (
                        "encoder-aggregated",
                        "native_encoder_handoff_to_aggregated",
                        2,
                        8.4,
                        None,
                    ),
                    (
                        "encoder-disaggregated",
                        "native_encoder_handoff_to_prefill_decode",
                        3,
                        14.5,
                        None,
                    ),
                ]
                if backend == "vllm"
                else []
            ),
        )
        for devices in ((1, 2) if name == "handoff" else (1,))
    ],
)
@pytest.mark.parametrize("num_system_ports", [3], indirect=True)
def test_native_integration(
    backend,
    scenario,
    engines,
    devices,
    tmp_path,
    dynamo_dynamic_ports,
    predownload_models,
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
    has_encoder = "encoder_handoff" in scenario
    is_multimodal = "raw_image" in scenario or has_encoder
    gpu_ids = map_cuda_visible_devices(
        list(range(devices)), os.environ.get("CUDA_VISIBLE_DEVICES")
    ).split(",")
    assert len(set(gpu_ids)) == devices, "handoff requires distinct assigned GPUs"
    probe = tmp_path / "transfers.jsonl"
    probe.write_text("")
    gate = tmp_path / "transfer-gate"
    gate.mkdir()
    hook = tmp_path / "native-hook"
    hook.mkdir()
    if engines == 2 and backend == "sglang":
        (hook / "sitecustomize.py").write_text("import native_probe\n")
    model = (
        os.environ.get("SIDECAR_NATIVE_MM_MODEL_PATH", MM_MODEL)
        if is_multimodal
        else os.environ.get("SIDECAR_NATIVE_MODEL_PATH", MODEL)
    )
    environment = dict(
        os.environ,
        SIDECAR_NATIVE_MODEL="sidecar-native",
        SIDECAR_NATIVE_TRANSFER_PROBE=str(probe),
        SIDECAR_NATIVE_BACKEND=backend,
    )
    is_managed_sidecar = scenario == "managed_sidecar_serves_through_worker_ingress"
    if is_managed_sidecar:
        environment.update(
            DYN_DISCOVERY_BACKEND="file",
            DYN_FILE_KV=str(tmp_path / "discovery"),
            DYN_REQUEST_PLANE="tcp",
            DYN_EVENT_PLANE="zmq",
            DYN_NAMESPACE=f"native-managed-{tmp_path.name}",
            DYN_SYSTEM_HOST="127.0.0.1",
            DYN_SYSTEM_PORT="0",
            DYN_HEALTH_CHECK_ENABLED="false",
            DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS="0",
            DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT="2",
            DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS="3",
        )
    if is_multimodal:
        from PIL import Image
        from transformers import AutoProcessor

        image = Image.new("RGB", (64, 64), "red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        environment["SIDECAR_NATIVE_NVTX_PROBE"] = str(tmp_path / "nvtx.jsonl")
        environment["SIDECAR_NATIVE_IMAGE"] = (
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        environment["SIDECAR_NATIVE_IMAGE_PROMPT"] = json.dumps(
            processor.tokenizer.encode(prompt, add_special_tokens=False)
        )
        from vllm.config import ModelConfig
        from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
            extract_mm_features,
        )
        from vllm.multimodal import MULTIMODAL_REGISTRY

        native_processor = MULTIMODAL_REGISTRY.create_processor(
            ModelConfig(
                model=model,
                runner="generate",
                max_model_len=8192,
                limit_mm_per_prompt={"image": 1, "video": 0},
                dtype="bfloat16",
            )
        )
        processed = native_processor(
            prompt,
            mm_items=native_processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={},
        )
        features_path = tmp_path / "preprocessed-image.json"
        features_path.write_text(
            json.dumps(
                {
                    "tokens": processed["prompt_token_ids"],
                    "features": extract_mm_features(processed).model_dump(),
                }
            )
        )
        environment["SIDECAR_NATIVE_IMAGE_FEATURES"] = str(features_path)
    structured_tokens = tmp_path / "structured-tokens.json"
    if (
        scenario == "native_logprobs_and_structured_output_are_compatible"
        or "lora" in scenario
    ):
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
        environment["SIDECAR_NATIVE_LORA_PROMPT"] = json.dumps(tokens)
    if "lora" in scenario:
        from huggingface_hub import snapshot_download

        adapter_path = snapshot_download(LORA, local_files_only=True)
        environment["SIDECAR_NATIVE_LORA_PATH"] = adapter_path
        environment["DYN_LORA_ENABLED"] = "true"
        if backend == "vllm":
            environment["VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES"] = adapter_path
    ec_storage = tmp_path / "encoder-cache"
    if has_encoder:
        ec_storage.mkdir()
        environment["SIDECAR_NATIVE_EC_STORAGE"] = str(ec_storage)
        environment["SIDECAR_NATIVE_EC_PROBE"] = str(tmp_path / "ec-loads.jsonl")
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
                if "lora" in scenario:
                    command += [
                        "--enable-lora",
                        "--max-loras",
                        "2",
                        "--max-lora-rank",
                        "64",
                    ]
                if scenario == "native_pause_sleep_and_weight_version_recover":
                    command += ["--enable-sleep-mode"]
                if is_multimodal:
                    command += [
                        "--limit-mm-per-prompt",
                        '{"image":1,"video":0}',
                        "--skip-mm-profiling",
                    ]
                if "raw_image" in scenario:
                    command += [
                        "--enable-layerwise-nvtx-tracing",
                        "--worker-extension-cls",
                        "native_probe.TransferProbe",
                    ]
                if has_encoder:
                    command += ["--worker-extension-cls", "native_probe.TransferProbe"]
                    if index < 2:
                        command += [
                            "--ec-transfer-config",
                            json.dumps(
                                {
                                    "ec_connector": "ECExampleConnector",
                                    "ec_role": (
                                        "ec_producer" if index == 0 else "ec_consumer"
                                    ),
                                    "ec_connector_extra_config": {
                                        "shared_storage_path": str(ec_storage)
                                    },
                                }
                            ),
                        ]
                    if index == 0:
                        cache_flag = command.index("--kv-cache-memory-bytes")
                        del command[cache_flag : cache_flag + 2]
                        command[command.index("--gpu-memory-utilization") + 1] = "0.05"
                        command += ["--mm-encoder-only", "--no-enable-prefix-caching"]
                if (engines == 2 and not has_encoder) or (
                    has_encoder and engines == 3 and index > 0
                ):
                    command += [
                        "--kv-transfer-config",
                        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
                    ]
                    if not has_encoder:
                        command += [
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
                if "lora" in scenario:
                    command += [
                        "--enable-lora",
                        "--max-lora-rank",
                        "64",
                        "--lora-paths",
                        f"native-adapter={adapter_path}",
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
                if is_managed_sidecar:
                    command += ["--sidecar", "dynamo.sglang.sidecar"]
            env = dict(
                environment,
                CUDA_VISIBLE_DEVICES=gpu_ids[index % devices],
                VLLM_NIXL_SIDE_CHANNEL_PORT=str(nixl_port),
                VLLM_PLUGINS="",
                PYTHONPATH=os.pathsep.join(
                    [
                        str(ROOT / "lib/sidecar/testkit"),
                        str(hook),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            )
            if backend == "vllm" and (
                "raw_image" in scenario or (has_encoder and index == 0)
            ):
                env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            if (
                engines == 2
                and not has_encoder
                and index == (0 if backend == "vllm" else 1)
            ):
                env["SIDECAR_NATIVE_TRANSFER_GATE"] = str(gate)
            health_urls = [f"http://127.0.0.1:{http_port}/health"]
            if is_managed_sidecar:
                env["DYN_SYSTEM_PORT"] = str(dynamo_dynamic_ports.system_ports[1])
                health_urls.append(f"http://127.0.0.1:{env['DYN_SYSTEM_PORT']}/health")
            managed = stack.enter_context(
                ManagedProcess(
                    command=command,
                    working_dir=str(tmp_path),
                    display_name=f"{backend}-native-{index}",
                    env=env,
                    health_check_urls=health_urls,
                    timeout=600,
                    log_dir=str(tmp_path / f"engine-{index}"),
                    terminate_all_matching_process_names=False,
                )
            )
            role = (
                ("ENCODER_", "PREFILL_", "")[index]
                if has_encoder
                else ("" if index == 0 else "PREFILL_")
            )
            environment[f"SIDECAR_NATIVE_{role}GRPC"] = f"http://127.0.0.1:{grpc_port}"
            environment[
                f"SIDECAR_NATIVE_{role}METRICS"
            ] = f"http://127.0.0.1:{http_port}/metrics"
        environment["SIDECAR_NATIVE_TRANSFER_GATE"] = str(gate)
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
        if is_managed_sidecar:
            children = psutil.Process(managed.proc.pid).children(recursive=True)
            sidecars = [
                child
                for child in children
                if any(
                    connection.laddr.port == dynamo_dynamic_ports.system_ports[1]
                    for connection in child.net_connections(kind="tcp")
                    if connection.laddr
                )
            ]
            assert (
                len(sidecars) == 1
            ), "managed child must own the sidecar health endpoint"
            managed.proc.terminate()
            managed.proc.wait(timeout=60)
            _, alive = psutil.wait_procs(sidecars, timeout=10)
            assert not alive, "SGLang shutdown left its managed sidecar running"
        if (
            backend == "sglang"
            and scenario == "native_logprobs_and_structured_output_are_compatible"
        ):
            output = tokenizer.decode(
                json.loads(structured_tokens.read_text()), skip_special_tokens=True
            )
            assert json.loads(output) == {"ok": True}, output
