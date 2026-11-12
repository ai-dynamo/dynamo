# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the packaged vllm-rs/Python LoRA protocol without GPU or model weights."""

import importlib.metadata
import importlib.util
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import pytest
import requests
import zmq
from tokenizers import Tokenizer, models

from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]


def _local_model(root: Path) -> Path:
    """Only config/tokenizer files: the native frontend never loads weights."""
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "vocab_size": 4,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "eos_token_id": 1,
            }
        )
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
        )
    )
    tokenizer.save(str(model / "tokenizer.json"))
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "model_max_length": 128,
            }
        )
    )
    (model / "generation_config.json").write_text("{}")
    return model


def _installed_lora_schema():
    # Execute the installed definition, avoiding unrelated GPU initialization
    # from vllm's package imports. Never duplicate LoRARequest in this test.
    dist = importlib.metadata.distribution("vllm")
    spec = importlib.util.spec_from_file_location(
        "installed_lora_request", dist.locate_file("vllm/lora/request.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoRARequest, dist.version


def _consume_utilities(handshake_port, schema, version, stop, decoded):
    """Replace engine compute only AFTER decoding the real native wire payload."""

    def receive(sock, multipart=False):
        deadline = time.monotonic() + 40
        while not stop.is_set():
            if sock.poll(100):
                return sock.recv_multipart() if multipart else sock.recv()
            if time.monotonic() >= deadline:
                raise TimeoutError("No native EngineCore message within 40s")
        raise InterruptedError("Consumer stopped")

    with zmq.Context() as ctx:
        with (
            ctx.socket(zmq.DEALER) as handshake,
            ctx.socket(zmq.DEALER) as incoming,
            ctx.socket(zmq.PUSH) as outgoing,
        ):
            for sock in (handshake, incoming, outgoing):
                sock.setsockopt(zmq.LINGER, 0)
            for sock in (handshake, incoming):
                sock.setsockopt(zmq.IDENTITY, b"\x00\x00")
            handshake.connect(f"tcp://127.0.0.1:{handshake_port}")
            handshake.send(
                msgspec.msgpack.encode(
                    {"status": "HELLO", "local": False, "headless": True}
                )
            )
            try:
                addresses = msgspec.msgpack.decode(receive(handshake))["addresses"]
                assert len(addresses["inputs"]) == len(addresses["outputs"]) == 1
                assert addresses["coordinator_input"] is None
                incoming.connect(addresses["inputs"][0])
                outgoing.connect(addresses["outputs"][0])
                incoming.send(
                    msgspec.msgpack.encode(
                        {
                            "max_model_len": 128,
                            "num_gpu_blocks": 8,
                            "block_size": 16,
                            "dp_stats_address": None,
                            "dtype": "float32",
                            "vllm_version": version,
                            "world_size": 1,
                            "data_parallel_size": 1,
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "decode_context_parallel_size": 1,
                            "data_parallel_rank": 0,
                            "max_num_seqs": 4,
                            "max_num_batched_tokens": 128,
                            "instance_id": "lora-wire-test",
                            "supports_lora": True,
                            "max_loras": 4,
                            "kv_cache_size_tokens": 128,
                            "kv_cache_max_concurrency": 1.0,
                        }
                    )
                )
                handshake.send(
                    msgspec.msgpack.encode(
                        {"status": "READY", "local": False, "headless": True}
                    )
                )
                loaded = set()
                while not stop.is_set():
                    kind, payload = receive(incoming, multipart=True)
                    assert kind == b"\x03", (
                        f"Unexpected EngineCore request kind: {kind!r}"
                    )
                    _, call_id, method, args = msgspec.msgpack.decode(
                        payload, type=tuple[int, int, str, list[msgspec.Raw]]
                    )
                    error, result = None, None
                    try:
                        if method == "add_lora":
                            # Raw preserves the bytes serialized by the installed Rust binary.
                            request = msgspec.msgpack.decode(args[0], type=schema)
                            decoded.append(request)
                            loaded.add(request.lora_int_id)
                            result = True
                        elif method == "remove_lora":
                            loaded.remove(msgspec.msgpack.decode(args[0], type=int))
                            result = True
                        elif method == "list_loras":
                            result = sorted(loaded)
                        elif method == "get_supported_tasks":
                            result = ["generate"]
                        else:
                            raise AssertionError(f"Unexpected utility: {method}")
                    except msgspec.ValidationError as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    utility = [call_id, error, None if error else [None, result]]
                    outgoing.send(
                        msgspec.msgpack.encode([0, [], None, time.time(), utility])
                    )
            except InterruptedError:
                if not stop.is_set():
                    raise


def test_native_lora_matches_installed_python_schema(tmp_path):
    """Runtime load must serialize a Python-compatible LoRARequest.

    Start the packaged native frontend against a CPU EngineCore protocol peer,
    decode its actual add_lora argument with the installed Python schema, and
    verify adapter identity and load/list/unload. No inference is simulated or
    asserted here; weight loading is replaced only after protocol validation.
    """
    schema, version = _installed_lora_schema()
    binary = shutil.which("vllm-rs")
    assert binary is not None, "The vLLM runtime must include its native vllm-rs binary"
    model = _local_model(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    # Negative control for the extra nullable tensorizer slot from vLLM 0.32.
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(
                ["wire-adapter", 1, str(adapter), None, None, False, False]
            ),
            type=schema,
        )

    http_port, handshake_port = allocate_ports(2, DynamoPortRange.SERVE)
    stop, decoded = threading.Event(), []
    base = f"http://127.0.0.1:{http_port}"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            consumer = pool.submit(
                _consume_utilities, handshake_port, schema, version, stop, decoded
            )
            try:
                with ManagedProcess(
                    command=[
                        binary,
                        "serve",
                        str(model),
                        "--data-parallel-size-local",
                        "0",
                        "--data-parallel-size",
                        "1",
                        "--handshake-port",
                        str(handshake_port),
                        "--port",
                        str(http_port),
                        "--tool-call-parser",
                        "none",
                        "--reasoning-parser",
                        "none",
                    ],
                    env={
                        **os.environ,
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                        "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES": str(tmp_path),
                    },
                    health_check_urls=[f"{base}/health"],
                    timeout=30,
                    terminate_all_matching_process_names=False,
                    log_dir=str(tmp_path),
                    display_name="vllm-rs-lora-protocol",
                ):
                    response = requests.post(
                        f"{base}/v1/load_lora_adapter",
                        json={
                            "lora_name": "wire-adapter",
                            "lora_path": str(adapter),
                        },
                        timeout=10,
                    )
                    assert response.status_code == 200, response.text
                    assert len(decoded) == 1
                    request = decoded[0]
                    assert request.lora_name == "wire-adapter"
                    assert request.lora_int_id > 0
                    assert request.lora_path == str(adapter)
                    assert request.base_model_name is None
                    assert request.load_inplace is False
                    assert request.is_3d_lora_weight is False
                    response = requests.get(f"{base}/v1/models", timeout=10)
                    response.raise_for_status()
                    assert "wire-adapter" in {
                        item["id"] for item in response.json()["data"]
                    }
                    response = requests.post(
                        f"{base}/v1/unload_lora_adapter",
                        json={
                            "lora_name": "wire-adapter",
                        },
                        timeout=10,
                    )
                    assert response.status_code == 200, response.text
                    response = requests.get(f"{base}/v1/models", timeout=10)
                    response.raise_for_status()
                    assert "wire-adapter" not in {
                        item["id"] for item in response.json()["data"]
                    }
            finally:
                stop.set()
                consumer.result(timeout=5)
    finally:
        deallocate_ports([http_port, handshake_port])
