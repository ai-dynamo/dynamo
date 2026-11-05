#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run image E+PD or E/P/D against an unmodified official TRT-LLM nightly."""

import contextlib
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import run_image_test as image_test

BACKEND = "trtllm-nixl-encoder-v1"
FATAL_LOG_MESSAGES = (
    "Producer process has been terminated before all shared CUDA tensors released",
    "terminate called after throwing",
    "Observed timeout on context request",
    "Traceback (most recent call last)",
    "CUDA error",
)


def _child_pidfds(parent):
    # MPI workers may outlive the CLI's successful shutdown return briefly.
    processes = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            processes[int(path.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    owned = {parent}
    while children := {pid for pid, ppid in processes.items() if ppid in owned} - owned:
        owned.update(children)
    descriptors = []
    for pid in owned - {parent}:
        try:
            descriptors.append(os.pidfd_open(pid))
        except ProcessLookupError:
            continue
    return descriptors


def _wait_pidfds(descriptors, timeout):
    pending = descriptors[:]
    deadline = time.monotonic() + timeout
    while pending:
        ready, _, _ = select.select(
            pending, [], [], max(0, deadline - time.monotonic())
        )
        if not ready:
            raise subprocess.TimeoutExpired("engine descendants", timeout)
        pending = [fd for fd in pending if fd not in ready]


def _stop_engine(process, log):
    descriptors = _child_pidfds(process.pid)
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            returncode = process.wait(timeout=60)
            _wait_pidfds(descriptors, 30)
        except subprocess.TimeoutExpired as error:
            try:
                image_test.stop_engine(process)
            finally:
                # These stable handles still identify workers if their parent exited.
                for descriptor in descriptors:
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    except ProcessLookupError:
                        continue
                _wait_pidfds(descriptors, 10)
            raise RuntimeError(f"Engine required forced shutdown; see {log}") from error
        if returncode != 0:
            image_test.stop_engine(process)
            raise RuntimeError(f"Engine exited with {returncode}; see {log}")
        text = log.read_text()
        for message in FATAL_LOG_MESSAGES:
            if message in text:
                raise RuntimeError(f"Native engine failure: {message}; see {log}")
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def require_encoder_support(info, role):
    capability = getattr(info, "encoder", None)
    encoder = role == "encoder"
    if (
        capability is None
        or capability.schema_version != 1
        or capability.encoder_only != encoder
        or (not encoder and not capability.accepts_cache)
        or BACKEND not in capability.transfer_backends
    ):
        raise RuntimeError(
            f"Published nightly lacks the required {role} NIXL encoder capability; "
            "use an official nightly containing OpenEngine EPD support"
        )


def _check_capabilities(port, model, role, results):
    # Optional inference dependencies are only installed in the nightly image.
    import grpc
    from google.protobuf.json_format import MessageToDict
    from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            info = openengine_pb2_grpc.ControlStub(channel).GetModelInfo(
                model_pb2.GetModelInfoRequest(model=model), timeout=30
            )
    except grpc.RpcError as error:
        raise RuntimeError(
            f"Published nightly cannot discover {role} encoder capabilities"
        ) from error
    require_encoder_support(info, role)
    (results / f"{role}-model-info.json").write_text(
        json.dumps(MessageToDict(info, preserving_proto_field_name=True), indent=2)
        + "\n"
    )


def _start_engine(model_path, model, revision, role, split, results, cleanup):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    options = {
        "max_batch_size": 1,
        "max_num_tokens": 2048,
        "multimodal_config": {"encoder_cache_max_bytes": 0},
    }
    if role == "encoder":
        options["mm_encoder_only"] = True
    else:
        options["kv_cache_config"] = {
            "max_tokens": 2048,
            "free_gpu_memory_fraction": 0.10,
            "enable_block_reuse": False,
        }
        if role != "baseline":
            options["disable_mm_encoder"] = True
            if split:
                options["cache_transceiver_config"] = {"backend": "NIXL"}
    config = results / f"{role}.json"
    config.write_text(json.dumps(options, indent=2) + "\n")
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    for name in (
        "TRTLLM_OPENENGINE_ENCODER_TRANSFER",
        "TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION",
        "TRTLLM_OPENENGINE_ENCODER_ENDPOINT",
    ):
        env.pop(name, None)
    if role != "baseline":
        env.update(
            TRTLLM_OPENENGINE_ENCODER_TRANSFER="nixl",
            TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION=revision,
        )
        if role == "encoder":
            env["TRTLLM_OPENENGINE_ENCODER_ENDPOINT"] = f"127.0.0.1:{port}"
    log = results / f"{role}-engine.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm.commands.serve",
                str(model_path),
                "--grpc",
                "--grpc-protocol",
                "openengine",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--max_seq_len",
                "2048",
                "--extra_llm_api_options",
                str(config),
            ],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    cleanup.callback(_stop_engine, process, log)
    image_test.wait_for_engine(process, port)
    if role != "baseline":
        _check_capabilities(port, model, role, results)
    return f"http://127.0.0.1:{port}"


def _validate_cases(path, tokenizer):
    response = json.loads(path.read_text())
    cases = response.get("cases", [])
    if len(cases) != 2:
        raise RuntimeError(f"Expected both opposite-color image responses in {path}")
    for case, colors in zip(cases, (("red", "blue"), ("blue", "red")), strict=True):
        caption = tokenizer.decode(case["token_ids"], skip_special_tokens=True)
        image_test.validate_response(
            "image",
            caption,
            case,
            len(image_test.FIXTURES["image"]["token_ids"]),
            colors,
        )
        if case.get("cached_tokens") != 0:
            raise AssertionError(f"EPD parity requires fresh image/KV caches: {case}")
        case["text"] = caption
    path.write_text(json.dumps(response, indent=2) + "\n")


def run(args, results):
    # Keep CPU harness tests independent of optional model/backend packages.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config = image_test.load_runtime_config(args, results)
    test = image_test.EPD_TESTS[args.scenario]
    binary = args.binary.resolve()
    binary.chmod(binary.stat().st_mode | 0o111)
    for selected in (test, "tests::e2e::e2e_real_openengine_image"):
        listing = subprocess.check_output(
            [str(binary), "--list", "--exact", selected, "--format", "terse"], text=True
        )
        if listing.splitlines().count(f"{selected}: test") != 1:
            raise RuntimeError(
                f"Expected exactly one {selected} integration test, got: {listing}"
            )
    model_path = args.model_path or snapshot_download(
        config["model"], revision=config["model_revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": image_test.FIXTURES["image"]["prompt"]},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = (
        ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "input_ids") else ids
    )
    if ids != image_test.FIXTURES["image"]["token_ids"]:
        raise RuntimeError(f"Image fixture tokenizer mismatch: {ids}")
    images = results / "images.json"
    images.write_text(json.dumps(image_test.routing_fixture()) + "\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TRTLLM_E2E_")
    }
    env.update(TRTLLM_E2E_MODEL=config["model"], TRTLLM_E2E_EPD_IMAGES=str(images))
    baseline = results / "baseline"
    baseline.mkdir(exist_ok=True)
    baseline_result = baseline / "response.json"
    baseline_result.unlink(missing_ok=True)
    (results / "response.json").unlink(missing_ok=True)
    with contextlib.ExitStack() as cleanup:
        endpoint = _start_engine(
            model_path,
            config["model"],
            config["model_revision"],
            "baseline",
            False,
            baseline,
            cleanup,
        )
        image_test.run_rust_test(
            binary,
            dict(
                env,
                TRTLLM_E2E_ENDPOINT=endpoint,
                TRTLLM_E2E_RESULT=str(baseline_result),
            ),
            "tests::e2e::e2e_real_openengine_image",
            baseline,
        )
    _validate_cases(baseline_result, tokenizer)
    split = args.scenario == "image-epd-disaggregated"
    with contextlib.ExitStack() as cleanup:
        endpoints = {
            role: _start_engine(
                model_path,
                config["model"],
                config["model_revision"],
                role,
                split,
                results,
                cleanup,
            )
            for role in ("encoder", "prefill", "decode")[: 3 if split else 2]
        }
        env.update(
            TRTLLM_E2E_ENCODER_ENDPOINT=endpoints["encoder"],
            TRTLLM_E2E_PREFILL_ENDPOINT=endpoints["prefill"],
            TRTLLM_E2E_EPD_BASELINE=str(baseline_result),
            TRTLLM_E2E_RESULT=str(results / "response.json"),
        )
        env.pop("TRTLLM_E2E_DECODE_ENDPOINT", None)
        if split:
            env["TRTLLM_E2E_DECODE_ENDPOINT"] = endpoints["decode"]
        image_test.run_rust_test(binary, env, test, results, timeout=360)
    _validate_cases(results / "response.json", tokenizer)
