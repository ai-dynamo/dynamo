#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Rust adapter against an unmodified upstream TRT-LLM image."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

FIXTURES = {
    "image": {
        "prompt": "Name the colors from left to right.",
        "token_ids": [
            151644,
            872,
            198,
            151652,
            151655,
            151653,
            675,
            279,
            7987,
            504,
            2115,
            311,
            1290,
            13,
            151645,
            198,
            151644,
            77091,
            198,
        ],
        "context_length": 2048,
    },
    "video": {
        "prompt": "Describe the shape and its motion in this video.",
        "token_ids": [
            151644,
            872,
            198,
            151652,
            151656,
            151653,
            74785,
            279,
            6083,
            323,
            1181,
            11379,
            304,
            419,
            2766,
            13,
            151645,
            198,
            151644,
            77091,
            198,
        ],
        "context_length": 2048,
    },
    "audio": {
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "token_ids": [],
        "context_length": 1500,
    },
}
VIDEO_PATH = (
    Path(__file__).resolve().parents[3] / "llm/tests/data/media/triangle_240p_10.mp4"
)
VIDEO_SHA256 = "04171a85ee3ba5ac9118b0a5858adf0892b49474cdf15320e32599a3d1ab9919"

AUDIO_URL = (
    "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/"
    "8e35ed77aeece087fa53abf2a68a282dc13efc8c/"
    "datasets/mini_en/wav/1221-135766-0002.wav"
)
AUDIO_SHA256 = "65b222837919ccbb924a4e1077413ea7cc6af3e68b663b012a9539d5c05850f0"

EPD_TESTS = {
    "image-epd": "tests::e2e::epd::e2e_real_openengine_image_epd",
    "image-epd-disaggregated": "tests::e2e::epd::e2e_real_openengine_image_epd_disaggregated",
}


def prepare_audio(results):
    with urllib.request.urlopen(AUDIO_URL, timeout=30) as response:
        data = response.read(1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != AUDIO_SHA256:
        raise ValueError("Audio fixture does not match the validated speech sample")
    path = results / "audio-data-uri.txt"
    path.write_text("data:audio/wav;base64," + base64.b64encode(data).decode("ascii"))
    return path


def model_config(config, media):
    if media == "audio":
        return {**config, **config["audio"]}
    return config


def load_runtime_config(args, results):
    config = model_config(json.loads(args.config.read_text()), args.media)
    identity = {
        **config,
        "scenario": args.scenario,
        "media": args.media,
        "installed_version": importlib.metadata.version("tensorrt_llm"),
        "installed_commit": os.environ.get("TRT_LLM_GIT_COMMIT"),
    }
    (results / "runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    if identity["installed_commit"] != config["upstream_commit"]:
        raise RuntimeError(
            f"Unexpected upstream commit: {identity['installed_commit']}"
        )
    return config


def video_data_uri(path=VIDEO_PATH):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != VIDEO_SHA256:
        raise ValueError(
            "Video fixture changed; revalidate its content and expected response"
        )
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def validate_response(
    media, caption, response, input_token_count, expected_colors=("red", "blue")
):
    words = re.findall(r"[a-z]+", caption.lower())
    if media == "image":
        colors = [word for word in words if word in ("red", "blue")]
        if colors[:2] != list(expected_colors):
            raise AssertionError(f"Expected {expected_colors}, got: {caption}")
    elif media == "audio":
        if "thoughts affected hester" not in " ".join(words):
            raise AssertionError(f"Unexpected speech transcription: {caption}")
    elif not {"triangle", "triangular"}.intersection(words):
        raise AssertionError(
            f"Expected a triangle in the video description, got: {caption}"
        )
    if (
        response["completion_tokens"] <= 0
        or response["prompt_tokens"] <= input_token_count
    ):
        raise AssertionError(f"Missing authoritative {media} token usage: {response}")


def wait_for_engine(process, port, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Engine exited with {process.returncode}; see engine.log"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("Engine did not become ready within 600 seconds")


def stop_engine(process):
    # The serving CLI may spawn MPI workers; reap the entire process group.
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_rust_test(binary, env, test, results, timeout=240):
    completed = subprocess.run(
        [str(binary), "--exact", test, "--ignored", "--nocapture", "--color", "never"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    log = results / (
        "rust-events-test.log" if test.endswith("event_conversion") else "rust-test.log"
    )
    log.write_text(completed.stdout)
    print(completed.stdout, flush=True)
    completed.check_returncode()
    if "1 passed; 0 failed; 0 ignored;" not in completed.stdout:
        raise RuntimeError(f"{test} did not execute exactly once")


def routing_fixture():
    # Keep CPU harness tests importable without model-processing dependencies.
    from PIL import Image, ImageOps, PngImagePlugin

    encoded = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAK0lEQVR4nGP8z0AsYGQgVi0T0WaSAEYNHTV01NBRQ0cNHTV01NBRQwefoQBvcAI3xbX7+QAAAABJRU5ErkJggg=="
    image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    alias = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("cache-alias", "same pixels, different URI")
    image.save(alias, format="PNG", pnginfo=metadata)
    changed = BytesIO()
    ImageOps.mirror(image).save(changed, format="PNG")
    return tuple(
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        for buffer in (alias, changed)
    )


def run(args, results):
    if args.scenario in EPD_TESTS:
        from run_epd_test import run as run_epd

        return run_epd(args, results)

    # CPU harness tests import this module without the inference dependencies.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    routing = args.scenario == "image-routing"
    fixture = FIXTURES[args.media]
    test = f"tests::e2e::e2e_real_openengine_{args.media}"
    config = load_runtime_config(args, results)

    video = video_data_uri() if args.media == "video" else None
    audio_path = prepare_audio(results) if args.media == "audio" else None

    binary = args.binary.resolve()
    binary.chmod(binary.stat().st_mode | 0o111)
    listing = subprocess.check_output(
        [str(binary), "--list", "--exact", test, "--format", "terse"], text=True
    )
    if listing.splitlines().count(f"{test}: test") != 1:
        raise RuntimeError(
            f"Expected exactly one {args.media} integration test, got: {listing}"
        )

    model_path = args.model_path or snapshot_download(
        config["model"], revision=config["model_revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    expected_ids = fixture["token_ids"]
    if args.media != "audio":
        # The fixed Rust fixture uses Qwen3-VL's rendering of this exact message.
        # Check it against the pinned tokenizer so a template change fails loudly.
        token_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": args.media},
                        {"type": "text", "text": fixture["prompt"]},
                    ],
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = (
            token_ids["input_ids"]
            if isinstance(token_ids, dict) or hasattr(token_ids, "input_ids")
            else token_ids
        )
        if token_ids != expected_ids:
            raise RuntimeError(f"{args.media} fixture tokenizer mismatch: {token_ids}")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    options = results / "engine.yaml"
    context_length = fixture["context_length"]
    if args.media == "audio":
        options.write_text(Path(__file__).with_name("whisper.yaml").read_text())
    else:
        options.write_text(
            f"max_batch_size: 1\nmax_num_tokens: {context_length}\n"
            f"kv_cache_config:\n  max_tokens: {context_length}\n"
            "  free_gpu_memory_fraction: 0.15\n"
        )
    result_path = results / "response.json"
    engine = None
    context = subscriber = None
    env_extra = {}
    try:
        if routing:
            import zmq

            alias, changed = routing_fixture()
            env_extra = {
                "TRTLLM_E2E_IMAGE_ROUTING": "1",
                "TRTLLM_E2E_ALIAS_IMAGE": alias,
                "TRTLLM_E2E_MODEL_PATH": str(model_path),
                "TRTLLM_E2E_CHANGED_IMAGE": changed,
            }
            context = zmq.Context()
            subscriber = context.socket(zmq.SUB)
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            event_port = subscriber.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{event_port}"
            subscriber.unbind(endpoint)
            subscriber.connect(endpoint)
            with options.open("a") as stream:
                stream.write(
                    "  enable_block_reuse: true\n  use_kv_cache_manager_v2: true\n"
                    "  kv_events_config:\n    enable_kv_cache_events: true\n"
                    f"    publisher: zmq\n    endpoint: {endpoint}\n"
                )
        with (results / "engine.log").open("w") as log:
            engine = subprocess.Popen(
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
                    str(context_length),
                    "--extra_llm_api_options",
                    str(options),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_engine(engine, port)
            env = dict(
                os.environ,
                TRTLLM_E2E_ENDPOINT=f"http://127.0.0.1:{port}",
                TRTLLM_E2E_MODEL=config["model"],
                TRTLLM_E2E_RESULT=str(result_path),
            )
            env.update(env_extra)
            if video is not None:
                env["TRTLLM_E2E_VIDEO_DATA_URI"] = video
            if audio_path is not None:
                env["TRTLLM_E2E_AUDIO_DATA_URI_FILE"] = str(audio_path)
                env["TRTLLM_E2E_AUDIO_PROMPT"] = fixture["prompt"]
            run_rust_test(binary, env, test, results)
            if routing:
                payloads = []
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if subscriber.poll(500):
                        payloads.append(list(subscriber.recv_multipart()[-1]))
                if not payloads:
                    raise AssertionError("No native KV events were published")
                events_path = results / "routing-events.json"
                events_path.write_text(json.dumps(payloads))
                env["TRTLLM_E2E_ROUTING_EVENTS"] = str(events_path)
                run_rust_test(
                    binary,
                    env,
                    "tests::e2e::e2e_real_image_routing_event_conversion",
                    results,
                )
    finally:
        stop_engine(engine)
        if subscriber is not None:
            subscriber.close(linger=0)
        if context is not None:
            context.term()

    response = json.loads(result_path.read_text())
    cases = response["cases"] if routing else [response]
    for index, case in enumerate(cases):
        caption = tokenizer.decode(case["token_ids"], skip_special_tokens=True)
        case["text"] = caption
        colors = ("blue", "red") if routing and index == 3 else ("red", "blue")
        validate_response(args.media, caption, case, len(expected_ids), colors)
        print(f"{args.media} response {index}: {caption}", flush=True)
    result_path.write_text(json.dumps(response, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=["image", "video", "audio", "image-routing", *EPD_TESTS]
    )
    parser.add_argument("--media", choices=FIXTURES)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="Resolved nightly image manifest"
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument(
        "--model-path", type=Path, help="Use an already downloaded model snapshot"
    )
    args = parser.parse_args()
    args.scenario = args.scenario or args.media or "image"
    media = (
        "image"
        if args.scenario == "image-routing" or args.scenario in EPD_TESTS
        else args.scenario
    )
    if args.media is not None and args.media != media:
        parser.error("--media conflicts with --scenario")
    args.media = media
    results = (
        args.results or Path(f"test-results/trtllm-nightly-{args.scenario}")
    ).resolve()
    results.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    failure = None
    try:
        run(args, results)
    except Exception:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr)
    duration = str(time.monotonic() - start)
    suite = ET.Element(
        "testsuite",
        name=f"trtllm-nightly-{args.scenario}",
        tests="1",
        failures=str(int(failure is not None)),
        skipped="0",
        time=duration,
    )
    case = ET.SubElement(
        suite,
        "testcase",
        name=EPD_TESTS.get(args.scenario)
        or (
            f"tests::e2e::e2e_real_openengine_{args.media}"
            + ("[image-routing]" if args.scenario == "image-routing" else "")
        ),
        classname="trtllm.sidecar",
        time=duration,
    )
    if failure:
        ET.SubElement(case, "failure", message=failure.splitlines()[-1]).text = failure
    rust_log = results / "rust-test.log"
    if rust_log.exists():
        ET.SubElement(case, "system-out").text = rust_log.read_text()
    ET.ElementTree(suite).write(
        results / "junit.xml", encoding="utf-8", xml_declaration=True
    )
    return int(failure is not None)


if __name__ == "__main__":
    sys.exit(main())
