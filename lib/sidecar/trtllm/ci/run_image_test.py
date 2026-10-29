#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Rust adapter against an unmodified upstream TRT-LLM image."""

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

TEST = "tests::e2e::e2e_real_openengine_image"


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


def run(args, results):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    identity = {
        **config,
        "installed_version": importlib.metadata.version("tensorrt_llm"),
        "installed_commit": os.environ.get("TRT_LLM_GIT_COMMIT"),
    }
    (results / "runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    if identity["installed_commit"] != config["upstream_commit"]:
        raise RuntimeError(
            f"Unexpected upstream commit: {identity['installed_commit']}"
        )

    binary = args.binary.resolve()
    binary.chmod(binary.stat().st_mode | 0o111)
    listing = subprocess.check_output(
        [str(binary), "--list", "--exact", TEST, "--format", "terse"], text=True
    )
    if listing.splitlines().count(f"{TEST}: test") != 1:
        raise RuntimeError(
            f"Expected exactly one image integration test, got: {listing}"
        )

    model_path = args.model_path or snapshot_download(
        config["model"], revision=config["model_revision"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # The fixed Rust fixture uses Qwen3-VL's rendering of this exact message.
    # Check it against the pinned tokenizer so a template change fails loudly.
    token_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Name the colors from left to right."},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    expected_ids = [
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
    ]
    token_ids = (
        token_ids["input_ids"]
        if isinstance(token_ids, dict) or hasattr(token_ids, "input_ids")
        else token_ids
    )
    if token_ids != expected_ids:
        raise RuntimeError(f"Image fixture tokenizer mismatch: {token_ids}")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    options = results / "engine.yaml"
    options.write_text(
        "max_batch_size: 1\nmax_num_tokens: 2048\n"
        "kv_cache_config:\n  max_tokens: 2048\n"
        "  free_gpu_memory_fraction: 0.15\n"
    )
    result_path = results / "response.json"
    engine = None
    try:
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
                    "2048",
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
            completed = subprocess.run(
                [
                    str(binary),
                    "--exact",
                    TEST,
                    "--ignored",
                    "--nocapture",
                    "--color",
                    "never",
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=240,
            )
            (results / "rust-test.log").write_text(completed.stdout)
            print(completed.stdout, flush=True)
            completed.check_returncode()
            if "1 passed; 0 failed; 0 ignored;" not in completed.stdout:
                raise RuntimeError(
                    "Image integration test did not execute exactly once"
                )
    finally:
        stop_engine(engine)

    response = json.loads(result_path.read_text())
    caption = tokenizer.decode(response["token_ids"], skip_special_tokens=True)
    response["text"] = caption
    result_path.write_text(json.dumps(response, indent=2) + "\n")
    print(f"Image response: {caption}", flush=True)
    words = caption.lower().replace("-", " ").split()
    # Accept punctuation and phrasing differences, but require both fixture colors.
    colors = {word.strip(".,:;!?()\"'") for word in words}
    if not {"red", "blue"} <= colors:
        raise AssertionError(
            f"Expected red and blue in the image description, got: {caption}"
        )
    if response["completion_tokens"] <= 0 or response["prompt_tokens"] <= len(
        expected_ids
    ):
        raise AssertionError(f"Missing authoritative image token usage: {response}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="Resolved nightly image manifest"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("test-results/trtllm-nightly-image")
    )
    parser.add_argument(
        "--model-path", type=Path, help="Use an already downloaded model snapshot"
    )
    args = parser.parse_args()
    results = args.results.resolve()
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
        name="trtllm-nightly-image",
        tests="1",
        failures=str(int(failure is not None)),
        skipped="0",
        time=duration,
    )
    case = ET.SubElement(
        suite, "testcase", name=TEST, classname="trtllm.sidecar", time=duration
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
