# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU HTTP parity/TTFT benchmark with a deterministic token mock worker.

Requires bindings rebuilt from the tested commit and SGLang frontend dependencies.
Uses file discovery, TCP and ZMQ: no etcd, NATS, model weights or GPU. Every
response echoes the digest of the token IDs received by the backend, allowing
exact downstream parity checks against full HF tokenization.
"""
import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from array import array
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import httpx
from benchmark_delta_tokenizer import summarize
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def digest(tokens):
    return hashlib.sha256(array("I", tokens)).hexdigest()


async def worker(args):
    # Optional native runtime is loaded only by this subprocess.
    from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
    from dynamo.runtime import DistributedRuntime

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    endpoint = runtime.endpoint(f"{os.environ['DYN_NAMESPACE']}.backend.generate")
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        args.model_dir,
        model_name="delta-test",
        worker_type=WorkerType.Aggregated,
    )

    async def generate(request, context):
        response = digest(request["token_ids"])
        token_ids = tokenizer.encode(response, add_special_tokens=False)
        if context.metadata.get("delta-test-delay") == "true":
            midpoint = len(token_ids) // 2
            yield {"token_ids": token_ids[:midpoint]}
            await asyncio.sleep(0.5)
            token_ids = token_ids[midpoint:]
        elif args.worker_delay:
            await asyncio.sleep(args.worker_delay)
        yield {
            "token_ids": token_ids,
            "finish_reason": "length",
        }

    completion_endpoint = runtime.endpoint(
        f"{os.environ['DYN_NAMESPACE']}.backend.completion"
    )
    await register_model(
        ModelInput.Tokens,
        ModelType.Completions,
        completion_endpoint,
        args.model_dir,
        model_name="delta-completion",
        worker_type=WorkerType.Aggregated,
    )
    await asyncio.gather(
        endpoint.serve_endpoint(generate), completion_endpoint.serve_endpoint(generate)
    )


@contextmanager
def child(command, env, log):
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def process_cpu_seconds(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def metrics(text):
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == "dynamo_frontend_delta_tokenizer_events_total":
                result[sample.labels["outcome"]] = sample.value
            elif sample.name in (
                "dynamo_frontend_delta_tokenizer_entries",
                "dynamo_frontend_delta_tokenizer_retained_bytes",
            ):
                result[sample.name.rsplit("tokenizer_", 1)[1]] = sample.value
    return result


async def arm(args, enabled, index, directory):
    port = free_port()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "DYN_FILE_KV": str(directory / "discovery"),
        "DYN_NAMESPACE": f"delta-http-{index}",
        "DYN_SYSTEM_PORT": "0",
        "DYN_SYSTEM_HOST": "127.0.0.1",
        "DYN_TCP_RPC_HOST": "127.0.0.1",
        "DYN_RUNTIME_NUM_WORKER_THREADS": "4",
        "DYN_RUNTIME_MAX_BLOCKING_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
    }
    template = directory / "template.jinja"
    template.write_text(_TEMPLATE)
    server = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(port),
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--namespace",
        env["DYN_NAMESPACE"],
        "--router-mode",
        "round-robin",
        "--dyn-chat-processor",
        "sglang",
        "--dyn-preprocess-workers",
        "0",
        "--tokenizer",
        "default",
        "--chat-template",
        str(template),
    ]
    if enabled:
        server += [
            "--experimental-delta-tokenizer",
            "--delta-tokenizer-verify-rate",
            "0",
        ]
    mock = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-dir",
        args.model_dir,
    ]
    if args.frontend_cpus:
        server = ["taskset", "-c", args.frontend_cpus, *server]
    if args.worker_cpus:
        mock = ["taskset", "-c", args.worker_cpus, *mock]
    reference = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    reference.chat_template = _TEMPLATE
    base = f"http://127.0.0.1:{port}"
    with child(server, env, directory / "frontend.log") as frontend:
        with child(mock, env, directory / "worker.log") as backend:
            async with httpx.AsyncClient(base_url=base, timeout=120) as client:
                deadline = time.monotonic() + 120
                while True:
                    if frontend.poll() is not None or backend.poll() is not None:
                        raise RuntimeError(f"service exited; inspect {directory}")
                    try:
                        response = await client.get("/v1/models")
                        ready = response.status_code == 200 and any(
                            model["id"] == "delta-test"
                            for model in response.json().get("data", [])
                        )
                    except httpx.TransportError:
                        ready = False
                    if ready:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"model discovery timeout; inspect {directory}"
                        )
                    await asyncio.sleep(0.2)

                def expected_digest(messages):
                    expected = reference.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                    if isinstance(expected, Mapping):
                        expected = expected["input_ids"]
                    return digest(expected)

                async def send(messages, headers, expected=None):
                    expected = expected or expected_digest(messages)
                    started = time.perf_counter()
                    first = None
                    parts = []
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": "delta-test",
                            "messages": messages,
                            "stream": True,
                            "max_tokens": 128,
                        },
                    ) as response:
                        if response.status_code != 200:
                            await response.aread()
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            for choice in event.get("choices", []):
                                text = choice.get("delta", {}).get("content") or ""
                                if text:
                                    first = first or time.perf_counter()
                                    parts.append(text)
                    if "".join(parts) != expected:
                        raise AssertionError(
                            "downstream token digest differs from full HF"
                        )
                    if first is None:
                        raise AssertionError("no SSE content received")
                    return first - started

                headers = {
                    "x-dynamo-meta-delta-tokenizer-session": "conversation",
                    "x-dynamo-meta-delta-tokenizer-scope": "trusted-test-scope",
                }
                messages = [
                    {
                        "role": "user",
                        "content": "Inspect this source code carefully.\n"
                        * args.prefix_repeats,
                    }
                ]
                # Two requests confirm an append; wait for certified seeding via metrics.
                await send(messages, headers)
                messages += [
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "Continue."},
                ]
                await send(messages, headers)
                if enabled:
                    deadline = time.monotonic() + 90
                    while True:
                        current = metrics((await client.get("/metrics")).text)
                        if current.get("verified", 0) >= 1:
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "seed verification not visible through /metrics"
                            )
                        await asyncio.sleep(0.2)
                # Prepare all parity oracles before timing the serving loop.
                # The client must not full-tokenize between measured requests.
                cases = []
                for turn in range(args.turns):
                    messages += [
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": f"Continue {turn}."},
                    ]
                    cases.append((messages.copy(), expected_digest(messages)))
                samples = []
                cpu_before = process_cpu_seconds(frontend.pid)
                started = time.perf_counter()
                for turn_messages, expected in cases:
                    samples.append(await send(turn_messages, headers, expected))
                wall = time.perf_counter() - started
                cpu_seconds = process_cpu_seconds(frontend.pid) - cpu_before
                if enabled:
                    # Missing scope and a different scope cannot reuse this session.
                    await send(messages, {})
                    await send(
                        messages,
                        {
                            **headers,
                            "x-dynamo-meta-delta-tokenizer-scope": "other-scope",
                        },
                    )
                if enabled:
                    before = metrics((await client.get("/metrics")).text)
                    prompt = "Simple completion request."
                    completion = await client.post(
                        "/v1/completions",
                        headers={
                            **headers,
                            "x-dynamo-meta-dynamo.frontend.api": "chat_completions",
                        },
                        json={
                            "model": "delta-completion",
                            "prompt": prompt,
                            "max_tokens": 128,
                            "stream": False,
                        },
                    )
                    completion.raise_for_status()
                    expected = digest(
                        reference.encode(prompt, add_special_tokens=False)
                    )
                    if completion.json()["choices"][0]["text"] != expected:
                        raise AssertionError("completion API reference parity failed")
                    after = metrics((await client.get("/metrics")).text)
                    if after.get("incremental", 0) != before.get("incremental", 0):
                        raise AssertionError(
                            "caller forged the server-owned API marker"
                        )
                # Disconnect after the first content chunk, then revisit the
                # previous prompt. Cancellation must not poison subsequent reuse.
                if enabled:
                    cancelled_messages = messages + [
                        {"role": "assistant", "content": "ok"},
                        {"role": "user", "content": "This request is cancelled."},
                    ]
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={**headers, "x-dynamo-meta-delta-test-delay": "true"},
                        json={
                            "model": "delta-test",
                            "messages": cancelled_messages,
                            "stream": True,
                            "max_tokens": 128,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: ") and "content" in line:
                                break
                    await send(messages, headers)
                counters = metrics((await client.get("/metrics")).text)
                if enabled and (
                    counters.get("incremental", 0) < args.turns
                    or counters.get("mismatches", 0)
                ):
                    raise AssertionError(
                        f"missing warm hits or token mismatch: {counters}"
                    )
                report = summarize(samples, cpu_seconds, len(samples), counters)
                report["wall_seconds"] = wall
                report["closed_loop_throughput_rps"] = len(samples) / wall
                report["note"] = (
                    "TTFT includes loopback and token-mock digest; "
                    "CPU is frontend-only, excluding setup."
                )
                return report


async def benchmark(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # A/B interleaving with a fresh process and discovery directory per arm.
    with tempfile.TemporaryDirectory(prefix="dynamo-delta-http-") as temporary:
        for index in range(2 * args.repeats):
            directory = Path(temporary) / str(index)
            directory.mkdir()
            try:
                result = await arm(args, bool(index % 2), index, directory)
            finally:
                # Retain logs with the report even if a service or assertion fails.
                for log in directory.glob("*.log"):
                    (output.parent / f"{output.stem}-{index}-{log.name}").write_text(
                        log.read_text()
                    )
            results.append({"enabled": bool(index % 2), **result})
            output.write_text(
                json.dumps(
                    {"kind": "token_mock_http_ttft", "results": results}, indent=2
                )
                + "\n"
            )
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="delta-http.json")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--prefix-repeats", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frontend-cpus")
    parser.add_argument("--worker-cpus")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-delay", type=float, default=0)
    args = parser.parse_args()
    if min(args.turns, args.prefix_repeats, args.repeats) <= 0:
        parser.error("turns, prefix repeats and repeats must be positive")
    asyncio.run(worker(args) if args.worker else benchmark(args))


if __name__ == "__main__":
    main()
