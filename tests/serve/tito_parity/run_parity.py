# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import copy
import difflib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATE_PATH = "/inference/v1/generate"
UPSTREAM_PORT = 8000
DYNAMO_PORT = 8000
MODEL_MAX_LEN = 4096
MAX_NUM_SEQS = 4
KV_CACHE_BYTES = 4_294_967_296


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    prompt: str
    colors: tuple[str, ...] = ()
    ignore_eos: bool = True
    sampling_overrides: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RequestRun:
    key: str
    request: dict[str, Any]


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare upstream vLLM and Dynamo token-in/token-out responses"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--suite", choices=("smoke", "expanded", "full"), default="smoke"
    )
    parser.add_argument(
        "--dynamo-topology",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="Dynamo worker topology to compare against upstream vLLM.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        choices=(1, 4),
        default=1,
        help=(
            "Run exact parity sequentially by default. Set to 4 to add a "
            "concurrent batching stress stage."
        ),
    )
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def make_image_data_uri(colors: tuple[str, ...], size: int = 224) -> str:
    image = Image.new("RGB", (size, size), colors[0])
    draw = ImageDraw.Draw(image)
    width = max(1, size // len(colors))
    for index, color in enumerate(colors):
        left = index * width
        right = size if index == len(colors) - 1 else (index + 1) * width
        draw.rectangle((left, 0, right, size), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def smoke_cases() -> list[Case]:
    return [
        Case(
            name="text-smoke",
            kind="text",
            prompt="List the first four prime numbers, separated by commas.",
        ),
        Case(
            name="vlm-smoke",
            kind="vlm",
            prompt="Name the two colors in this image.",
            colors=("red", "blue"),
        ),
    ]


def expanded_cases() -> list[Case]:
    return [
        Case(
            name="text-forced-length",
            kind="text",
            prompt="Write a compact description of deterministic decoding.",
        ),
        Case(
            name="text-eos",
            kind="text",
            prompt="Reply with exactly the word green.",
            ignore_eos=False,
        ),
        Case(
            name="text-logprobs",
            kind="text",
            prompt="Continue this sequence: 2, 4, 8,",
            sampling_overrides=(("logprobs", 5),),
        ),
        Case(
            name="text-prompt-logprobs",
            kind="text",
            prompt="The quick brown fox",
            sampling_overrides=(("prompt_logprobs", 1),),
        ),
        Case(
            name="vlm-red-blue",
            kind="vlm",
            prompt="Report the colors from left to right.",
            colors=("red", "blue"),
        ),
        Case(
            name="vlm-green-yellow",
            kind="vlm",
            prompt="Report the colors from left to right.",
            colors=("green", "yellow"),
        ),
        Case(
            name="vlm-two-identical",
            kind="vlm",
            prompt="Compare these two images in one sentence.",
            colors=("purple", "purple"),
        ),
        Case(
            name="vlm-two-distinct",
            kind="vlm",
            prompt="Compare these two images in one sentence.",
            colors=("orange", "cyan"),
        ),
    ]


def cases_for_suite(suite: str) -> list[Case]:
    if suite == "smoke":
        return smoke_cases()
    if suite == "expanded":
        return expanded_cases()
    return smoke_cases() + expanded_cases()


def render_payload(
    case: Case, model: str, max_tokens: int
) -> tuple[str, dict[str, Any]]:
    common = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": False,
    }
    if case.kind == "text":
        return "/v1/completions/render", {"prompt": case.prompt, **common}

    image_parts = []
    if case.name in ("vlm-two-identical", "vlm-two-distinct"):
        for color in case.colors:
            uri = make_image_data_uri((color,))
            image_parts.append({"type": "image_url", "image_url": {"url": uri}})
    else:
        uri = make_image_data_uri(case.colors)
        image_parts.append({"type": "image_url", "image_url": {"url": uri}})
    content = [{"type": "text", "text": case.prompt}, *image_parts]
    return "/v1/chat/completions/render", {
        "messages": [{"role": "user", "content": content}],
        **common,
    }


def post_json(
    port: int, path: str, payload: dict[str, Any], timeout: int
) -> HttpResult:
    response = requests.post(
        f"http://127.0.0.1:{port}{path}", json=payload, timeout=timeout
    )
    try:
        body = response.json()
    except requests.JSONDecodeError as error:
        raise RuntimeError(
            f"{path} returned non-JSON status={response.status_code}: {response.text[:1000]}"
        ) from error
    if not isinstance(body, dict):
        raise RuntimeError(f"{path} returned non-object JSON: {type(body).__name__}")
    return HttpResult(status=response.status_code, body=body)


def render_cases(
    cases: list[Case], model: str, suite: str, timeout: int
) -> dict[str, dict[str, Any]]:
    rendered: dict[str, dict[str, Any]] = {}
    for case in cases:
        max_tokens = 8 if case.name.endswith("smoke") else 64
        path, payload = render_payload(case, model, max_tokens)
        response = requests.post(
            f"http://127.0.0.1:{UPSTREAM_PORT}{path}",
            json=payload,
            timeout=timeout,
        )
        body = response.json()
        if response.status_code != 200:
            raise RuntimeError(
                f"render failed for {case.name}: status={response.status_code} body={body}"
            )
        if isinstance(body, list):
            if len(body) != 1:
                raise RuntimeError(
                    f"render returned {len(body)} requests for {case.name}"
                )
            body = body[0]
        if not isinstance(body, dict):
            raise RuntimeError(f"render returned non-object for {case.name}")

        request = copy.deepcopy(body)
        request["request_id"] = f"parity-{suite}-{case.name}"
        request["model"] = model
        request["stream"] = False
        sampling = request.setdefault("sampling_params", {})
        sampling.update(
            {
                "temperature": 0,
                "top_p": 1,
                "n": 1,
                "seed": 0,
                "max_tokens": max_tokens,
                "ignore_eos": case.ignore_eos,
            }
        )
        sampling.update(dict(case.sampling_overrides))
        rendered[case.name] = request
    return rendered


def build_runs(
    cases: list[Case],
    rendered: dict[str, dict[str, Any]],
    suite: str,
    max_concurrency: int,
) -> list[tuple[int, list[RequestRun]]]:
    groups: list[tuple[int, list[RequestRun]]] = []
    smoke_names = {case.name for case in smoke_cases()}
    smoke = [case for case in cases if case.name in smoke_names]
    expanded = [case for case in cases if case.name not in smoke_names]
    if smoke:
        groups.append(
            (
                1,
                [
                    RequestRun(
                        key=f"smoke-r0-{case.name}",
                        request=copy.deepcopy(rendered[case.name]),
                    )
                    for case in smoke
                ],
            )
        )
    if expanded:
        concurrencies = (1, 4) if max_concurrency == 4 else (1,)
        for concurrency in concurrencies:
            for repetition in range(2):
                requests_for_run = []
                for case in expanded:
                    request = copy.deepcopy(rendered[case.name])
                    request[
                        "request_id"
                    ] = f"parity-{suite}-c{concurrency}-r{repetition}-{case.name}"
                    requests_for_run.append(
                        RequestRun(
                            key=f"c{concurrency}-r{repetition}-{case.name}",
                            request=request,
                        )
                    )
                groups.append((concurrency, requests_for_run))
    return groups


def execute_group(
    port: int, concurrency: int, runs: list[RequestRun], timeout: int
) -> dict[str, HttpResult]:
    def send(run: RequestRun) -> tuple[str, HttpResult]:
        return run.key, post_json(port, GENERATE_PATH, run.request, timeout)

    if concurrency == 1:
        return dict(send(run) for run in runs)
    results: dict[str, HttpResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, run) for run in runs]
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
    return results


def execute_runs(
    port: int,
    groups: list[tuple[int, list[RequestRun]]],
    timeout: int,
) -> tuple[dict[str, HttpResult], dict[str, dict[str, Any]]]:
    results: dict[str, HttpResult] = {}
    requests_by_key: dict[str, dict[str, Any]] = {}
    for concurrency, runs in groups:
        requests_by_key.update({run.key: run.request for run in runs})
        results.update(execute_group(port, concurrency, runs, timeout))
    return results, requests_by_key


def normalize_response(body: dict[str, Any], key: str, side: str) -> dict[str, Any]:
    normalized = copy.deepcopy(body)
    request_id = normalized.pop("request_id", None)
    if not isinstance(request_id, str) or not request_id:
        raise AssertionError(f"{side} {key} has invalid request_id={request_id!r}")
    return normalized


def compare_results(
    upstream: dict[str, HttpResult],
    dynamo: dict[str, HttpResult],
    requests_by_key: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    failures = []
    artifact_path = output_dir / "responses.jsonl"
    with artifact_path.open("w", encoding="utf-8") as stream:
        for key in sorted(upstream):
            upstream_result = upstream[key]
            dynamo_result = dynamo[key]
            record = {
                "key": key,
                "request": requests_by_key[key],
                "upstream": {
                    "status": upstream_result.status,
                    "body": upstream_result.body,
                },
                "dynamo": {
                    "status": dynamo_result.status,
                    "body": dynamo_result.body,
                },
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")

            if upstream_result.status != dynamo_result.status:
                failures.append(
                    f"{key}: status upstream={upstream_result.status} "
                    f"dynamo={dynamo_result.status}"
                )
                continue
            if upstream_result.status != 200:
                failures.append(
                    f"{key}: both servers returned {upstream_result.status}"
                )
                continue

            upstream_body = normalize_response(upstream_result.body, key, "upstream")
            dynamo_body = normalize_response(dynamo_result.body, key, "dynamo")
            if upstream_body != dynamo_body:
                before = json.dumps(
                    upstream_body, indent=2, sort_keys=True
                ).splitlines()
                after = json.dumps(dynamo_body, indent=2, sort_keys=True).splitlines()
                diff = "\n".join(
                    difflib.unified_diff(
                        before, after, fromfile="upstream", tofile="dynamo", lineterm=""
                    )
                )
                failures.append(f"{key}: response mismatch\n{diff}")

    if failures:
        failure_path = output_dir / "failures.txt"
        failure_path.write_text("\n\n".join(failures) + "\n", encoding="utf-8")
        raise AssertionError(
            f"{len(failures)} parity checks failed; see {failure_path}"
        )


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_health(port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
            last_error = f"health status={response.status_code}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"server on port {port} was not healthy: {last_error}")


def wait_for_model(port: int, model: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if response.status_code == 200:
                names = {item.get("id") for item in response.json().get("data", [])}
                if model in names:
                    time.sleep(2)
                    return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"model {model} was not registered on port {port}")


def wait_for_ready_file(path: Path, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"server exited with status {exit_code} before creating {path}"
            )
        time.sleep(1)
    raise TimeoutError(f"server did not create topology readiness file {path}")


@contextmanager
def run_server(
    name: str,
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    port: int,
    timeout: int,
    model: str,
    ready_file: Path | None = None,
) -> Iterator[None]:
    if port_is_open(port):
        raise RuntimeError(f"port {port} is already in use before starting {name}")
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            wait_for_health(port, timeout)
            if ready_file is not None:
                wait_for_ready_file(ready_file, process, timeout)
            wait_for_model(port, model, timeout)
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            deadline = time.monotonic() + 30
            while port_is_open(port) and time.monotonic() < deadline:
                time.sleep(0.5)


def write_engine_config(output_dir: Path, model: str, topology: str) -> Path:
    path = output_dir / "engine-config.json"
    config = {
        "model": model,
        "served_model_name": [model],
        "tensor_parallel_size": 1,
        "max_model_len": MODEL_MAX_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MODEL_MAX_LEN,
        "kv_cache_memory_bytes": KV_CACHE_BYTES,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "generation_config": "vllm",
        "mm_processor_cache_gb": 0,
    }
    if topology == "disaggregated":
        # Explicit KV bytes own cache sizing. A low fraction only disables
        # vLLM's co-resident free-memory rejection for the second model copy.
        config["gpu_memory_utilization"] = 0.01
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def upstream_command(model: str, topology: str) -> list[str]:
    command = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        model,
        "--port",
        str(UPSTREAM_PORT),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(MODEL_MAX_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MODEL_MAX_LEN),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_BYTES),
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--mm-processor-cache-gb",
        "0",
    ]
    if topology == "disaggregated":
        command.extend(("--gpu-memory-utilization", "0.01"))
    return command


def dynamo_command(
    model: str,
    engine_config: Path,
    topology: str,
    startup_timeout: int,
    ready_file: Path,
) -> list[str]:
    if topology == "aggregated":
        return [
            "bash",
            str(REPO_ROOT / "examples/backends/vllm/launch/agg_multimodal.sh"),
            "--model",
            model,
            "--engine-config-json",
            str(engine_config),
        ]
    return [
        "bash",
        str(REPO_ROOT / "tests/serve/tito_parity/launch_dynamo_disagg.sh"),
        model,
        str(engine_config),
        str(startup_timeout),
        str(ready_file),
    ]


def base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        }
    )
    return environment


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT
        / "logs"
        / "tito-parity"
        / f"{timestamp}-{args.dynamo_topology}-{args.suite}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_config = write_engine_config(output_dir, args.model, args.dynamo_topology)
    topology_ready_file = output_dir / "dynamo-topology.ready"
    cases = cases_for_suite(args.suite)
    request_timeout = max(180, args.startup_timeout)

    environment = base_environment()
    with run_server(
        "upstream-vllm",
        upstream_command(args.model, args.dynamo_topology),
        environment,
        output_dir / "upstream.log",
        UPSTREAM_PORT,
        args.startup_timeout,
        args.model,
    ):
        rendered = render_cases(cases, args.model, args.suite, request_timeout)
        (output_dir / "rendered-requests.json").write_text(
            json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        groups = build_runs(cases, rendered, args.suite, args.max_concurrency)
        upstream_results, requests_by_key = execute_runs(
            UPSTREAM_PORT, groups, request_timeout
        )

    dynamo_environment = environment.copy()
    dynamo_environment.update(
        {
            "DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE": "1",
            "DYN_HTTP_BODY_LIMIT_MB": "200",
            "DYN_FILE_KV_TTL_SECS": "1800",
            "MAX_MODEL_LEN": str(MODEL_MAX_LEN),
            "MAX_CONCURRENT_SEQS": str(MAX_NUM_SEQS),
            "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES": str(KV_CACHE_BYTES),
        }
    )
    with run_server(
        "dynamo-vllm",
        dynamo_command(
            args.model,
            engine_config,
            args.dynamo_topology,
            args.startup_timeout,
            topology_ready_file,
        ),
        dynamo_environment,
        output_dir / "dynamo.log",
        DYNAMO_PORT,
        args.startup_timeout,
        args.model,
        topology_ready_file if args.dynamo_topology == "disaggregated" else None,
    ):
        dynamo_results, dynamo_requests = execute_runs(
            DYNAMO_PORT, groups, request_timeout
        )

    if requests_by_key != dynamo_requests:
        raise AssertionError("upstream and Dynamo did not receive identical requests")
    compare_results(upstream_results, dynamo_results, requests_by_key, output_dir)
    print(
        f"TITO parity passed for {len(upstream_results)} requests "
        f"({args.dynamo_topology}): {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
