# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import copy
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import run_parity as parity
import run_pd_three_way as pd_parity
from PIL import Image
from transformers import AutoTokenizer

TASK_SEED = 42
TASK_INDEX = 7
MAX_TURNS = 3
COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
COLOR_MAP = {
    "red": "A",
    "green": "B",
    "blue": "C",
    "yellow": "D",
    "purple": "E",
    "cyan": "F",
    "orange": "G",
    "white": "H",
    "black": "I",
}
SYSTEM_PROMPT = """You will be shown colored squares across multiple turns. Each color maps to a letter:

Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I

Example: Turn 1 shows Red, Blue. Turn 2 shows Green, Yellow. The full codeword is "ACBD" (all 4 letters in order).

After each turn, output your accumulated codeword so far. Output ONLY the letters with NO spaces."""
FATAL_LOG_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "engine-dead": re.compile(r"EngineDeadError"),
    "cuda-oom": re.compile(r"CUDA out of memory", re.IGNORECASE),
    "process-oom": re.compile(r"OutOfMemoryError"),
    "panic": re.compile(r"\bpanicked?\b", re.IGNORECASE),
    "segfault": re.compile(r"segmentation fault|core dumped", re.IGNORECASE),
}


@dataclass(frozen=True)
class Cohort:
    seed_base: int
    answer: str
    requests: dict[str, dict[str, Any]]
    groups: list[tuple[int, list[parity.RequestRun]]]
    responses: dict[str, parity.HttpResult]
    decoded: dict[str, str]
    rewards: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one seeded multimodal RL cohort through native vLLM, "
            "Dynamo aggregated, and Dynamo P/D"
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.35)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="Number of alternative logprobs; zero still returns sampled-token logprobs",
    )
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--seed-search-attempts", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def task_sequence(task_index: int) -> tuple[list[str], str]:
    rng = random.Random(TASK_SEED)
    colors = list(COLOR_MAP)
    sequence: list[str] = []
    for _ in range(task_index + 1):
        sequence = [rng.choice(colors) for _ in range(MAX_TURNS)]
    return sequence, "".join(COLOR_MAP[color] for color in sequence)


def color_data_url(color: str) -> str:
    buffer = BytesIO()
    Image.new("RGB", (100, 100), COLOR_RGB[color]).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def user_message(color: str, turn: int) -> dict[str, Any]:
    if turn == 0:
        text = "Here are 1 squares."
    elif turn == MAX_TURNS - 1:
        text = (
            "Here are 1 more squares. Combine your previous answer with these "
            f"new letters to output all {turn + 1} letters."
        )
    else:
        text = "Here are 1 more squares."
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": color_data_url(color)},
            },
            {"type": "text", "text": text},
        ],
    }


def extract_codeword(text: str) -> str:
    upper = text.upper()
    matches = re.findall(r"\b[A-I]+\b", upper)
    if matches:
        return max(matches, key=len)
    return "".join(character for character in upper if character in "ABCDEFGHI")


def render_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
    seed: int,
    turn: int,
    temperature: float,
    max_tokens: int,
    logprobs: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = requests.post(
        f"http://127.0.0.1:{parity.UPSTREAM_PORT}/v1/chat/completions/render",
        json=payload,
        timeout=timeout,
    )
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(
            f"render failed for seed={seed} turn={turn}: "
            f"status={response.status_code} body={body}"
        )
    if isinstance(body, list):
        if len(body) != 1:
            raise RuntimeError(
                f"render returned {len(body)} requests for seed={seed} turn={turn}"
            )
        body = body[0]
    if not isinstance(body, dict):
        raise RuntimeError(
            f"render returned {type(body).__name__} for seed={seed} turn={turn}"
        )

    request = copy.deepcopy(body)
    request["request_id"] = f"seeded-rl-s{seed}-t{turn}"
    request["model"] = model
    request["stream"] = False
    request["cache_salt"] = "seeded-rl-policy-0"
    sampling = request.setdefault("sampling_params", {})
    sampling.update(
        {
            "temperature": temperature,
            "top_p": 1.0,
            "n": 1,
            "seed": seed,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "skip_special_tokens": False,
        }
    )
    return request


def first_choice(result: parity.HttpResult, key: str) -> dict[str, Any]:
    if result.status != 200:
        raise RuntimeError(f"{key} returned status={result.status}: {result.body}")
    choices = result.body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"{key} returned invalid choices: {choices!r}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"{key} returned a non-object choice")
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError(f"{key} returned invalid token_ids={token_ids!r}")
    return choice


def run_native_cohort(
    *,
    model: str,
    tokenizer: Any,
    group_size: int,
    seed_base: int,
    temperature: float,
    max_tokens: int,
    logprobs: int,
    timeout: int,
) -> Cohort:
    colors, answer = task_sequence(TASK_INDEX)
    seeds = list(range(seed_base, seed_base + group_size))
    messages = {seed: [{"role": "system", "content": SYSTEM_PROMPT}] for seed in seeds}
    requests_by_key: dict[str, dict[str, Any]] = {}
    responses: dict[str, parity.HttpResult] = {}
    decoded: dict[str, str] = {}
    groups: list[tuple[int, list[parity.RequestRun]]] = []

    for turn, color in enumerate(colors):
        runs: list[parity.RequestRun] = []
        for seed in seeds:
            messages[seed].append(user_message(color, turn))
            key = f"seed-{seed}-turn-{turn}"
            request = render_request(
                model=model,
                messages=messages[seed],
                seed=seed,
                turn=turn,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                timeout=timeout,
            )
            requests_by_key[key] = request
            runs.append(parity.RequestRun(key=key, request=request))

        turn_results = parity.execute_group(
            parity.UPSTREAM_PORT, group_size, runs, timeout
        )
        responses.update(turn_results)
        groups.append((group_size, runs))
        for seed in seeds:
            key = f"seed-{seed}-turn-{turn}"
            token_ids = first_choice(turn_results[key], key)["token_ids"]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded[key] = text
            messages[seed].append({"role": "assistant", "content": text})

    rewards = {}
    for seed in seeds:
        key = f"seed-{seed}-turn-{MAX_TURNS - 1}"
        rewards[str(seed)] = float(extract_codeword(decoded[key]) == answer)
    return Cohort(
        seed_base=seed_base,
        answer=answer,
        requests=requests_by_key,
        groups=groups,
        responses=responses,
        decoded=decoded,
        rewards=rewards,
    )


def execute_groups(
    port: int,
    groups: list[tuple[int, list[parity.RequestRun]]],
    timeout: int,
) -> dict[str, parity.HttpResult]:
    results: dict[str, parity.HttpResult] = {}
    for concurrency, runs in groups:
        results.update(parity.execute_group(port, concurrency, runs, timeout))
    return results


def validate_multimodal_manifest(requests_by_key: dict[str, dict[str, Any]]) -> int:
    required_feature_keys = {"kwargs_data", "mm_hashes", "mm_placeholders"}
    for key, request in requests_by_key.items():
        token_ids = request.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"{key} is missing rendered token_ids")
        features = request.get("features")
        if not isinstance(features, dict):
            raise RuntimeError(f"{key} is missing the rendered VLM features object")
        if features.keys() != required_feature_keys:
            raise RuntimeError(
                f"{key} feature keys differ: "
                f"{sorted(features)} != {sorted(required_feature_keys)}"
            )
        for feature_name, value in features.items():
            if not isinstance(value, dict) or "image" not in value:
                raise RuntimeError(f"{key} features.{feature_name}.image is missing")
    return len(requests_by_key)


def check_server_health(port: int, model: str, label: str) -> dict[str, int]:
    health = requests.get(f"http://127.0.0.1:{port}/health", timeout=30)
    models = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=30)
    if health.status_code != 200:
        raise RuntimeError(f"{label} post-wave /health returned {health.status_code}")
    if models.status_code != 200:
        raise RuntimeError(
            f"{label} post-wave /v1/models returned {models.status_code}"
        )
    models_body = models.json()
    model_ids = {
        item.get("id") for item in models_body.get("data", []) if isinstance(item, dict)
    }
    if model not in model_ids:
        raise RuntimeError(f"{label} post-wave model list is missing {model}")
    return {"health": health.status_code, "models": models.status_code}


def workload_log_boundary(path: Path) -> int:
    size = path.stat().st_size
    if size == 0:
        raise RuntimeError(f"{path} is empty at the post-wave health boundary")
    return size


def assert_no_pre_shutdown_fatal_signatures(path: Path, boundary: int) -> int:
    workload_log = path.read_bytes()[:boundary].decode("utf-8", errors="replace")
    signatures = [
        name
        for name, pattern in FATAL_LOG_PATTERNS.items()
        if pattern.search(workload_log)
    ]
    if signatures:
        raise RuntimeError(
            f"{path} contains pre-shutdown fatal signatures: {signatures}"
        )
    return 0


def normalized_response(body: dict[str, Any], key: str, side: str) -> dict[str, Any]:
    normalized = parity.normalize_response(body, key, side)

    def normalize_bytes(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                item_key: None if item_key == "bytes" else normalize_bytes(item_value)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [normalize_bytes(item) for item in value]
        return value

    return normalize_bytes(normalized)


def semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, (int, float)):
        # Separate server processes can select numerically equivalent reduced-
        # precision kernels while still producing identical sampled tokens.
        return math.isclose(left, float(right), rel_tol=1e-5, abs_tol=5e-4)
    if isinstance(right, float) and isinstance(left, (int, float)):
        return math.isclose(float(left), right, rel_tol=1e-5, abs_tol=5e-4)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantic_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantic_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def first_mismatch(left: Any, right: Any, path: str = "response") -> str | None:
    if semantic_equal(left, right):
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return f"{path} keys differ: {sorted(left)} != {sorted(right)}"
        for key in left:
            mismatch = first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} lengths differ: {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = first_mismatch(a, b, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
    return f"{path} differs: {left!r} != {right!r}"


def compare_results(
    *,
    reference: dict[str, parity.HttpResult],
    candidate: dict[str, parity.HttpResult],
    side: str,
) -> list[str]:
    failures: list[str] = []
    if reference.keys() != candidate.keys():
        return [
            f"{side} response keys differ: "
            f"{sorted(reference)} != {sorted(candidate)}"
        ]
    for key in sorted(reference):
        expected = reference[key]
        actual = candidate[key]
        if expected.status != actual.status:
            failures.append(
                f"{key}: status native={expected.status} {side}={actual.status}"
            )
            continue
        if expected.status != 200:
            failures.append(f"{key}: both sides returned non-200 {expected.status}")
            continue
        expected_body = normalized_response(expected.body, key, "native")
        actual_body = normalized_response(actual.body, key, side)
        mismatch = first_mismatch(expected_body, actual_body)
        if mismatch is not None:
            failures.append(f"{key}: {mismatch}")
    return failures


def write_results(path: Path, results: dict[str, parity.HttpResult]) -> None:
    path.write_text(
        json.dumps(
            {key: asdict(result) for key, result in sorted(results.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def reward_summary(rewards: dict[str, float]) -> tuple[float, dict[str, float]]:
    mean = sum(rewards.values()) / len(rewards)
    return mean, {seed: reward - mean for seed, reward in rewards.items()}


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("group-size must be at least 2 for group-relative advantages")
    if args.seed_search_attempts < 1:
        raise ValueError("seed-search-attempts must be positive")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        parity.REPO_ROOT / "logs" / "tito-seeded-rl-parity" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_timeout = max(180, args.startup_timeout)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )

    environment = parity.base_environment()
    environment["VLLM_BATCH_INVARIANT"] = "1"
    with parity.run_server(
        "upstream-vllm",
        parity.upstream_command(args.model, "aggregated"),
        environment,
        output_dir / "native-vllm.log",
        parity.UPSTREAM_PORT,
        args.startup_timeout,
        args.model,
    ):
        initial_seed = args.seed_base or 0
        attempts = 1 if args.seed_base is not None else args.seed_search_attempts
        cohort: Cohort | None = None
        calibration: list[dict[str, Any]] = []
        for attempt in range(attempts):
            seed_base = initial_seed + attempt * args.group_size
            candidate = run_native_cohort(
                model=args.model,
                tokenizer=tokenizer,
                group_size=args.group_size,
                seed_base=seed_base,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                logprobs=args.logprobs,
                timeout=request_timeout,
            )
            correct = int(sum(candidate.rewards.values()))
            calibration.append({"seed_base": seed_base, "correct": correct})
            if 0 < correct < args.group_size:
                cohort = candidate
                break
        if cohort is None:
            raise AssertionError(
                "no mixed-reward native cohort found; " f"calibration={calibration}"
            )
        native_health = check_server_health(
            parity.UPSTREAM_PORT, args.model, "native-vllm"
        )
        native_log_boundary = workload_log_boundary(output_dir / "native-vllm.log")

    native_fatal_signatures = assert_no_pre_shutdown_fatal_signatures(
        output_dir / "native-vllm.log", native_log_boundary
    )
    multimodal_requests = validate_multimodal_manifest(cohort.requests)

    (output_dir / "rendered-requests.json").write_text(
        json.dumps(cohort.requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_results(output_dir / "native-responses.json", cohort.responses)

    native_pd_dir = output_dir / "native-pd"
    native_pd_dir.mkdir()
    with pd_parity.upstream_pd_pair(
        args.model,
        environment,
        native_pd_dir,
        1,
        args.startup_timeout,
    ):
        native_pd_results, native_pd_requests = pd_parity.execute_pd_runs(
            cohort.groups, request_timeout
        )
        if native_pd_requests != cohort.requests:
            raise AssertionError(
                "native aggregated and P/D received different requests"
            )
        native_pd_health = {
            "prefill": check_server_health(
                pd_parity.PREFILL_PORT, args.model, "native-pd-prefill"
            ),
            "decode": check_server_health(
                pd_parity.DECODE_PORT, args.model, "native-pd-decode"
            ),
        }
        native_pd_log_boundaries = {
            "prefill": workload_log_boundary(
                native_pd_dir / "upstream-run-1-prefill.log"
            ),
            "decode": workload_log_boundary(
                native_pd_dir / "upstream-run-1-decode.log"
            ),
        }
    native_pd_fatal_signatures = {
        role: assert_no_pre_shutdown_fatal_signatures(
            native_pd_dir / f"upstream-run-1-{role}.log", boundary
        )
        for role, boundary in native_pd_log_boundaries.items()
    }
    write_results(native_pd_dir / "responses.json", native_pd_results)
    native_topology_failures = compare_results(
        reference=cohort.responses,
        candidate=native_pd_results,
        side="native-pd",
    )
    if native_topology_failures:
        (native_pd_dir / "aggregated-differences.txt").write_text(
            "\n".join(native_topology_failures) + "\n", encoding="utf-8"
        )

    agg_dir = output_dir / "dynamo-aggregated"
    agg_dir.mkdir()
    agg_engine_config = parity.write_engine_config(agg_dir, args.model, "aggregated")
    dynamo_environment = environment.copy()
    dynamo_environment.update(
        {
            "DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE": "1",
            "DYN_HTTP_BODY_LIMIT_MB": "200",
            "DYN_FILE_KV_TTL_SECS": "1800",
            "MAX_MODEL_LEN": str(parity.MODEL_MAX_LEN),
            "MAX_CONCURRENT_SEQS": str(parity.MAX_NUM_SEQS),
            "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES": str(parity.KV_CACHE_BYTES),
        }
    )
    with parity.run_server(
        "dynamo-aggregated",
        parity.dynamo_command(
            args.model,
            agg_engine_config,
            "aggregated",
            args.startup_timeout,
            agg_dir / "unused.ready",
        ),
        dynamo_environment,
        agg_dir / "server.log",
        parity.DYNAMO_PORT,
        args.startup_timeout,
        args.model,
    ):
        agg_results = execute_groups(parity.DYNAMO_PORT, cohort.groups, request_timeout)
        agg_health = check_server_health(
            parity.DYNAMO_PORT, args.model, "dynamo-aggregated"
        )
        agg_log_boundary = workload_log_boundary(agg_dir / "server.log")
    agg_fatal_signatures = assert_no_pre_shutdown_fatal_signatures(
        agg_dir / "server.log", agg_log_boundary
    )
    write_results(agg_dir / "responses.json", agg_results)
    agg_failures = compare_results(
        reference=cohort.responses,
        candidate=agg_results,
        side="dynamo-aggregated",
    )
    if agg_failures:
        (agg_dir / "failures.txt").write_text(
            "\n".join(agg_failures) + "\n", encoding="utf-8"
        )
        raise AssertionError(
            f"Dynamo aggregated had {len(agg_failures)} mismatch(es); "
            f"see {agg_dir / 'failures.txt'}"
        )

    pd_dir = output_dir / "dynamo-pd"
    pd_dir.mkdir()
    pd_engine_config = parity.write_engine_config(pd_dir, args.model, "disaggregated")
    pd_ready = pd_dir / "topology.ready"
    with parity.run_server(
        "dynamo-pd",
        parity.dynamo_command(
            args.model,
            pd_engine_config,
            "disaggregated",
            args.startup_timeout,
            pd_ready,
        ),
        dynamo_environment,
        pd_dir / "server.log",
        parity.DYNAMO_PORT,
        args.startup_timeout,
        args.model,
        pd_ready,
    ):
        pd_results = execute_groups(parity.DYNAMO_PORT, cohort.groups, request_timeout)
        pd_health = check_server_health(parity.DYNAMO_PORT, args.model, "dynamo-pd")
        pd_log_boundary = workload_log_boundary(pd_dir / "server.log")
    pd_fatal_signatures = assert_no_pre_shutdown_fatal_signatures(
        pd_dir / "server.log", pd_log_boundary
    )
    write_results(pd_dir / "responses.json", pd_results)
    pd_failures = compare_results(
        reference=native_pd_results,
        candidate=pd_results,
        side="dynamo-pd",
    )
    if pd_failures:
        (pd_dir / "failures.txt").write_text(
            "\n".join(pd_failures) + "\n", encoding="utf-8"
        )
        raise AssertionError(
            f"Dynamo P/D had {len(pd_failures)} mismatch(es); "
            f"see {pd_dir / 'failures.txt'}"
        )

    reward_mean, advantages = reward_summary(cohort.rewards)
    trainable = sum(value != 0.0 for value in advantages.values())
    summary = {
        "model": args.model,
        "task_index": TASK_INDEX,
        "answer": cohort.answer,
        "temperature": args.temperature,
        "logprobs": args.logprobs,
        "group_size": args.group_size,
        "seed_base": cohort.seed_base,
        "calibration": calibration,
        "rewards": cohort.rewards,
        "reward_mean": reward_mean,
        "advantages": advantages,
        "trainable": trainable,
        "requests": len(cohort.responses),
        "multimodal_requests": multimodal_requests,
        "native_aggregated_post_wave_health": native_health,
        "native_pd_post_wave_health": native_pd_health,
        "dynamo_aggregated_post_wave_health": agg_health,
        "dynamo_pd_post_wave_health": pd_health,
        "native_aggregated_pre_shutdown_fatal_signatures": native_fatal_signatures,
        "native_pd_pre_shutdown_fatal_signatures": native_pd_fatal_signatures,
        "dynamo_aggregated_pre_shutdown_fatal_signatures": agg_fatal_signatures,
        "dynamo_pd_pre_shutdown_fatal_signatures": pd_fatal_signatures,
        "native_aggregated_http_200": sum(
            result.status == 200 for result in cohort.responses.values()
        ),
        "native_pd_http_200": sum(
            result.status == 200 for result in native_pd_results.values()
        ),
        "dynamo_aggregated_http_200": sum(
            result.status == 200 for result in agg_results.values()
        ),
        "dynamo_pd_http_200": sum(
            result.status == 200 for result in pd_results.values()
        ),
        "native_topology_semantic_mismatches": len(native_topology_failures),
        "dynamo_aggregated_semantic_mismatches": len(agg_failures),
        "dynamo_pd_semantic_mismatches": len(pd_failures),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Seeded RL parity passed: "
        f"native aggregated = Dynamo aggregated and native P/D = Dynamo P/D "
        f"for {len(cohort.responses)} "
        f"requests; rewards={int(sum(cohort.rewards.values()))}/{args.group_size}; "
        f"trainable={trainable}/{args.group_size}; upstream topology "
        f"mismatches={len(native_topology_failures)}; {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
