# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import requests
import run_parity as parity

PREFILL_PORT = 8100
DECODE_PORT = 8200
UPSTREAM_RUNS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two cold-started upstream vLLM P/D runs with one "
            "Dynamo vLLM P/D run"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--suite", choices=("smoke", "expanded", "full"), default="smoke"
    )
    parser.add_argument("--max-concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def upstream_worker_command(model: str, port: int) -> list[str]:
    return [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(parity.MODEL_MAX_LEN),
        "--max-num-seqs",
        str(parity.MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(parity.MODEL_MAX_LEN),
        "--kv-cache-memory-bytes",
        str(parity.KV_CACHE_BYTES),
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--mm-processor-cache-gb",
        "0",
        "--gpu-memory-utilization",
        "0.01",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    ]


@contextmanager
def upstream_pd_pair(
    model: str,
    environment: dict[str, str],
    output_dir: Path,
    run_number: int,
    timeout: int,
) -> Iterator[None]:
    prefix = f"upstream-run-{run_number}"
    decode_environment = environment.copy()
    decode_environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "20099"
    prefill_environment = environment.copy()
    prefill_environment["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "20098"
    with parity.run_server(
        f"{prefix}-decode",
        upstream_worker_command(model, DECODE_PORT),
        decode_environment,
        output_dir / f"{prefix}-decode.log",
        DECODE_PORT,
        timeout,
        model,
    ):
        with parity.run_server(
            f"{prefix}-prefill",
            upstream_worker_command(model, PREFILL_PORT),
            prefill_environment,
            output_dir / f"{prefix}-prefill.log",
            PREFILL_PORT,
            timeout,
            model,
        ):
            yield


def render_cases(
    cases: list[parity.Case], model: str, suite: str, timeout: int
) -> dict[str, dict[str, Any]]:
    rendered: dict[str, dict[str, Any]] = {}
    for case in cases:
        max_tokens = 8 if case.name.endswith("smoke") else 64
        path, payload = parity.render_payload(case, model, max_tokens)
        result = requests.post(
            f"http://127.0.0.1:{DECODE_PORT}{path}",
            json=payload,
            timeout=timeout,
        )
        body = result.json()
        if result.status_code != 200:
            raise RuntimeError(
                f"render failed for {case.name}: status={result.status_code} body={body}"
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
        request["request_id"] = f"pd-three-way-{suite}-{case.name}"
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


def with_kv_transfer(
    request: dict[str, Any], kv_transfer_params: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(request)
    sampling = result["sampling_params"]
    extra_args = dict(sampling.get("extra_args") or {})
    extra_args["kv_transfer_params"] = kv_transfer_params
    sampling["extra_args"] = extra_args
    return result


def safe_post_json(
    port: int, payload: dict[str, Any], timeout: int, phase: str
) -> parity.HttpResult:
    try:
        return parity.post_json(port, parity.GENERATE_PATH, payload, timeout)
    except (requests.RequestException, RuntimeError, ValueError) as error:
        return parity.HttpResult(
            status=0,
            body={"error": {"message": str(error), "phase": phase}},
        )


def execute_pd_request(
    run: parity.RequestRun, timeout: int
) -> tuple[str, parity.HttpResult]:
    prefill_kv = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request = with_kv_transfer(run.request, prefill_kv)
    prefill_sampling = prefill_request["sampling_params"]
    prefill_sampling["max_tokens"] = 1
    prefill_sampling["min_tokens"] = 1
    prefill_result = safe_post_json(PREFILL_PORT, prefill_request, timeout, "prefill")
    if prefill_result.status != 200:
        return run.key, prefill_result

    kv_transfer_params = prefill_result.body.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict) or not kv_transfer_params:
        return run.key, parity.HttpResult(
            status=0,
            body={
                "error": {
                    "message": "prefill returned no kv_transfer_params",
                    "phase": "prefill",
                    "response": prefill_result.body,
                }
            },
        )

    decode_request = with_kv_transfer(run.request, kv_transfer_params)
    requested_prompt_logprobs = decode_request["sampling_params"].get("prompt_logprobs")
    if requested_prompt_logprobs is not None:
        # Prompt logprobs are produced by P while it evaluates the prompt. D
        # consumes transferred KV and must not recompute them from skipped
        # prompt positions. The proxy composes P's metadata with D's output.
        decode_request["sampling_params"]["prompt_logprobs"] = None
    decode_result = safe_post_json(DECODE_PORT, decode_request, timeout, "decode")
    if decode_result.status != 200 or requested_prompt_logprobs is None:
        return run.key, decode_result

    body = copy.deepcopy(decode_result.body)
    body["prompt_logprobs"] = prefill_result.body.get("prompt_logprobs")
    return run.key, parity.HttpResult(status=200, body=body)


def execute_pd_runs(
    groups: list[tuple[int, list[parity.RequestRun]]], timeout: int
) -> tuple[dict[str, parity.HttpResult], dict[str, dict[str, Any]]]:
    results: dict[str, parity.HttpResult] = {}
    requests_by_key: dict[str, dict[str, Any]] = {}
    for concurrency, runs in groups:
        requests_by_key.update({run.key: run.request for run in runs})
        if concurrency == 1:
            results.update(dict(execute_pd_request(run, timeout) for run in runs))
            continue
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(execute_pd_request, run, timeout) for run in runs
            ]
            for future in as_completed(futures):
                key, result = future.result()
                results[key] = result
    return results, requests_by_key


def execute_safe_runs(
    port: int,
    groups: list[tuple[int, list[parity.RequestRun]]],
    timeout: int,
    phase: str,
) -> tuple[dict[str, parity.HttpResult], dict[str, dict[str, Any]]]:
    results: dict[str, parity.HttpResult] = {}
    requests_by_key: dict[str, dict[str, Any]] = {}
    for concurrency, runs in groups:
        requests_by_key.update({run.key: run.request for run in runs})
        if concurrency == 1:
            for run in runs:
                results[run.key] = safe_post_json(port, run.request, timeout, phase)
            continue
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    safe_post_json, port, run.request, timeout, phase
                ): run.key
                for run in runs
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    return results, requests_by_key


def result_record(result: parity.HttpResult) -> dict[str, Any]:
    return asdict(result)


def normalized_body(
    result: parity.HttpResult, key: str, side: str
) -> dict[str, Any] | None:
    if result.status != 200:
        return None
    return parity.normalize_response(result.body, key, side)


def choice_field(body: dict[str, Any] | None, name: str) -> Any:
    if body is None:
        return None
    choices = body.get("choices")
    if not isinstance(choices, list):
        return None
    return [choice.get(name) for choice in choices]


def choice_field_equal(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    name: str,
) -> bool:
    return (
        left is not None
        and right is not None
        and choice_field(left, name) == choice_field(right, name)
    )


def response_field_equal(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    name: str,
) -> bool:
    return left is not None and right is not None and left.get(name) == right.get(name)


def compare_three_way(
    upstream_runs: list[dict[str, parity.HttpResult]],
    dynamo: dict[str, parity.HttpResult],
    requests_by_key: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    failures: list[str] = []
    counts = {
        "total": 0,
        "all_http_200": 0,
        "upstream_repeat_exact": 0,
        "dynamo_equals_upstream_run_1": 0,
        "dynamo_equals_upstream_run_2": 0,
        "upstream_repeat_token_ids_equal": 0,
        "dynamo_run_1_token_ids_equal": 0,
        "dynamo_run_2_token_ids_equal": 0,
    }
    artifact_path = output_dir / "responses.jsonl"
    with artifact_path.open("w", encoding="utf-8") as stream:
        for key in sorted(requests_by_key):
            counts["total"] += 1
            run_1_result = upstream_runs[0][key]
            run_2_result = upstream_runs[1][key]
            dynamo_result = dynamo[key]
            run_1 = normalized_body(run_1_result, key, "upstream_run_1")
            run_2 = normalized_body(run_2_result, key, "upstream_run_2")
            dynamo_body = normalized_body(dynamo_result, key, "dynamo")

            all_http_200 = all(
                result.status == 200
                for result in (run_1_result, run_2_result, dynamo_result)
            )
            upstream_repeat_exact = run_1 is not None and run_1 == run_2
            dynamo_equals_run_1 = run_1 is not None and run_1 == dynamo_body
            dynamo_equals_run_2 = run_2 is not None and run_2 == dynamo_body
            upstream_tokens_equal = choice_field_equal(run_1, run_2, "token_ids")
            dynamo_run_1_tokens_equal = choice_field_equal(
                run_1, dynamo_body, "token_ids"
            )
            dynamo_run_2_tokens_equal = choice_field_equal(
                run_2, dynamo_body, "token_ids"
            )
            request = requests_by_key[key]
            run_1_missing_logprobs = parity.missing_requested_logprob_fields(
                run_1, request
            )
            run_2_missing_logprobs = parity.missing_requested_logprob_fields(
                run_2, request
            )
            dynamo_missing_logprobs = parity.missing_requested_logprob_fields(
                dynamo_body, request
            )

            checks = {
                "all_http_200": all_http_200,
                "upstream_repeat_exact": upstream_repeat_exact,
                "dynamo_equals_upstream_run_1": dynamo_equals_run_1,
                "dynamo_equals_upstream_run_2": dynamo_equals_run_2,
                "upstream_repeat_token_ids_equal": upstream_tokens_equal,
                "dynamo_run_1_token_ids_equal": dynamo_run_1_tokens_equal,
                "dynamo_run_2_token_ids_equal": dynamo_run_2_tokens_equal,
                "upstream_repeat_logprobs_equal": choice_field_equal(
                    run_1, run_2, "logprobs"
                ),
                "dynamo_run_1_logprobs_equal": choice_field_equal(
                    run_1, dynamo_body, "logprobs"
                ),
                "dynamo_run_2_logprobs_equal": choice_field_equal(
                    run_2, dynamo_body, "logprobs"
                ),
                "upstream_repeat_prompt_logprobs_equal": response_field_equal(
                    run_1, run_2, "prompt_logprobs"
                ),
                "dynamo_run_1_prompt_logprobs_equal": response_field_equal(
                    run_1, dynamo_body, "prompt_logprobs"
                ),
                "dynamo_run_2_prompt_logprobs_equal": response_field_equal(
                    run_2, dynamo_body, "prompt_logprobs"
                ),
                "upstream_run_1_requested_logprobs_populated": (
                    not run_1_missing_logprobs
                ),
                "upstream_run_2_requested_logprobs_populated": (
                    not run_2_missing_logprobs
                ),
                "dynamo_requested_logprobs_populated": not dynamo_missing_logprobs,
            }
            for count_name in counts:
                if count_name != "total" and checks.get(count_name):
                    counts[count_name] += 1

            record = {
                "key": key,
                "request": requests_by_key[key],
                "checks": checks,
                "upstream_run_1": result_record(run_1_result),
                "upstream_run_2": result_record(run_2_result),
                "dynamo": result_record(dynamo_result),
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            if not all_http_200:
                failures.append(
                    f"{key}: HTTP statuses upstream1={run_1_result.status} "
                    f"upstream2={run_2_result.status} dynamo={dynamo_result.status}"
                )
            if not upstream_repeat_exact:
                failures.append(f"{key}: upstream P/D run 1 != run 2")
            if not dynamo_equals_run_1 or not dynamo_equals_run_2:
                failures.append(f"{key}: Dynamo P/D != both upstream P/D runs")
            for side, missing in (
                ("upstream P/D run 1", run_1_missing_logprobs),
                ("upstream P/D run 2", run_2_missing_logprobs),
                ("Dynamo P/D", dynamo_missing_logprobs),
            ):
                if missing:
                    failures.append(
                        f"{key}: {side} omitted requested fields: "
                        f"{', '.join(missing)}"
                    )

    (output_dir / "summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        failure_path = output_dir / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(json.dumps(counts, indent=2, sort_keys=True))
        raise AssertionError(
            f"{len(failures)} three-way checks failed; see {failure_path}"
        )


def write_partial_results(path: Path, results: dict[str, parity.HttpResult]) -> None:
    path.write_text(
        json.dumps(
            {key: result_record(value) for key, value in sorted(results.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        parity.REPO_ROOT / "logs" / "tito-pd-three-way" / f"{timestamp}-{args.suite}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "suite": args.suite,
                "max_concurrency": args.max_concurrency,
                "upstream_pd_runs": UPSTREAM_RUNS,
                "vllm_version": version("vllm"),
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
                "response_normalization": ["request_id"],
                "ssm_conv_state_layout": "DS",
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    engine_config = parity.write_engine_config(output_dir, args.model, "disaggregated")
    topology_ready_file = output_dir / "dynamo-topology.ready"
    cases = parity.cases_for_suite(args.suite)
    request_timeout = max(180, args.startup_timeout)
    environment = parity.base_environment()
    environment["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"

    rendered: dict[str, dict[str, Any]] | None = None
    groups: list[tuple[int, list[parity.RequestRun]]] | None = None
    requests_by_key: dict[str, dict[str, Any]] | None = None
    upstream_results: list[dict[str, parity.HttpResult]] = []
    for run_number in range(1, UPSTREAM_RUNS + 1):
        with upstream_pd_pair(
            args.model,
            environment,
            output_dir,
            run_number,
            args.startup_timeout,
        ):
            if rendered is None:
                rendered = render_cases(cases, args.model, args.suite, request_timeout)
                (output_dir / "rendered-requests.json").write_text(
                    json.dumps(rendered, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                groups = parity.build_runs(
                    cases, rendered, args.suite, args.max_concurrency
                )
            assert groups is not None
            results, run_requests = execute_pd_runs(groups, request_timeout)
            write_partial_results(
                output_dir / f"upstream-run-{run_number}-responses.json", results
            )
            upstream_results.append(results)
            if requests_by_key is None:
                requests_by_key = run_requests
            elif requests_by_key != run_requests:
                raise AssertionError("upstream P/D runs received different requests")

    assert groups is not None
    assert requests_by_key is not None
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
        "dynamo-vllm-pd",
        parity.dynamo_command(
            args.model,
            engine_config,
            "disaggregated",
            args.startup_timeout,
            topology_ready_file,
        ),
        dynamo_environment,
        output_dir / "dynamo.log",
        parity.DYNAMO_PORT,
        args.startup_timeout,
        args.model,
        topology_ready_file,
    ):
        dynamo_results, dynamo_requests = execute_safe_runs(
            parity.DYNAMO_PORT,
            groups,
            request_timeout,
            "dynamo",
        )
        write_partial_results(output_dir / "dynamo-responses.json", dynamo_results)

    if requests_by_key != dynamo_requests:
        raise AssertionError("upstream and Dynamo received different requests")
    compare_three_way(upstream_results, dynamo_results, requests_by_key, output_dir)
    print(
        f"Three-way P/D parity passed for {len(requests_by_key)} requests: "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
