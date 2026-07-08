# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Prometheus checks for Dynamo's native Generate endpoint."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import run_parity as parity

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


MetricKey = tuple[str, tuple[tuple[str, str], ...]]
MetricSnapshot = dict[MetricKey, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify /inference/v1/generate frontend metric deltas"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--topology", choices=("aggregated", "disaggregated"), required=True
    )
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--rendered-requests",
        type=Path,
        help="Reuse a prior rendered-requests.json instead of launching vLLM to render.",
    )
    return parser.parse_args()


def parse_labels(raw: str | None) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    labels = []
    for match in LABEL_RE.finditer(raw):
        # Prometheus quoted strings use JSON-compatible escaping for the values
        # emitted by this service.
        labels.append((match.group(1), json.loads(f'"{match.group(2)}"')))
    return tuple(sorted(labels))


def scrape_metrics(port: int) -> tuple[str, MetricSnapshot]:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
    response.raise_for_status()
    snapshot: MetricSnapshot = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if match is None:
            continue
        snapshot[(match.group("name"), parse_labels(match.group("labels")))] = float(
            match.group("value")
        )
    return response.text, snapshot


def matching_values(
    snapshot: MetricSnapshot, name: str, required_labels: dict[str, str]
) -> list[float]:
    values = []
    for (candidate_name, raw_labels), value in snapshot.items():
        if candidate_name != name:
            continue
        labels = dict(raw_labels)
        if all(
            labels.get(key) == expected for key, expected in required_labels.items()
        ):
            values.append(value)
    return values


def metric_value(
    snapshot: MetricSnapshot, name: str, required_labels: dict[str, str]
) -> float:
    values = matching_values(snapshot, name, required_labels)
    if len(values) != 1:
        raise AssertionError(
            f"expected one {name} series for {required_labels}, found {len(values)}"
        )
    return values[0]


def metric_value_or_zero(
    snapshot: MetricSnapshot, name: str, required_labels: dict[str, str]
) -> float:
    values = matching_values(snapshot, name, required_labels)
    if not values:
        return 0.0
    if len(values) != 1:
        raise AssertionError(
            f"expected at most one {name} series for {required_labels}, found {len(values)}"
        )
    return values[0]


def metric_delta(
    before: MetricSnapshot,
    after: MetricSnapshot,
    name: str,
    required_labels: dict[str, str],
) -> float:
    return metric_value_or_zero(after, name, required_labels) - metric_value_or_zero(
        before, name, required_labels
    )


def output_tokens(result: parity.HttpResult) -> int:
    if result.status != 200:
        raise AssertionError(f"Generate returned {result.status}: {result.body}")
    choices = result.body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError(f"Generate response has no choices: {result.body}")
    return sum(len(choice.get("token_ids", [])) for choice in choices)


def prepare_requests(
    rendered: dict[str, dict[str, Any]], model: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = copy.deepcopy(rendered["text-smoke"])
    prompt_tokens = text["token_ids"]
    repeat_count = max(1, 3000 // len(prompt_tokens))
    text["token_ids"] = prompt_tokens * repeat_count
    text["model"] = model
    text["stream"] = False
    text["sampling_params"].update(
        {
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "n": 1,
            "max_tokens": 32,
            "ignore_eos": True,
        }
    )

    vlm = copy.deepcopy(rendered["vlm-smoke"])
    vlm["model"] = model
    vlm["stream"] = False
    vlm["sampling_params"].update(
        {
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "n": 1,
            "max_tokens": 16,
            "ignore_eos": True,
        }
    )
    return text, vlm


def observe_live_gauges(
    request: dict[str, Any], before: MetricSnapshot, output_dir: Path, timeout: int
) -> parity.HttpResult:
    model = request["model"]
    baseline_active = metric_value_or_zero(
        before, "dynamo_frontend_active_requests", {"model": model}
    )
    baseline_inflight = metric_value_or_zero(
        before, "dynamo_frontend_inflight_requests", {"model": model}
    )
    baseline_queue = metric_value_or_zero(
        before, "dynamo_frontend_queued_requests", {"model": model}
    )
    active_observed = False
    inflight_observed = False
    queue_observed = False
    live_text = ""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            parity.post_json,
            parity.DYNAMO_PORT,
            parity.GENERATE_PATH,
            request,
            timeout,
        )
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            live_text, live = scrape_metrics(parity.DYNAMO_PORT)
            active = metric_value_or_zero(
                live, "dynamo_frontend_active_requests", {"model": model}
            )
            inflight = metric_value_or_zero(
                live, "dynamo_frontend_inflight_requests", {"model": model}
            )
            queued = metric_value_or_zero(
                live, "dynamo_frontend_queued_requests", {"model": model}
            )
            active_observed |= active > baseline_active
            inflight_observed |= inflight > baseline_inflight
            queue_observed |= queued > baseline_queue
            if active_observed and inflight_observed and queue_observed:
                break
            time.sleep(0.005)
        result = future.result(timeout=timeout)

    (output_dir / "metrics-live.prom").write_text(live_text, encoding="utf-8")
    if not active_observed:
        raise AssertionError("active_requests never rose while Generate was running")
    if not inflight_observed:
        raise AssertionError("inflight_requests never rose while Generate was running")
    if not queue_observed:
        raise AssertionError("queued_requests never rose before the first output token")
    return result


def assert_metric_deltas(
    before: MetricSnapshot,
    after: MetricSnapshot,
    requests_sent: list[dict[str, Any]],
    results: list[parity.HttpResult],
    topology: str,
    model: str,
) -> dict[str, float]:
    request_count = len(requests_sent)
    expected_input_tokens = sum(len(request["token_ids"]) for request in requests_sent)
    expected_output_tokens = sum(output_tokens(result) for result in results)
    model_label = {"model": model}
    generate_started = {**model_label, "endpoint": "generate", "request_type": "unary"}
    generate_success = {
        **generate_started,
        "status": "success",
        "error_type": "",
    }

    deltas = {
        "requests_started": metric_delta(
            before,
            after,
            "dynamo_frontend_requests_started_total",
            generate_started,
        ),
        "requests_succeeded": metric_delta(
            before, after, "dynamo_frontend_requests_total", generate_success
        ),
        "request_duration_count": metric_delta(
            before,
            after,
            "dynamo_frontend_request_duration_seconds_count",
            model_label,
        ),
        "input_sequence_count": metric_delta(
            before,
            after,
            "dynamo_frontend_input_sequence_tokens_count",
            model_label,
        ),
        "input_sequence_sum": metric_delta(
            before,
            after,
            "dynamo_frontend_input_sequence_tokens_sum",
            model_label,
        ),
        "output_sequence_count": metric_delta(
            before,
            after,
            "dynamo_frontend_output_sequence_tokens_count",
            model_label,
        ),
        "output_sequence_sum": metric_delta(
            before,
            after,
            "dynamo_frontend_output_sequence_tokens_sum",
            model_label,
        ),
        "output_tokens": metric_delta(
            before, after, "dynamo_frontend_output_tokens_total", model_label
        ),
        "ttft_count": metric_delta(
            before,
            after,
            "dynamo_frontend_time_to_first_token_seconds_count",
            model_label,
        ),
        "itl_count": metric_delta(
            before,
            after,
            "dynamo_frontend_inter_token_latency_seconds_count",
            model_label,
        ),
        "cached_count": metric_delta(
            before, after, "dynamo_frontend_cached_tokens_count", model_label
        ),
        "cached_sum": metric_delta(
            before, after, "dynamo_frontend_cached_tokens_sum", model_label
        ),
    }

    exact = {
        "requests_started": request_count,
        "requests_succeeded": request_count,
        "request_duration_count": request_count,
        "input_sequence_count": request_count,
        "input_sequence_sum": expected_input_tokens,
        "output_sequence_count": request_count,
        "output_sequence_sum": expected_output_tokens,
        "output_tokens": expected_output_tokens,
        "ttft_count": request_count,
    }
    for name, expected in exact.items():
        if deltas[name] != float(expected):
            raise AssertionError(
                f"{name}: expected delta {expected}, got {deltas[name]}"
            )
    if deltas["itl_count"] <= 0:
        raise AssertionError("inter-token latency histogram was not populated")
    if deltas["cached_count"] <= 0 or deltas["cached_sum"] <= 0:
        raise AssertionError(
            "cached-token histogram did not record the repeated prompt cache hit"
        )

    for gauge in (
        "dynamo_frontend_inflight_requests",
        "dynamo_frontend_active_requests",
        "dynamo_frontend_queued_requests",
    ):
        if metric_value_or_zero(after, gauge, model_label) != metric_value_or_zero(
            before, gauge, model_label
        ):
            raise AssertionError(f"{gauge} did not return to its baseline")

    tokenizer_before = sum(
        matching_values(before, "dynamo_frontend_tokenizer_latency_ms_count", {})
    )
    tokenizer_after = sum(
        matching_values(after, "dynamo_frontend_tokenizer_latency_ms_count", {})
    )
    if tokenizer_after != tokenizer_before:
        raise AssertionError("Generate incorrectly reported frontend tokenizer latency")

    if topology == "disaggregated":
        worker_ttft = matching_values(
            after,
            "dynamo_frontend_worker_last_time_to_first_token_seconds",
            {"worker_type": "prefill"},
        )
        worker_input = matching_values(
            after,
            "dynamo_frontend_worker_last_input_sequence_tokens",
            {"worker_type": "prefill"},
        )
        worker_itl = matching_values(
            after,
            "dynamo_frontend_worker_last_inter_token_latency_seconds",
            {"worker_type": "decode"},
        )
        if not worker_ttft or not worker_input or not worker_itl:
            raise AssertionError(
                f"missing P/D worker timing series: ttft={len(worker_ttft)} "
                f"input={len(worker_input)} itl={len(worker_itl)}"
            )
    return deltas


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        parity.REPO_ROOT
        / "logs"
        / "tito-generate-metrics"
        / f"{timestamp}-{args.topology}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_timeout = max(180, args.startup_timeout)
    environment = parity.base_environment()
    if args.topology == "disaggregated":
        environment["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"

    if args.rendered_requests is not None:
        rendered = json.loads(args.rendered_requests.read_text(encoding="utf-8"))
    else:
        with parity.run_server(
            "upstream-vllm-renderer",
            parity.upstream_command(args.model, args.topology),
            environment,
            output_dir / "upstream-renderer.log",
            parity.UPSTREAM_PORT,
            args.startup_timeout,
            args.model,
        ):
            rendered = parity.render_cases(
                parity.smoke_cases(), args.model, "metrics", request_timeout
            )
    rendered_models = {
        request.get("model")
        for request in rendered.values()
        if isinstance(request, dict)
    }
    if rendered_models != {args.model}:
        raise ValueError(
            f"rendered request models {rendered_models!r} do not match {args.model}"
        )

    (output_dir / "rendered-requests.json").write_text(
        json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text_request, vlm_request = prepare_requests(rendered, args.model)
    engine_config = parity.write_engine_config(output_dir, args.model, args.topology)
    config = json.loads(engine_config.read_text(encoding="utf-8"))
    config["enable_prefix_caching"] = True
    engine_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    ready_file = output_dir / "dynamo-topology.ready"

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
    if args.topology == "disaggregated":
        # The KV router records the selected prefill/decode workers on the
        # request tracker, enabling the per-worker timing metric assertions.
        dynamo_environment["DYN_TITO_ROUTER_MODE"] = "kv"
    with parity.run_server(
        "dynamo-vllm",
        parity.dynamo_command(
            args.model,
            engine_config,
            args.topology,
            args.startup_timeout,
            ready_file,
        ),
        dynamo_environment,
        output_dir / "dynamo.log",
        parity.DYNAMO_PORT,
        args.startup_timeout,
        args.model,
        ready_file if args.topology == "disaggregated" else None,
    ):
        before_text, before = scrape_metrics(parity.DYNAMO_PORT)
        first_text = copy.deepcopy(text_request)
        first_text["request_id"] = f"metrics-{args.topology}-text-first"
        first_result = observe_live_gauges(
            first_text, before, output_dir, request_timeout
        )

        cached_text = copy.deepcopy(text_request)
        cached_text["request_id"] = f"metrics-{args.topology}-text-cached"
        cached_result = parity.post_json(
            parity.DYNAMO_PORT,
            parity.GENERATE_PATH,
            cached_text,
            request_timeout,
        )

        vlm_request["request_id"] = f"metrics-{args.topology}-vlm"
        vlm_result = parity.post_json(
            parity.DYNAMO_PORT,
            parity.GENERATE_PATH,
            vlm_request,
            request_timeout,
        )
        after_text, after = scrape_metrics(parity.DYNAMO_PORT)

    requests_sent = [first_text, cached_text, vlm_request]
    results = [first_result, cached_result, vlm_result]
    (output_dir / "metrics-before.prom").write_text(before_text, encoding="utf-8")
    (output_dir / "metrics-after.prom").write_text(after_text, encoding="utf-8")
    (output_dir / "requests.json").write_text(
        json.dumps(requests_sent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "responses.json").write_text(
        json.dumps(
            [{"status": result.status, "body": result.body} for result in results],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    deltas = assert_metric_deltas(
        before, after, requests_sent, results, args.topology, args.model
    )
    report = {
        "model": args.model,
        "topology": args.topology,
        "requests": len(results),
        "metric_deltas": deltas,
        "status": "passed",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Generate metrics E2E passed: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
