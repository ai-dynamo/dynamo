# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential HTTP contract tests against the installed native vLLM server.

Both servers use the same model revision, engine settings, and Python environment.
Raw responses are retained even on assertion failure. This is aggregated, current
version coverage, not evidence for release skew or disaggregated deployments.
"""

import gzip
import json
import math
import os
import sys
import uuid
from importlib.metadata import version
from pathlib import Path

import pytest
import requests

from tests.utils.http_checks import models_available
from tests.utils.managed_process import ManagedProcess

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
KV_BYTES = 268435456
# config.json at REVISION; full-vocabulary checks must not accept top-k output.
VOCAB_SIZE = 151936

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.model(MODEL),
    pytest.mark.requested_vllm_kv_cache_bytes(KV_BYTES),
    # Measured on both processor variants with profile_pytest.py, 256 MiB KV.
    pytest.mark.profiled_vram_gib(2.7),
    pytest.mark.timeout(900),
]


def _cases():
    for endpoint in ("chat/completions", "completions"):
        base = {"model": MODEL, "temperature": 0, "max_tokens": 3, "seed": 42}
        if endpoint == "chat/completions":
            base["messages"] = [{"role": "user", "content": "Say hello."}]
        else:
            base["prompt"] = "Say hello."
        for count in (1, 2):
            for stream in (False, True):
                for return_ids in (False, True):
                    payload = dict(
                        base,
                        n=count,
                        stream=stream,
                        logprobs=True if endpoint == "chat/completions" else 1,
                        return_tokens_as_token_ids=return_ids,
                    )
                    if endpoint == "chat/completions":
                        payload["top_logprobs"] = 1
                    if count > 1:
                        # Native vLLM rejects n>1 with greedy temperature=0.
                        # top_k=1 keeps this legal sampling case deterministic.
                        payload.update(temperature=0.7, top_k=1)
                    yield (
                        f"{endpoint}-choices-{count}-ids-{return_ids}-stream-{stream}",
                        endpoint,
                        payload,
                    )
        for stream in (False, True):
            for value in (None, -2, -1, 0, 1):
                yield (
                    f"{endpoint}-prompt-{value}-stream-{stream}",
                    endpoint,
                    dict(base, prompt_logprobs=value, stream=stream),
                )
        for value in (None, [], [0]):
            yield (
                f"{endpoint}-allowed-{value}",
                endpoint,
                dict(base, allowed_token_ids=value),
            )
            yield (
                f"{endpoint}-logprob-selection-{value}",
                endpoint,
                dict(
                    base,
                    logprobs=True if endpoint == "chat/completions" else 0,
                    logprob_token_ids=value,
                ),
            )
        if endpoint == "chat/completions":
            # Keep omission distinct from explicit null: native Pydantic
            # defaults do not apply to an explicitly supplied null value.
            for top in ("omitted", None, 0, 1):
                for selection in (None, [], [0]):
                    for stream in (False, True):
                        payload = dict(
                            base,
                            logprobs=True,
                            logprob_token_ids=selection,
                            stream=stream,
                        )
                        if top != "omitted":
                            payload["top_logprobs"] = top
                        yield (
                            f"{endpoint}-top-{top}-selection-{selection}-stream-{stream}",
                            endpoint,
                            payload,
                        )
        else:
            for count in (0, 1):
                for selection in (None, [], [0]):
                    for return_ids in (False, True):
                        for stream in (False, True):
                            yield (
                                f"{endpoint}-count-{count}-selection-{selection}-ids-{return_ids}-stream-{stream}",
                                endpoint,
                                dict(
                                    base,
                                    logprobs=count,
                                    logprob_token_ids=selection,
                                    return_tokens_as_token_ids=return_ids,
                                    stream=stream,
                                ),
                            )


def _capture(port, cases, output_path):
    responses = {}
    for name, endpoint, payload in cases:
        # Deliberately retain errors as data: later comparisons must expose
        # differences, without losing subsequent independent probe results.
        with requests.post(
            f"http://127.0.0.1:{port}/v1/{endpoint}", json=payload, timeout=60
        ) as response:
            responses[name] = {
                "request": payload,
                "status": response.status_code,
                "content_type": response.headers.get("content-type"),
                "body": response.text,
            }
        if output_path.suffix == ".gz":
            with gzip.open(output_path, "wt", compresslevel=1) as output:
                json.dump(responses, output)
        else:
            output_path.write_text(json.dumps(responses, indent=2))
    return responses


def _engine_args():
    return [
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--max-model-len",
        "256",
        # Fix batch shape for probability comparisons, including n=2 choices.
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        os.environ.get("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", str(KV_BYTES)),
        "--generation-config",
        "vllm",
    ]


def _prompt_count_cases():
    """Native characterization, not a Dynamo support declaration.

    Native characterization uses a constant two-token server chat template.
    The Dynamo parity suite instead uses the model's real conversation template.
    """
    for endpoint in ("chat/completions", "completions"):
        base = {"model": MODEL, "temperature": 0, "max_tokens": 1}
        if endpoint == "chat/completions":
            base["messages"] = [{"role": "user", "content": "Hello world"}]
        else:
            base["prompt"] = "Hello world"
        for stream in (False, True):
            for value in (None, -2, -1, 0, 1):
                yield (
                    f"{endpoint}-prompt-{value}-stream-{stream}",
                    endpoint,
                    dict(base, prompt_logprobs=value, stream=stream),
                )


def _assert_native_prompt_count(
    record, endpoint, *, max_logprobs, vocab_size, expected_prompt_tokens=2
):
    """Separate HTTP admission, configured engine limits, and output semantics."""
    request = record["request"]
    count, stream = request["prompt_logprobs"], request["stream"]
    rejected = count == -2 or (
        count is not None
        and ((stream and count != 0) or (count == -1 and max_logprobs != -1))
    )
    assert record["status"] == (400 if rejected else 200), (
        endpoint,
        count,
        stream,
        max_logprobs,
        record["status"],
    )
    if rejected:
        # An SSE error after HTTP 200 is not a native pre-stream rejection.
        assert record["content_type"].startswith("application/json")
        body = json.loads(record["body"])
        assert set(body) == {"error"}, body
        error = body["error"]
        assert (
            error.get("code") == 400 and error.get("type") == "BadRequestError"
        ), error
        assert error["param"] == "prompt_logprobs", error
        assert error["message"], error
        return
    assert record["content_type"].startswith(
        "text/event-stream" if stream else "application/json"
    )
    items = _decoded(record)
    payloads = [p for p in _prompt_payloads(items, endpoint) if p is not None]
    if stream or count is None:
        assert not payloads, "unexpected prompt payload in streaming/unrequested output"
        return
    assert len(payloads) == 1
    rows = payloads[0]
    prompt_tokens = items[0]["usage"]["prompt_tokens"]
    if expected_prompt_tokens is not None:
        assert prompt_tokens == expected_prompt_tokens
    assert len(rows) == prompt_tokens and prompt_tokens > 1 and rows[0] is None
    for row in rows[1:]:
        assert isinstance(row, dict)
        if count == -1:
            assert len(row) == vocab_size, (len(row), vocab_size)
            assert set(map(int, row)) == set(range(vocab_size))
        else:
            # Include the actual prompt token in addition to top-k alternatives.
            assert 1 <= len(row) <= count + 1
        for value in row.values():
            assert math.isfinite(value["logprob"])
            assert value["rank"] > 0
            assert isinstance(value["decoded_token"], str)


@pytest.mark.parametrize("max_logprobs", [20, -1])
def test_native_vllm_prompt_count_contract(
    max_logprobs, dynamo_dynamic_ports, predownload_models, tmp_path: Path
):
    """Establish native -1 ground truth without treating equal errors as parity."""
    port = dynamo_dynamic_ports.frontend_port
    (tmp_path / "runtime-identity.json").write_text(
        json.dumps(
            {
                "scope": "native_only_prompt_count_characterization",
                "vllm_version": version("vllm"),
                "model": MODEL,
                "model_revision": REVISION,
                "max_logprobs": max_logprobs,
                "vocab_size": VOCAB_SIZE,
                "chat_template": "{{ 'Hello world' }}",
            },
            indent=2,
        )
    )
    cases = list(_prompt_count_cases())
    with ManagedProcess(
        command=[
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--port",
            str(port),
            *_engine_args(),
            "--max-logprobs",
            str(max_logprobs),
            "--chat-template",
            "{{ 'Hello world' }}",
        ],
        env=os.environ.copy(),
        health_check_urls=[(f"http://127.0.0.1:{port}/v1/models", models_available)],
        timeout=300,
        terminate_all_matching_process_names=False,
        display_name="native-vllm-prompt-count",
        log_dir=str(tmp_path / "native"),
    ):
        records = _capture(port, cases, tmp_path / "native-responses.json.gz")
    # Finish capture and tear down the server before asserting any case.
    for name, endpoint, _ in cases:
        _assert_native_prompt_count(
            records[name], endpoint, max_logprobs=max_logprobs, vocab_size=VOCAB_SIZE
        )


def _capture_catalog(port, output_path):
    """Retain diagnostic responses before any assertions, including HTTP errors."""
    records = {}
    for name, path in (
        ("catalog", f"/v1/models/{MODEL}/compatibility"),
        ("openapi", "/openapi.json"),
    ):
        with requests.get(f"http://127.0.0.1:{port}{path}", timeout=30) as response:
            records[name] = {
                "status": response.status_code,
                "body": response.text,
            }
        output_path.write_text(json.dumps(records, indent=2))
    return records


def _assert_registered_catalog(
    catalog, processor, engine_version, upstream_commit, *, full_vocab_admitted=False
):
    """Check pipeline facts, not an assertion of complete native conformance."""
    assert catalog["schema_version"] == 1
    assert catalog["scope"] == "registered_pipeline_admission"
    assert catalog["model"] == MODEL
    assert catalog["coverage_complete"] is False
    assert catalog["unlisted_fields"] == "not_catalogued"
    assert catalog["end_to_end_conformance"] == "unverified"
    profiles = catalog["profiles"]
    assert len(profiles) == 2, profiles
    expected_processors = {
        "/v1/chat/completions": "rust" if processor == "dynamo" else "vllm",
        # Chat's Python factory must not change the Rust completion processor.
        "/v1/completions": "rust",
    }
    assert {item["endpoint"] for item in profiles} == set(expected_processors)
    for entry in profiles:
        assert entry["full_vocab_prompt_logprobs_unary_admitted"] is full_vocab_admitted
        admission = entry["admission"]
        assert admission["descriptor_version"] == 1
        assert admission["endpoint"] == entry["endpoint"]
        assert admission["target"] == (
            f"vllm/{engine_version}" if upstream_commit else "vllm/unverified"
        )
        assert admission["upstream_commit"] == upstream_commit
        assert admission["prompt_logprobs_admission"] == (
            "reject_positive_streaming" if upstream_commit else "unverified_target"
        )
        assert admission["pipeline"] == {
            "processor": expected_processors[entry["endpoint"]],
            "transport": "preprocessed_rpc",
            "transport_protocol_version": None,
            "deployment": "aggregated",
        }
        rules = entry["sampling_fields"]
        assert len(rules) == 3
        assert {rule["field"] for rule in rules} == {
            "allowed_token_ids",
            "bad_words_token_ids",
            "logprob_token_ids",
        }
        assert all(rule["request_location"] == "root" for rule in rules)
        assert all(rule["transport"] == "v1_with_legacy_copy" for rule in rules)


def _decoded(record):
    if record["content_type"].startswith("text/event-stream"):
        data = [
            line.removeprefix("data: ")
            for line in record["body"].splitlines()
            if line.startswith("data: ")
        ]
        assert data and data[-1] == "[DONE]", record
        chunks = [json.loads(item) for item in data[:-1]]
        assert chunks and all("error" not in chunk for chunk in chunks), record
        return chunks
    return [json.loads(record["body"])]


def _prompt_payloads(records, endpoint):
    if endpoint == "chat/completions":
        return [item.get("prompt_logprobs") for item in records]
    return [
        choice.get("prompt_logprobs") for item in records for choice in item["choices"]
    ]


def _matches_native(expected, actual):
    """Compare declared native fields, permitting additive Dynamo metadata.

    Float tolerance allows the Rust response's f32 conversion, but structural
    differences (such as a token-object list instead of a map) remain failures.
    """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_native(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_matches_native(a, b) for a, b in zip(expected, actual))
        )
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isclose(
            expected, actual, rel_tol=1e-5, abs_tol=1e-6
        )
    return expected == actual


def _generated_logprobs(records, endpoint, *, stream=False):
    # Choice order and SSE interleaving may differ, but choice identity must
    # survive normalization. Never concatenate probabilities across choices.
    by_choice = {}
    for record in records:
        seen = set()
        for choice in record["choices"]:
            index = choice["index"]
            assert type(index) is int and index >= 0, choice
            assert index not in seen, record
            seen.add(index)
            assert stream or index not in by_choice, records
            by_choice.setdefault(index, []).append({"choices": [choice]})
    return {
        index: _choice_logprobs(chunks, endpoint, stream=stream)
        for index, chunks in by_choice.items()
    }


def _choice_logprobs(records, endpoint, *, stream=False):
    # SSE chunk boundaries are not a compatibility promise. Compare the
    # ordered per-token payload across all chunks, including the final one.
    if endpoint == "chat/completions":
        return [
            token
            for record in records
            for choice in record["choices"]
            for token in (choice.get("logprobs") or {}).get("content", []) or []
        ]
    # Native vLLM offsets are chunk-dependent for token-ID placeholders:
    # each chunk starts at previously emitted *text* length, then advances by
    # returned token-string lengths inside that chunk. Validate this rule for
    # each server independently; comparing flattened offsets would incorrectly
    # require identical SSE chunk boundaries. Unary uses one token-string span.
    previous_text_lengths = {}
    for record in records:
        for choice in record["choices"]:
            index = choice["index"]
            logprobs = choice.get("logprobs")
            if logprobs is not None:
                tokens = logprobs["tokens"]
                assert all(
                    len(logprobs[field]) == len(tokens)
                    for field in ("text_offset", "token_logprobs", "top_logprobs")
                ), logprobs
                offset = previous_text_lengths.get(index, 0) if stream else 0
                expected_offsets = []
                for token in tokens:
                    expected_offsets.append(offset)
                    offset += len(token)
                assert logprobs["text_offset"] == expected_offsets, logprobs
            previous_text_lengths[index] = previous_text_lengths.get(index, 0) + len(
                choice["text"]
            )
    return {
        field: [
            value
            for record in records
            for choice in record["choices"]
            for value in (choice.get("logprobs") or {}).get(field, [])
        ]
        for field in ("tokens", "token_logprobs", "top_logprobs")
    }


@pytest.mark.parametrize("processor", ["dynamo", "vllm"])
@pytest.mark.parametrize("full_vocab", [False, True])
def test_native_vllm_http_contract(
    processor,
    full_vocab,
    dynamo_dynamic_ports,
    file_storage_backend,
    predownload_models,
    tmp_path: Path,
):
    ports = dynamo_dynamic_ports
    port = ports.frontend_port
    engine_version = version("vllm")
    pins = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "lib/llm/src/protocols/openai/compatibility/vllm_pins.json"
        ).read_text()
    )
    upstream_commit = next(
        (
            item["commit"]
            for item in pins["versions"]
            if item["version"] == engine_version
        ),
        None,
    )
    (tmp_path / "runtime-identity.json").write_text(
        json.dumps(
            {
                "vllm_version": engine_version,
                "inspected_upstream_commit": upstream_commit,
                "chat_processor": processor,
                "model_revision": REVISION,
            },
            indent=2,
        )
    )
    env = os.environ.copy()
    env["DYN_SYSTEM_PORT"] = str(ports.system_ports[0])
    env["DYN_FORWARDPASS_METRIC_PORT"] = str(ports.fpm_port)
    env["DYN_NAMESPACE"] = f"protocol-{uuid.uuid4().hex}"
    engine_args = _engine_args()
    if full_vocab:
        engine_args.extend(["--max-logprobs", "-1"])
    runtime_args = [
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    ready = [(f"http://127.0.0.1:{port}/v1/models", models_available)]
    cases = list(_prompt_count_cases() if full_vocab else _cases())
    response_suffix = ".json.gz" if full_vocab else ".json"
    with ManagedProcess(
        command=[
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--port",
            str(port),
            *engine_args,
        ],
        env=env,
        health_check_urls=ready,
        timeout=300,
        terminate_all_matching_process_names=False,
        display_name="native-vllm",
        log_dir=str(tmp_path / "native"),
    ):
        native = _capture(port, cases, tmp_path / f"native-responses{response_suffix}")

    with (
        ManagedProcess(
            command=[
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(port),
                "--dyn-chat-processor",
                processor,
                *runtime_args,
            ],
            env=env,
            health_check_ports=[port],
            timeout=120,
            terminate_all_matching_process_names=False,
            display_name="dynamo-frontend",
            log_dir=str(tmp_path / "frontend"),
        ),
        ManagedProcess(
            command=[
                sys.executable,
                "-m",
                "dynamo.vllm",
                *engine_args,
                *runtime_args,
                "--kv-events-config",
                '{"enable_kv_cache_events": false}',
            ],
            env=env,
            health_check_urls=ready,
            timeout=300,
            terminate_all_matching_process_names=False,
            display_name="dynamo-worker",
            log_dir=str(tmp_path / "worker"),
        ),
    ):
        diagnostic = _capture_catalog(port, tmp_path / "dynamo-catalog.json")
        actual = _capture(port, cases, tmp_path / f"dynamo-responses{response_suffix}")

    assert diagnostic["catalog"]["status"] == 200, diagnostic["catalog"]
    catalog = json.loads(diagnostic["catalog"]["body"])
    _assert_registered_catalog(
        catalog,
        processor,
        engine_version,
        upstream_commit,
        full_vocab_admitted=full_vocab,
    )
    assert env["DYN_NAMESPACE"] not in diagnostic["catalog"]["body"]
    assert diagnostic["openapi"]["status"] == 200, diagnostic["openapi"]
    spec = json.loads(diagnostic["openapi"]["body"])
    operation = spec["paths"]["/v1/models/{model_id}/compatibility"]["get"]
    assert operation["parameters"][0]["name"] == "model_id"
    assert operation["parameters"][0]["required"] is True
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ModelCompatibilityCatalog"
    }

    mismatches = []
    for name, endpoint, payload in cases:
        expected, got = native[name], actual[name]
        if full_vocab:
            for record in (expected, got):
                _assert_native_prompt_count(
                    record,
                    endpoint,
                    max_logprobs=-1,
                    vocab_size=VOCAB_SIZE,
                    expected_prompt_tokens=None,
                )
        if "-choices-" in name and expected["status"] != 200:
            mismatches.append(f"{name}: native positive-control request rejected")
            continue
        if got["status"] != expected["status"]:
            mismatches.append(f"{name}: status {got['status']} != {expected['status']}")
            continue
        if expected["status"] != 200:
            if not got["content_type"].startswith("application/json"):
                mismatches.append(f"{name}: rejection was not pre-stream JSON")
            continue
        try:
            expected_items, got_items = _decoded(expected), _decoded(got)
        except (AssertionError, json.JSONDecodeError) as error:
            mismatches.append(f"{name}: invalid response or streaming error: {error}")
            continue
        expected_prompts = [
            item
            for item in _prompt_payloads(expected_items, endpoint)
            if item is not None
        ]
        got_prompts = [
            item for item in _prompt_payloads(got_items, endpoint) if item is not None
        ]
        # Include decoded_token and probabilities, not only token IDs/ranks:
        # a token-only worker may require frontend detokenization of the payload.
        if not _matches_native(expected_prompts, got_prompts):
            mismatches.append(f"{name}: prompt-logprob placement or payload differs")
        if "logprobs" in payload:
            try:
                expected_logprobs = _generated_logprobs(
                    expected_items, endpoint, stream=payload.get("stream", False)
                )
                got_logprobs = _generated_logprobs(
                    got_items, endpoint, stream=payload.get("stream", False)
                )
                expected_indices = set(range(payload.get("n", 1)))
                assert set(expected_logprobs) == expected_indices, expected_logprobs
                assert set(got_logprobs) == expected_indices, got_logprobs
            except AssertionError as error:
                mismatches.append(
                    f"{name}: invalid logprob offsets or lengths: {error}"
                )
                continue
            if not _matches_native(expected_logprobs, got_logprobs):
                mismatches.append(f"{name}: generated-logprob shape or values differ")
        if payload.get("allowed_token_ids") == [0]:
            choice = got_items[0]["choices"][0]
            text = (
                choice["message"]["content"]
                if endpoint == "chat/completions"
                else choice["text"]
            )
            if text != "!!!":
                mismatches.append(f"{name}: allowed token directive lost: {text!r}")
    assert not mismatches, "\n".join(mismatches)
