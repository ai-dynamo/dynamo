# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP regressions for vLLM output constraints on both endpoints.

Run the same constraints on direct vLLM and Dynamo backed by that installed vLLM.
The constrained answer differs from the prompt, so HTTP 200 or ordinary model
obedience cannot pass as constraint enforcement. Raw requests/responses and launch
arguments stay in tmp_path for diagnosis and external evidence archival. Use
``-o tmp_path_retention_policy=all`` to retain evidence from passing tests too.

Per-request whitespace enforcement is deliberately not asserted: the pinned
xgrammar implementation uses its server configuration (vLLM issue #48765).
"""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import pytest
import requests
from transformers import AutoTokenizer

from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(3.4),
    pytest.mark.post_merge,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.requested_vllm_kv_cache_bytes(1_073_741_824),
    pytest.mark.timeout(235),
]


@pytest.fixture(params=["vllm", "dynamo"])
def structured_server(request, tmp_path):
    """Yield one real server URL and evidence directory; stop only owned processes.

    Both variants use the same model revision, engine limits, qwen3 reasoner and
    hermes tool parser. Automatic tool choice is enabled for composition probes.
    File discovery
    isolates Dynamo without shared etcd/NATS; reserved ports avoid fixed listeners.
    Startup errors propagate through ManagedProcess's cleanup. KV allocation honors
    the repository profiler override; no process changes CUDA device selection.
    """
    source = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS": "0",
    }
    engine = [
        "--model",
        "Qwen/Qwen3-0.6B",
        "--revision",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "--served-model-name",
        "structured-output-test",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "2",
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        environment.get("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", "1073741824"),
    ]
    with reserved_ports(2, 19000) as ports, ExitStack() as stack:
        http_port, worker_port = ports
        base = f"http://127.0.0.1:{http_port}"
        commands = []
        common = {
            "working_dir": str(source),
            "log_dir": str(tmp_path),
            "terminate_all_matching_process_names": False,
            "timeout": 300,
        }
        if request.param == "vllm":
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                *engine,
                "--host",
                "127.0.0.1",
                "--port",
                str(http_port),
                "--reasoning-parser",
                "qwen3",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="direct-vllm",
                    health_check_urls=[base + "/health"],
                    **common,
                )
            )
        else:
            namespace = "structured-" + uuid4().hex
            environment.update(
                {
                    "DYN_DISCOVERY_BACKEND": "file",
                    "DYN_FILE_KV": str(tmp_path / "discovery"),
                    "DYN_REQUEST_PLANE": "tcp",
                    "DYN_EVENT_PLANE": "zmq",
                    "DYN_NAMESPACE": namespace,
                    "DYN_SYSTEM_PORT": str(worker_port),
                }
            )
            command = [
                sys.executable,
                "-m",
                "dynamo.vllm",
                *engine,
                "--dyn-reasoning-parser",
                "qwen3",
                "--dyn-tool-call-parser",
                "hermes",
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="dynamo-worker",
                    health_check_urls=[f"http://127.0.0.1:{worker_port}/health"],
                    **common,
                )
            )
            command = [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(http_port),
                "--namespace",
                namespace,
            ]
            commands.append(command)
            stack.enter_context(
                ManagedProcess(
                    command=command,
                    env=environment,
                    display_name="dynamo-frontend",
                    health_check_urls=[
                        (
                            base + "/v1/models",
                            lambda response: any(
                                model["id"] == "structured-output-test"
                                for model in response.json().get("data", [])
                            ),
                        )
                    ],
                    **common,
                )
            )
        (tmp_path / "launch.json").write_text(json.dumps(commands, indent=2))
        with requests.get(base + "/openapi.json", timeout=30) as response:
            response.raise_for_status()
            (tmp_path / "openapi.json").write_bytes(response.content)
        yield base, tmp_path


def _generate(
    base, path, payload, evidence, expected_status=200, expected_finish="stop"
):
    """Save raw HTTP evidence and return content, or None for an expected HTTP error.

    Streaming must end with [DONE] and expected_finish (normally stop; use length
    when deliberately exhausting max_tokens). Malformed chunks,
    error events, truncation and non-200 statuses fail rather than being normalized
    away. Only content fragments are assembled, not IDs, timestamps or tool calls.
    For pre-header rejection checks, pass expected_status=400: require JSON rather
    than treating HTTP 200 with an SSE error as equivalent to an HTTP error.
    """
    evidence.mkdir()
    (evidence / "request.json").write_text(json.dumps(payload, indent=2))
    with requests.post(base + path, json=payload, timeout=90) as response:
        (evidence / "response.body").write_bytes(response.content)
        (evidence / "response.json").write_text(
            json.dumps(
                {"status": response.status_code, "headers": dict(response.headers)},
                indent=2,
            )
        )
        assert response.status_code == expected_status, response.text
        if expected_status != 200:
            assert response.headers["Content-Type"].startswith("application/json")
            return None
        chat = path == "/v1/chat/completions"
        if not payload["stream"]:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == expected_finish, choice
            return choice["message"]["content"] if chat else choice["text"]
        assert response.headers["Content-Type"].startswith("text/event-stream")
        events = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events and events[-1] == "[DONE]", response.text
        chunks = [json.loads(event) for event in events[:-1]]
        assert all("error" not in chunk for chunk in chunks), chunks
        choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
        assert [
            choice["finish_reason"]
            for choice in choices
            if choice.get("finish_reason") is not None
        ] == [expected_finish]
        return "".join(
            (
                (choice.get("delta", {}).get("content") or "")
                if chat
                else choice.get("text", "")
            )
            for choice in choices
        )


def test_structured_outputs_constrain_chat_and_completion(structured_server):
    """Guard shared admission and semantic delivery, in unary and streaming modes.

    The parent Dynamo returns 400 for the first nested constraint; unchanged direct
    vLLM must satisfy every case. Completion uses an already-finished reasoning
    prefix, because this reasoner activates constraints only after </think>.
    Whitespace inside valid JSON is allowed; exact string constraints are not trimmed.
    """
    base, evidence = structured_server
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "violet"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    constraints = [
        ("json", {"json": schema}),
        ("json-string", {"json": json.dumps(schema)}),
        ("json-object", {"json_object": True}),
        (
            "structural-tag",
            {
                "structural_tag": json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {"type": "json_schema", "json_schema": schema},
                    }
                )
            },
        ),
        ("choice", {"choice": ["violet"]}),
        ("regex", {"regex": "violet"}),
        ("grammar", {"grammar": 'root ::= "violet"'}),
    ]
    for endpoint in ("chat", "completion"):
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        if endpoint == "chat":
            payload.update(
                messages=[{"role": "user", "content": "Answer with the value seven."}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                "<|im_start|>user\nAnswer with the value seven.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = _generate(base, path, payload, evidence / (endpoint + "-baseline"))
        assert (
            "violet" not in baseline
        ), "The control must distinguish constrained output"
        for stream in (False, True):
            for name, constraint in constraints:
                result = _generate(
                    base,
                    path,
                    {**payload, "stream": stream, "structured_outputs": constraint},
                    evidence / f"{endpoint}-{name}-{stream}",
                )
                if name == "json-object":
                    assert isinstance(json.loads(result), dict), result
                elif name in ("json", "json-string", "structural-tag"):
                    assert json.loads(result) == {"answer": "violet"}, result
                else:
                    assert result == "violet", result
        if endpoint == "chat":
            # response_format replaces a same-kind constraint before its content
            # is validated. Text format instead leaves the explicit regex intact.
            overrides = [
                (
                    "replace-json",
                    {"json": "{"},
                    {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": schema},
                    },
                ),
                ("replace-object", {"json_object": False}, {"type": "json_object"}),
                ("text-regex", {"regex": "violet"}, {"type": "text"}),
            ]
            for stream in (False, True):
                for name, constraint, response_format in overrides:
                    result = _generate(
                        base,
                        path,
                        {
                            **payload,
                            "stream": stream,
                            "structured_outputs": constraint,
                            "response_format": response_format,
                        },
                        evidence / f"chat-{name}-{stream}",
                    )
                    if name == "replace-json":
                        assert json.loads(result) == {"answer": "violet"}, result
                    elif name == "replace-object":
                        assert isinstance(json.loads(result), dict), result
                    else:
                        assert result == "violet", result


def _root_property(document, node, name):
    """Find a request-root property through these exports' acyclic local refs/allOf.

    Return None for an absent property. Do not descend into arbitrary properties:
    a same-named field inside messages or tools does not describe this request root.
    Broken local refs raise KeyError instead of becoming evidence of absence.
    """
    if name in node.get("properties", {}):
        return node["properties"][name]
    if "$ref" in node:
        reference = node["$ref"]
        assert reference.startswith("#/"), reference
        resolved = document
        for part in reference[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        return _root_property(document, resolved, name)
    for part in node.get("allOf", []):
        found = _root_property(document, part, name)
        if found is not None:
            return found
    return None


def test_token_controls_are_effective_and_declared(structured_server):
    """Separate existing token-control effects from missing OpenAPI declarations.

    First prove restriction and stopping with real inference on both transports;
    then require each admitted root field in that server's live schema. On the
    schema-deficient parent, inference passes but the declaration assertion fails.
    This does not certify count limits, coercion, or error/empty-content parity.
    Immediate stops may be an empty string or unary chat null; preserve raw bodies
    and check only that the stop token is not emitted, not response-shape equality.
    """
    base, evidence = structured_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    spec = json.loads((evidence / "openapi.json").read_text())
    paths = ["/v1/chat/completions", "/v1/completions"]
    for path in paths:
        chat = path == paths[0]
        endpoint = "chat" if chat else "completion"
        instruction = "Reply with exactly the word hello."
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 4,
            "stream": False,
        }
        if chat:
            payload.update(
                messages=[{"role": "user", "content": instruction}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = _generate(base, path, payload, evidence / f"{endpoint}-ids-control")
        assert baseline != "7777", baseline
        for stream in (False, True):
            restricted = {**payload, "stream": stream, "allowed_token_ids": digit}
            result = _generate(
                base,
                path,
                restricted,
                evidence / f"{endpoint}-ids-allowed-{stream}",
                expected_finish="length",
            )
            assert result == "7777", result
            result = _generate(
                base,
                path,
                {**restricted, "stop_token_ids": digit},
                evidence / f"{endpoint}-ids-stop-{stream}",
            )
            assert result in ("", None), result
    for path in paths:
        root = spec["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        for name in ("allowed_token_ids", "stop_token_ids"):
            field = _root_property(spec, root, name)
            assert field is not None, f"{path} accepts {name} but does not declare it"
            arms = field.get("anyOf", [field])
            assert any(
                "array" in arm.get("type", [])
                and arm.get("items", {}).get("type") == "integer"
                for arm in arms
            ), field


def test_ignore_eos_preserves_explicit_user_stops(structured_server):
    """Ignore implicit EOS without discarding caller stop strings or token IDs.

    Force digit7 or the pinned tokenizer's EOS, so a lost stop yields four tokens
    and a length finish instead of an immediate stop. An EOS-only control proves
    ignore_eos itself works; an explicitly requested EOS stop must still win.
    Check both endpoints/transports against the unchanged reference. Empty unary
    chat content may be null; raw bodies retain that separate response difference.
    """
    base, evidence = structured_server
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca"
    )
    digit = tokenizer.encode("7", add_special_tokens=False)
    eos = [tokenizer.eos_token_id]
    eos_text = tokenizer.decode(eos, skip_special_tokens=False)
    assert len(digit) == 1 and tokenizer.decode(digit) == "7", digit
    assert eos_text, eos
    instruction = "Reply with exactly the word hello."
    for path, inputs in (
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": instruction}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "/v1/completions",
            {
                "prompt": f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            },
        ),
    ):
        endpoint = "chat" if path == "/v1/chat/completions" else "completion"
        for stream in (False, True):
            payload = {
                "model": "structured-output-test",
                "temperature": 0,
                "seed": 0,
                "max_tokens": 4,
                "stream": stream,
                "ignore_eos": True,
                "skip_special_tokens": False,
                **inputs,
            }
            for name, allowed, stop, expected in (
                ("eos-control", eos, {}, eos_text * 4),
                ("digit-control", digit, {}, "7777"),
                ("user-string", digit, {"stop": "7"}, None),
                ("user-id", digit, {"stop_token_ids": digit}, None),
                ("explicit-eos", eos, {"stop_token_ids": eos}, None),
            ):
                result = _generate(
                    base,
                    path,
                    {**payload, "allowed_token_ids": allowed, **stop},
                    evidence / f"{endpoint}-ignore-eos-{name}-{stream}",
                    expected_finish="stop" if stop else "length",
                )
                if expected is None:
                    assert result in ("", None), result
                else:
                    assert result == expected, result


def test_bad_words_exclude_baseline_words(structured_server):
    """Prove word exclusions reach sampling on both endpoints and transports.

    The control must emit a banned spelling before the constraint is tested;
    otherwise ordinary model output could masquerade as enforcement. Empty lists
    preserve the control and duplicates must not erase the exclusion. Reusing the
    same pinned real-server fixture keeps direct vLLM and Dynamo comparable.
    An empty entry must reject before streaming headers, not open an error stream.
    """
    base, evidence = structured_server
    instruction = "Reply with exactly the word hello."
    for chat in (True, False):
        endpoint = "chat" if chat else "completion"
        path = "/v1/chat/completions" if chat else "/v1/completions"
        payload = {
            "model": "structured-output-test",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 16,
            "stream": False,
        }
        if chat:
            payload.update(
                messages=[{"role": "user", "content": instruction}],
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            payload["prompt"] = (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        baseline = _generate(
            base, path, payload, evidence / f"{endpoint}-words-control"
        )
        excluded = ["Hello", "hello"]
        assert any(word in baseline for word in excluded), baseline
        for stream in (False, True):
            for name, words in [
                ("exclude", excluded),
                ("duplicates", excluded + excluded),
                ("empty", []),
            ]:
                result = _generate(
                    base,
                    path,
                    {**payload, "stream": stream, "bad_words": words},
                    evidence / f"{endpoint}-words-{name}-{stream}",
                )
                if words:
                    assert result and result != baseline, result
                    assert all(word not in result for word in excluded), result
                else:
                    assert result == baseline, result
            _generate(
                base,
                path,
                {**payload, "stream": stream, "bad_words": [""]},
                evidence / f"{endpoint}-words-empty-entry-{stream}",
                expected_status=400,
            )


def test_completion_response_format_controls_output(structured_server):
    """Guard completion-format admission, constraint effects and nested overrides.

    The prompt requests seven while schemas require violet. Legacy structural tags
    constrain only triggered regions; the Markdown fence must actually be emitted
    before its JSON constraint counts as enforced. Null/text retain unconstrained
    generation, and text must not erase a separately supplied regex constraint.
    The shared helper retains raw unary/SSE evidence and rejects error streams.
    """
    base, evidence = structured_server
    path = "/v1/completions"
    payload = {
        "model": "structured-output-test",
        "prompt": (
            "<|im_start|>user\nReturn the answer seven as JSON inside <answer> and </answer>."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }
    control = _generate(base, path, payload, evidence / "format-baseline")
    assert "violet" not in control, control
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "violet"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    json_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }
    tag_format = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema},
    }
    legacy_tag = {
        "type": "structural_tag",
        "structures": [{"begin": "```json\n", "schema": schema, "end": "\n```"}],
        "triggers": ["```json"],
    }
    cases = [
        ("json", json_format, None, "json"),
        ("object", {"type": "json_object"}, None, "object"),
        ("tag", tag_format, None, "json"),
        ("legacy-tag", legacy_tag, None, "tagged-json"),
        ("null", None, None, "unconstrained"),
        ("text", {"type": "text"}, None, "unconstrained"),
        ("json-override", json_format, {"json": "{"}, "json"),
        ("object-override", {"type": "json_object"}, {"json_object": False}, "object"),
        (
            "tag-override",
            tag_format,
            {
                "structural_tag": json.dumps(
                    {
                        "type": "structural_tag",
                        "format": {
                            "type": "json_schema",
                            "json_schema": {"const": "seven"},
                        },
                    }
                )
            },
            "json",
        ),
        ("text-keeps-regex", {"type": "text"}, {"regex": "violet"}, "regex"),
    ]
    for stream in (False, True):
        for name, response_format, nested, expected in cases:
            request = {**payload, "stream": stream, "response_format": response_format}
            if nested is not None:
                request["structured_outputs"] = nested
            result = _generate(
                base, path, request, evidence / f"format-{name}-{stream}"
            )
            if expected == "unconstrained":
                assert result and "violet" not in result, result
            elif expected == "regex":
                assert result == "violet", result
            elif expected == "object":
                assert isinstance(json.loads(result), dict), result
            else:
                if expected == "tagged-json":
                    assert result.startswith("```json\n") and result.endswith(
                        "\n```"
                    ), result
                    result = result[len("```json\n") : -len("\n```")]
                assert json.loads(result) == {"answer": "violet"}, result
