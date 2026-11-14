# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation must establish independent evidence before publishing a corpus."""

from __future__ import annotations

import base64
import copy
import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.frontend import test_trace_capture as capture_test
from tests.frontend.trace_replay import capture as capture_tool
from tests.frontend.trace_replay import prepare
from tests.frontend.trace_replay.contract import (
    ReplayError,
    dumps,
    load_corpus,
    loads,
    write_json,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
]

VENDOR = b"""import json
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    return "prompt:" + json.dumps([messages, thinking_mode, reasoning_effort])
def render_message(index, messages, thinking_mode, reasoning_effort=None):
    assert index == len(messages) - 1
    assert messages[index]["content"] == "OK"
    return "completion:" + thinking_mode
"""


class TinyTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return 1 if token == "<eos>" else None

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        if text.startswith("completion:"):
            return SimpleNamespace(ids=[9, 1] if text.endswith("chat") else [7, 9, 1])
        # A deterministic content-sensitive stand-in; no optional dependency.
        return SimpleNamespace(ids=[2, *hashlib.sha256(text.encode()).digest()[:4]])


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    model = tmp_path / "assets"
    model.mkdir()
    assets = {
        "config.json": {"eos_token_id": 1},
        "tokenizer_config.json": {"eos_token": {"content": "<eos>"}},
        "tokenizer.json": {"test": "verified bytes"},
    }
    for name, value in assets.items():
        write_json(model / name, value)
    (model / "encoding.py").write_bytes(VENDOR)
    profile = {
        "name": "test-v41",
        "model_name": "test-model",
        "model": {
            "repo_id": "vendor/model",
            "revision": "a" * 40,
            "assets": {name: sha((model / name).read_bytes()) for name in assets},
        },
        "runtime_config": {
            "tool_call_parser": "deepseek_v41",
            "reasoning_parser": "deepseek_v41",
            "context_length": 4096,
        },
    }
    reference = {
        "repo_id": "vendor/model",
        "revision": "a" * 40,
        "path": "encoding.py",
        "sha256": sha(VENDOR),
        "tokenizers_version": "0.21.4",
    }
    profile_path = tmp_path / "profile.json"
    write_json(
        profile_path,
        {"schema": prepare.SCHEMA, "profile": profile, "reference": reference},
    )
    seed = {
        "id": "case",
        "source": {"kind": "authored", "context": {}},
        "request": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking": False, "thinking_mode": "chat"},
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "n": 1,
        },
    }
    seeds = tmp_path / "requests.jsonl"
    seeds.write_text(dumps(seed) + "\n")
    monkeypatch.setattr(
        prepare.ReferenceEncoder,
        "_tokenizer",
        staticmethod(lambda data, version: TinyTokenizer()),
    )
    return {
        "model": model,
        "profile": profile,
        "reference": reference,
        "path": profile_path,
        "seed": seed,
        "seeds": seeds,
    }


def capture(
    setup: dict[str, Any],
    root: Path,
    *,
    output: list[int] | None = None,
    finish: str = "tool_calls",
) -> Path:
    root.mkdir()
    seed = setup["seed"]
    seeds_data = (dumps(seed) + "\n").encode()
    (root / "requests.jsonl").write_bytes(seeds_data)
    native = prepare.ReferenceEncoder(
        setup["profile"], setup["reference"], setup["model"]
    )
    prompt, _ = native.prompt(seed["request"], captured=True)
    output = [9, 1] if output is None else output
    response = {
        "prompt_token_ids": prompt,
        "choices": [
            {
                "index": 0,
                "token_ids": output,
                "finish_reason": finish,
                "stop_reason": None,
                "message": {"content": "diagnostic"},
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(output),
            "total_tokens": len(prompt) + len(output),
        },
    }
    response_bytes = dumps(response).encode()
    effective = {
        key: value
        for key, value in seed["request"].items()
        if key not in {"stream", "return_token_ids"}
    }
    wire = dumps({**effective, "return_token_ids": True, "stream": False})
    record = {
        "schema": prepare.RECORD_SCHEMA,
        "id": seed["id"],
        "source": seed["source"],
        "original_request": seed["request"],
        "wire_request_bytes": wire,
        "request_sha256": sha(wire.encode()),
        "http_status": 200,
        "response_bytes_base64": base64.b64encode(response_bytes).decode(),
        "response_sha256": sha(response_bytes),
        "response": response,
        "error": None,
    }
    if "reference" in seed:
        record["reference"] = seed["reference"]
    write_json(root / "record.json", record)
    write_json(
        root / "capture.json",
        {
            "schema": prepare.CAPTURE_SCHEMA,
            "complete": True,
            "count": 1,
            "successful_count": 1,
            "profile": setup["profile"],
            "requests_path": "requests.jsonl",
            "requests_sha256": sha(seeds_data),
            "records": [
                {
                    "id": seed["id"],
                    "path": "record.json",
                    "sha256": sha((root / "record.json").read_bytes()),
                    "status": "captured",
                }
            ],
        },
    )
    return root


def run(setup: dict[str, Any], output: Path, **kwargs: Any) -> dict[str, Any]:
    return prepare.prepare_corpus(setup["path"], setup["model"], output, **kwargs)


def test_reference_only_is_deterministic_and_never_promotes_seed_expectations(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    seed = {
        **setup["seed"],
        "expect": {"response": {"message": "untrusted"}},
        "replay": {"worker_events": "untrusted"},
    }
    setup["seeds"].write_text(dumps(seed) + "\n")
    first, second = tmp_path / "one", tmp_path / "different" / "two"
    run(setup, first, requests=setup["seeds"])
    run(setup, second, requests=setup["seeds"])
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "cases.jsonl").read_bytes() == (second / "cases.jsonl").read_bytes()
    corpus = load_corpus(first)
    assert corpus.cases[0].expected_response is None
    assert corpus.cases[0].reference["preparation"]["completion"]["kind"] == "synthetic"
    assert corpus.cases[0].worker_events[0]["token_ids"] == [9, 1]
    assert str(tmp_path) not in (first / "manifest.json").read_text()


def test_case_publication_is_exclusive_and_private(
    setup: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "racing"
    output.mkdir(mode=0o755)
    original = prepare.ReferenceEncoder.synthetic

    def competitor(
        self: prepare.ReferenceEncoder, request: dict[str, Any]
    ) -> list[int]:
        # Arrive after the output preflight but before its publication point.
        (output / "cases.jsonl").write_bytes(b"other writer owns this file")
        return original(self, request)

    monkeypatch.setattr(prepare.ReferenceEncoder, "synthetic", competitor)
    with pytest.raises(FileExistsError):
        run(setup, output, requests=setup["seeds"])
    assert (output / "cases.jsonl").read_bytes() == b"other writer owns this file"
    assert not (output / "manifest.json").exists()
    monkeypatch.setattr(prepare.ReferenceEncoder, "synthetic", original)
    private = tmp_path / "existing-public-directory"
    private.mkdir(mode=0o755)
    run(setup, private, requests=setup["seeds"])
    if os.name == "posix":
        assert stat.S_IMODE((private / "cases.jsonl").stat().st_mode) == 0o600


def test_captured_response_stays_diagnostic_and_controls_are_explicit(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    root = capture(setup, tmp_path / "capture")
    run(setup, tmp_path / "out", captures=root)
    case = load_corpus(tmp_path / "out").cases[0]
    assert case.expected_response is None
    assert (
        case.reference["preparation"]["capture"]["diagnostic_response"]["message"][
            "content"
        ]
        == "diagnostic"
    )
    assert case.worker_events[0]["finish_reason"] == "stop"
    fields = case.expected_preprocessed["fields_equal"]
    assert fields["/stop_conditions/max_tokens"] == 8
    assert fields["/eos_token_ids"] == [1]
    assert fields["/stop_conditions/stop_token_ids_hidden"] == [1]


@pytest.mark.parametrize("captured", [False, True])
def test_seed_reference_metadata_is_preserved_and_generated_evidence_is_namespaced(
    setup: dict[str, Any], tmp_path: Path, captured: bool
) -> None:
    original = {
        "capture": {"bfcl": {"category": "simple", "source_revision": "frozen"}},
        "completion": {"note": "seed diagnostic"},
        "reference_transformations": ["original metadata"],
    }
    setup["seed"]["reference"] = original
    setup["seeds"].write_text(dumps(setup["seed"]) + "\n")
    inputs = (
        {"captures": capture(setup, tmp_path / "capture")}
        if captured
        else {"requests": setup["seeds"]}
    )
    run(setup, tmp_path / "out", **inputs)
    reference = load_corpus(tmp_path / "out").cases[0].reference
    assert {key: reference[key] for key in original} == original
    assert set(reference) == set(original) | {"preparation"}
    assert ("capture" if captured else "completion") in reference["preparation"]
    setup["seed"]["reference"]["preparation"] = {"caller": "must not be overwritten"}
    setup["seeds"].write_text(dumps(setup["seed"]) + "\n")
    collision_inputs = (
        {"captures": capture(setup, tmp_path / "collision-capture")}
        if captured
        else {"requests": setup["seeds"]}
    )
    with pytest.raises(ReplayError, match="reference.preparation is reserved"):
        run(setup, tmp_path / "collision", **collision_inputs)
    assert not (tmp_path / "collision" / "manifest.json").exists()


def test_capture_to_prepare_preserves_unicode_json_line_and_canonical_wire_identity(
    setup: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "text\u2028line separator\u0085next line\u2029paragraph"
    setup["seed"]["request"]["messages"][0]["content"] = content
    # Selectors already in a seed must also be emitted in the canonical order.
    setup["seed"]["request"] = {
        "stream": False,
        **setup["seed"]["request"],
        "return_token_ids": True,
    }
    setup["seeds"].write_text(dumps(setup["seed"]) + "\n")
    native = prepare.ReferenceEncoder(
        setup["profile"], setup["reference"], setup["model"]
    )
    prompt, _ = native.prompt(setup["seed"]["request"], captured=True)
    response = {
        "prompt_token_ids": prompt,
        "choices": [
            {
                "index": 0,
                "token_ids": [9, 1],
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": 2,
            "total_tokens": len(prompt) + 2,
        },
    }
    monkeypatch.setattr(capture_test, "MODEL", setup["profile"]["model_name"])
    with capture_test.reference_server(body=dumps(response).encode()) as (url, seen):
        result = capture_tool.capture_requests(
            setup["seeds"], setup["path"], url, tmp_path / "http-capture"
        )
    assert result["complete"] is True
    run(setup, tmp_path / "out", captures=tmp_path / "http-capture")
    case = load_corpus(tmp_path / "out").cases[0]
    assert case.request["messages"][0]["content"] == content
    effective = {
        key: value
        for key, value in setup["seed"]["request"].items()
        if key not in {"stream", "return_token_ids"}
    }
    expected_wire = dumps(
        {**effective, "return_token_ids": True, "stream": False}
    ).encode()
    assert case.reference["preparation"]["capture"]["request_sha256"] == sha(
        expected_wire
    )
    assert len(seen["post"]) == 1


@pytest.mark.parametrize(
    "change",
    ["incomplete", "count", "missing", "record_hash", "response_hash", "wire", "seed"],
)
def test_invalid_capture_never_publishes_manifest(
    setup: dict[str, Any], tmp_path: Path, change: str
) -> None:
    root = capture(setup, tmp_path / "capture")
    manifest = loads((root / "capture.json").read_bytes())
    record = loads((root / "record.json").read_bytes())
    if change == "incomplete":
        manifest["complete"] = False
    elif change == "count":
        manifest["successful_count"] = 0
    elif change == "missing":
        manifest["records"] = []
    elif change == "record_hash":
        manifest["records"][0]["sha256"] = "0" * 64
    else:
        if change == "response_hash":
            record["response_sha256"] = "0" * 64
        elif change == "wire":
            record["wire_request_bytes"] = "{}"
            record["request_sha256"] = sha(b"{}")
        elif change == "seed":
            record["original_request"]["temperature"] = 1
        write_json(root / "record.json", record)
        manifest["records"][0]["sha256"] = sha((root / "record.json").read_bytes())
    write_json(root / "capture.json", manifest)
    with pytest.raises(ReplayError):
        run(setup, tmp_path / "out", captures=root)
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("stop", []),
        ("ignore_eos", False),
        ("min_tokens", 0),
        ("reasoning_effort", "high"),
        ("response_format", {}),
        ("parallel_tool_calls", True),
        ("tool_choice", "required"),
        ("chat_template_args", {}),
        ("unknown", None),
        ("max_completion_tokens", 8),
        ("n", True),
        ("stream", True),
    ],
)
def test_closed_request_contract_rejects_ignored_or_ambiguous_fields(
    setup: dict[str, Any], field: str, value: Any
) -> None:
    request = {**setup["seed"]["request"], field: value}
    with pytest.raises(ReplayError):
        prepare.rendering_input(request, captured=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"thinking": False, "thinking_mode": "thinking"},
        {"thinking": 1},
        {"thinking_mode": "chat", "unknown": 1},
    ],
)
def test_thinking_mode_must_be_explicit_consistent_and_supported(
    setup: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ReplayError):
        prepare.rendering_input(
            {**setup["seed"]["request"], "chat_template_kwargs": kwargs}, captured=False
        )


def test_segmented_reasoning_requires_captured_prompt_agreement(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": ["", "first", "second"],
    }
    setup["seed"]["request"]["messages"].insert(0, message)
    request = setup["seed"]["request"]
    with pytest.raises(ReplayError, match="captured prompt evidence"):
        prepare.rendering_input(request, captured=False)
    root = capture(setup, tmp_path / "capture")
    run(setup, tmp_path / "out", captures=root)
    case = load_corpus(tmp_path / "out").cases[0]
    assert case.request["messages"][0]["reasoning_content"] == ["", "first", "second"]
    assert case.reference["preparation"]["reference_transformations"]
    record = loads((root / "record.json").read_bytes())
    record["response"]["prompt_token_ids"][0] += 1
    encoded = dumps(record["response"]).encode()
    record["response_bytes_base64"] = base64.b64encode(encoded).decode()
    record["response_sha256"] = sha(encoded)
    write_json(root / "record.json", record)
    manifest = loads((root / "capture.json").read_bytes())
    manifest["records"][0]["sha256"] = sha((root / "record.json").read_bytes())
    write_json(root / "capture.json", manifest)
    with pytest.raises(ReplayError, match="disagree"):
        run(setup, tmp_path / "wrong", captures=root)


@pytest.mark.parametrize(
    "output,finish,cap,valid",
    [
        ([9, 1], "tool_calls", 8, "stop"),
        ([9, 8], "tool_calls", 2, "length"),
        ([9, 8], "length", 2, "length"),
        ([9, 8], "stop", 2, None),
        ([9, 8], "length", 8, None),
        ([1, 9], "stop", 8, None),
        ([9, 1], "length", 2, None),
    ],
)
def test_terminal_evidence_uses_native_eos_and_explicit_cap(
    setup: dict[str, Any],
    tmp_path: Path,
    output: list[int],
    finish: str,
    cap: int,
    valid: str | None,
) -> None:
    setup["seed"]["request"]["max_tokens"] = cap
    root = capture(setup, tmp_path / "capture", output=output, finish=finish)
    if valid:
        run(setup, tmp_path / "out", captures=root)
        assert (
            load_corpus(tmp_path / "out").cases[0].worker_events[0]["finish_reason"]
            == valid
        )
    else:
        with pytest.raises(ReplayError):
            run(setup, tmp_path / "out", captures=root)


def test_assets_encoder_and_eos_are_verified_before_use(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    (setup["model"] / "encoding.py").write_bytes(VENDOR + b"# changed")
    with pytest.raises(ReplayError, match="encoder hash"):
        run(setup, tmp_path / "bad", requests=setup["seeds"])
    (setup["model"] / "encoding.py").write_bytes(VENDOR)
    write_json(setup["model"] / "config.json", {"eos_token_id": 2})
    with pytest.raises(ReplayError, match="asset hash"):
        run(setup, tmp_path / "bad", requests=setup["seeds"])
    document = loads(setup["path"].read_bytes())
    document["profile"]["model"]["assets"]["config.json"] = sha(
        (setup["model"] / "config.json").read_bytes()
    )
    write_json(setup["path"], document)
    with pytest.raises(ReplayError, match="conflicting native EOS"):
        run(setup, tmp_path / "bad", requests=setup["seeds"])


def test_reference_pins_cannot_belong_to_another_model(setup: dict[str, Any]) -> None:
    document = loads(setup["path"].read_bytes())
    document["reference"]["revision"] = "b" * 40
    write_json(setup["path"], document)
    with pytest.raises(ReplayError, match="repository/revision"):
        prepare.load_profile(setup["path"])


def test_reference_imports_cannot_use_unhashed_siblings(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    code = b"import sibling\n" + VENDOR
    (setup["model"] / "encoding.py").write_bytes(code)
    document = loads(setup["path"].read_bytes())
    document["reference"]["sha256"] = sha(code)
    write_json(setup["path"], document)
    with pytest.raises(ReplayError, match="non-stdlib/sibling"):
        run(setup, tmp_path / "bad", requests=setup["seeds"])


def test_synthetic_completion_includes_thinking_close_and_fits_cap(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    setup["seed"]["request"]["chat_template_kwargs"] = {
        "thinking_mode": "thinking",
        "reasoning_effort": 75,
    }
    setup["seed"]["request"]["max_tokens"] = 2
    setup["seeds"].write_text(dumps(setup["seed"]) + "\n")
    with pytest.raises(ReplayError, match="exceeds"):
        run(setup, tmp_path / "bad", requests=setup["seeds"])
    setup["seed"]["request"]["max_tokens"] = 3
    setup["seeds"].write_text(dumps(setup["seed"]) + "\n")
    run(setup, tmp_path / "good", requests=setup["seeds"])
    assert load_corpus(tmp_path / "good").cases[0].worker_events[0]["token_ids"] == [
        7,
        9,
        1,
    ]


def test_reviewed_responses_require_explicit_ids_pins_and_exact_binding(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    captured = capture(setup, tmp_path / "capture")
    root = tmp_path / "reviewed"
    run(setup, root, captures=captured)
    case = loads((root / "cases.jsonl").read_bytes())
    case["expect"]["response"] = {
        "http_status": 200,
        "message": {"content": "independent golden"},
        "finish_reason": "stop",
        "usage": {},
    }
    (root / "cases.jsonl").write_text(dumps(case) + "\n")
    manifest = loads((root / "manifest.json").read_bytes())
    manifest["traces"][0]["sha256"] = sha((root / "cases.jsonl").read_bytes())
    manifest["provenance"]["golden_review"] = {
        "ids": ["case"],
        "encoder": {
            key: setup["reference"][key]
            for key in ("repo_id", "revision", "path", "sha256")
        },
        "tokenizers_version": "0.21.4",
        "approved_capture_identities": {
            "source": {
                "repo_id": "vendor/model",
                "revision": "a" * 40,
                "encoder_sha256": sha(VENDOR),
                "tokenizer_sha256": setup["profile"]["model"]["assets"][
                    "tokenizer.json"
                ],
            },
            "records": {
                "case": {
                    "request_bytes_sha256": loads(
                        (captured / "record.json").read_bytes()
                    )["request_sha256"],
                    "prompt_token_ids_sha256": sha(
                        dumps(case["expect"]["preprocessed"]["token_ids"]).encode()
                    ),
                    "output_token_ids_sha256": sha(
                        dumps(case["replay"]["worker_events"][0]["token_ids"]).encode()
                    ),
                }
            },
        },
    }
    write_json(root / "manifest.json", manifest)
    run(setup, tmp_path / "approved", captures=captured, reviewed_corpus=root)
    assert (
        load_corpus(tmp_path / "approved")
        .cases[0]
        .expected_response["message"]["content"]
        == "independent golden"
    )
    for mutation in (
        "request",
        "prompt",
        "output",
        "terminal",
        "stop_reason",
        "source",
        "encoder",
        "request_ledger",
    ):
        modified = copy.deepcopy(case)
        changed_manifest = copy.deepcopy(manifest)
        review = changed_manifest["provenance"]["golden_review"]
        if mutation == "request":
            modified["request"]["max_tokens"] = 7
        elif mutation == "prompt":
            modified["expect"]["preprocessed"]["token_ids"][0] += 1
        elif mutation == "output":
            modified["replay"]["worker_events"][0]["token_ids"][0] += 1
        elif mutation == "terminal":
            modified["replay"]["worker_events"][0]["finish_reason"] = "length"
        elif mutation == "stop_reason":
            modified["replay"]["worker_events"][0]["stop_reason"] = 1
        elif mutation == "source":
            review["approved_capture_identities"]["source"]["revision"] = "b" * 40
        elif mutation == "encoder":
            review["encoder"]["sha256"] = "0" * 64
        else:
            review["approved_capture_identities"]["records"]["case"][
                "request_bytes_sha256"
            ] = ("0" * 64)
        (root / "cases.jsonl").write_text(dumps(modified) + "\n")
        changed_manifest["traces"][0]["sha256"] = sha(
            (root / "cases.jsonl").read_bytes()
        )
        write_json(root / "manifest.json", changed_manifest)
        with pytest.raises(ReplayError):
            run(
                setup,
                tmp_path / ("bad-" + mutation),
                captures=captured,
                reviewed_corpus=root,
            )
    (root / "cases.jsonl").write_text(dumps(case) + "\n")
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ReplayError, match="require captured"):
        run(
            setup,
            tmp_path / "no-capture",
            requests=setup["seeds"],
            reviewed_corpus=root,
        )
    manifest["provenance"]["golden_review"]["ids"] = []
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ReplayError, match="explicit approval list"):
        run(setup, tmp_path / "unlisted", captures=captured, reviewed_corpus=root)


def test_invalid_seed_and_message_shapes_fail_closed(setup: dict[str, Any]) -> None:
    encoded = (dumps(setup["seed"]) + "\n").encode()
    with pytest.raises(ReplayError, match="unique|duplicate"):
        prepare.read_seeds(encoded + encoded, setup["profile"])
    for message in (
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": "hi", "name": "ignored"},
        {"role": "assistant", "content": "hi", "prefix": True},
        {"role": "tool", "content": "result"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call",
                    "function": {"name": "x", "arguments": "[]"},
                }
            ],
        },
    ):
        request = {**setup["seed"]["request"], "messages": [message]}
        with pytest.raises(ReplayError):
            prepare.rendering_input(request, captured=False)


def test_optional_tokenizers_gate_is_lazy_and_uses_verified_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda name: "0.22.0")
    with pytest.raises(ReplayError, match="requires tokenizers==0.21.4"):
        prepare.ReferenceEncoder._tokenizer(b"{}", "0.21.4")
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda name: "0.21.4")
    seen: list[str] = []
    monkeypatch.setattr(
        prepare.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            Tokenizer=SimpleNamespace(from_str=lambda text: seen.append(text))
        ),
    )
    prepare.ReferenceEncoder._tokenizer(b'{"verified":true}', "0.21.4")
    assert seen == ['{"verified":true}']
