# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public capture archives exact evidence and refuses incomplete inputs."""

from __future__ import annotations

import base64
import hashlib
import stat
import threading
from contextlib import contextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from tests.frontend.trace_replay.capture import capture_requests, main
from tests.frontend.trace_replay.contract import SCHEMA, ReplayError, dumps, loads

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.timeout(20),
]

MODEL = "capture-unit-model"


def seed(identifier: str = "one") -> dict[str, Any]:
    return {
        "id": identifier,
        "source": {"kind": "authored"},
        "request": {
            "model": MODEL,
            "messages": [{"role": "user", "content": identifier}],
        },
    }


def inputs(tmp_path: Path, seeds: list[dict[str, Any]]) -> tuple[Path, Path]:
    requests = tmp_path / "seeds.jsonl"
    requests.write_text("".join(dumps(item) + "\n" for item in seeds))
    profile = tmp_path / "profile.json"
    profile.write_text(
        dumps(
            {
                "schema": SCHEMA,
                "profile": {
                    "name": "capture-unit",
                    "model_name": MODEL,
                    "model": {
                        "repo_id": "test/model",
                        "revision": "a" * 40,
                        "assets": {
                            name: "b" * 64
                            for name in (
                                "config.json",
                                "tokenizer.json",
                                "tokenizer_config.json",
                            )
                        },
                    },
                    "runtime_config": {
                        "tool_call_parser": "deepseek_v41",
                        "reasoning_parser": "deepseek_v41",
                        "context_length": 32768,
                    },
                },
            }
        )
    )
    return requests, profile


def response() -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt_token_ids": [0, 15],
        "choices": [
            {
                "index": 0,
                "token_ids": [30, 1],
                "finish_reason": "tool_calls",
                "stop_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


@contextmanager
def reference_server(
    *,
    body: bytes | None = None,
    status: int = 200,
    content_type: str = "application/json",
    factory: Callable[[bytes], bytes] | None = None,
    metadata_status: int = 200,
) -> Iterator[tuple[str, dict[str, list[dict[str, Any]]]]]:
    seen: dict[str, list[dict[str, Any]]] = {"get": [], "post": []}
    default = dumps(response()).encode() if body is None else body

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def send(self, payload: bytes, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if 300 <= code < 400:
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            seen["get"].append(
                {"path": self.path, "authorization": self.headers.get("Authorization")}
            )
            if self.path == "/version":
                self.send(
                    dumps(
                        {"version": "test-1", "ignored_secret_field": "not-retained"}
                    ).encode(),
                    metadata_status,
                )
            elif self.path == "/v1/models":
                self.send(
                    dumps(
                        {"data": [{"id": MODEL, "root": "/private/model-path"}]}
                    ).encode(),
                    200,
                )
            else:
                self.send(b"{}", 404)

        def do_POST(self) -> None:
            data = self.rfile.read(int(self.headers["Content-Length"]))
            seen["post"].append(
                {
                    "path": self.path,
                    "body": data,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            self.send(factory(data) if factory else default, status)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_capture_preserves_wire_bytes_and_archives_only_selected_seed_metadata(
    tmp_path: Path,
) -> None:
    original = seed("../unsafe/name")
    original["request"]["temperature"] = Decimal("0.10000000000000001")
    original["reference"] = {"bfcl": {"case": "sample"}}
    original["expect"] = {"response": {"must_not_trust": True}}
    original["replay"] = {"worker_events": []}
    requests, profile = inputs(tmp_path, [original])
    raw = (
        " \n"
        + dumps(
            {**response(), "diagnostic": Decimal("9007199254740993.0"), "text": "café"}
        )
        + "\n"
    ).encode()
    output = tmp_path / "capture"
    with reference_server(body=raw, content_type="text/plain; charset=iso-8859-1") as (
        url,
        seen,
    ):
        manifest = capture_requests(requests, profile, url, output)
    assert manifest["complete"] is True
    assert manifest["count"] == manifest["successful_count"] == 1
    assert (output / "requests.jsonl").read_bytes() == requests.read_bytes()
    assert (
        manifest["requests_sha256"] == hashlib.sha256(requests.read_bytes()).hexdigest()
    )
    assert manifest["reference"] == {"url": url, "version": "test-1", "models": [MODEL]}
    entry = manifest["records"][0]
    assert (
        entry["path"]
        == "records/" + hashlib.sha256(original["id"].encode()).hexdigest() + ".json"
    )
    record_path = output / entry["path"]
    record = loads(record_path.read_bytes())
    assert record["original_request"] == original["request"]
    assert record["reference"] == original["reference"]
    assert "expect" not in record and "replay" not in record
    sent = seen["post"][0]["body"]
    assert sent == record["wire_request_bytes"].encode()
    assert record["request_sha256"] == hashlib.sha256(sent).hexdigest()
    wire = loads(sent)
    assert wire["stream"] is False and wire["return_token_ids"] is True
    assert wire["temperature"] == original["request"]["temperature"]
    assert base64.b64decode(record["response_bytes_base64"]) == raw
    assert record["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["response"]["diagnostic"] == Decimal("9007199254740993.0")
    assert record["response"]["choices"][0]["finish_reason"] == "tool_calls"
    assert entry["sha256"] == hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert loads((output / "capture.json").read_bytes()) == manifest
    assert not (output / ".capture-in-progress").exists()
    for path in (output / "requests.jsonl", output / "capture.json", record_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("separator", ["\u2028", "\u0085"])
def test_unicode_separators_inside_json_strings_are_not_record_boundaries(
    tmp_path: Path, separator: str
) -> None:
    original = seed()
    original["request"]["messages"][0]["content"] = "before" + separator + "after"
    requests, profile = inputs(tmp_path, [original])
    assert separator.encode("utf-8") in requests.read_bytes()
    output = tmp_path / "capture"
    with reference_server() as (url, seen):
        manifest = capture_requests(requests, profile, url, output)
    assert manifest["complete"] is True
    assert len(seen["post"]) == 1
    record = loads((output / manifest["records"][0]["path"]).read_bytes())
    assert record["original_request"] == original["request"]
    assert (output / "requests.jsonl").read_bytes() == requests.read_bytes()


@pytest.mark.parametrize("encoding", ["utf8_bom", "utf16", "invalid_utf8"])
def test_seed_encoding_errors_are_rejected_before_http(
    tmp_path: Path, encoding: str
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    raw = requests.read_bytes()
    if encoding == "utf8_bom":
        raw = b"\xef\xbb\xbf" + raw
    elif encoding == "utf16":
        raw = raw.decode("utf-8").encode("utf-16")
    else:
        raw = raw.replace(b"one", b"\xff", 1)
    requests.write_bytes(raw)
    output = tmp_path / "capture"
    with reference_server() as (url, seen):
        with pytest.raises(ReplayError):
            capture_requests(requests, profile, url, output)
    assert not seen["get"] and not seen["post"]
    assert not output.exists()


@pytest.mark.parametrize(
    "selectors", [{}, {"stream": False}, {"stream": False, "return_token_ids": True}]
)
def test_wire_selector_order_matches_the_reviewed_request_ledger(
    tmp_path: Path, selectors: dict[str, Any]
) -> None:
    original = seed()
    # Existing selectors can precede effective fields in a supplied seed.
    original["request"] = {**selectors, **original["request"]}
    requests, profile = inputs(tmp_path, [original])
    output = tmp_path / "capture"
    with reference_server() as (url, seen):
        manifest = capture_requests(requests, profile, url, output)
    effective = {
        key: value
        for key, value in original["request"].items()
        if key not in {"stream", "return_token_ids"}
    }
    expected = dumps({**effective, "return_token_ids": True, "stream": False}).encode()
    sent = seen["post"][0]["body"]
    assert sent == expected
    assert list(loads(sent))[-2:] == ["return_token_ids", "stream"]
    record = loads((output / manifest["records"][0]["path"]).read_bytes())
    assert record["request_sha256"] == hashlib.sha256(expected).hexdigest()
    assert record["original_request"] == original["request"]


@pytest.mark.parametrize(
    "change",
    [
        {"stream": True},
        {"stream": 0},
        {"stream_options": {}},
        {"n": 2},
        {"n": True},
        {"return_token_ids": False},
        {"return_prompt_token_ids": True},
        {"return_meta_info": False},
        {"model": "another-model"},
    ],
)
def test_all_seed_selector_conflicts_are_rejected_before_any_http(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    bad = seed("bad")
    bad["request"].update(change)
    requests, profile = inputs(tmp_path, [seed("valid-first"), bad])
    with reference_server() as (url, seen):
        with pytest.raises(ReplayError):
            capture_requests(requests, profile, url, tmp_path / "capture")
    assert seen == {"get": [], "post": []}
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize(
    "field,value", [("id", "one"), ("source", {"kind": "unknown"}), ("reference", [])]
)
def test_duplicate_ids_or_bad_source_metadata_preflight_before_http(
    tmp_path: Path, field: str, value: Any
) -> None:
    bad = seed("second")
    bad[field] = value
    requests, profile = inputs(tmp_path, [seed(), bad])
    with reference_server() as (url, seen):
        with pytest.raises(ReplayError):
            capture_requests(requests, profile, url, tmp_path / "capture")
    assert not seen["get"] and not seen["post"]


@pytest.mark.parametrize("ids", [[], [True], [-1], [2**32]])
@pytest.mark.parametrize("location", ["prompt", "output"])
def test_malformed_token_ids_remain_archived_but_cannot_complete_capture(
    tmp_path: Path, ids: list[Any], location: str
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    invalid = response()
    if location == "prompt":
        invalid["prompt_token_ids"] = ids
    else:
        invalid["choices"][0]["token_ids"] = ids
    raw = dumps(invalid).encode()
    with reference_server(body=raw) as (url, _):
        manifest = capture_requests(requests, profile, url, tmp_path / "capture")
    assert manifest["complete"] is False and manifest["successful_count"] == 0
    record = loads((tmp_path / "capture" / manifest["records"][0]["path"]).read_bytes())
    assert base64.b64decode(record["response_bytes_base64"]) == raw
    assert record["error"]["kind"] == "token_evidence"


def test_invalid_utf8_is_not_replacement_decoded(tmp_path: Path) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    raw = b'{"text":"\xff"}'
    with reference_server(body=raw) as (url, _):
        manifest = capture_requests(requests, profile, url, tmp_path / "capture")
    record = loads((tmp_path / "capture" / manifest["records"][0]["path"]).read_bytes())
    assert manifest["complete"] is False
    assert base64.b64decode(record["response_bytes_base64"]) == raw
    assert record["response"] is None
    assert record["error"] == {
        "kind": "response_encoding",
        "message": "UnicodeDecodeError",
    }


def test_partial_capture_manifest_keeps_exact_seed_order_and_is_incomplete(
    tmp_path: Path,
) -> None:
    requests, profile = inputs(tmp_path, [seed("one"), seed("bad"), seed("three")])

    def reply(data: bytes) -> bytes:
        return (
            b"{}"
            if loads(data)["messages"][0]["content"] == "bad"
            else dumps(response()).encode()
        )

    with reference_server(factory=reply) as (url, seen):
        manifest = capture_requests(
            requests, profile, url, tmp_path / "capture", concurrency=3
        )
    assert len(seen["post"]) == 3
    assert manifest["complete"] is False
    assert manifest["count"] == 3 and manifest["successful_count"] == 2
    assert [item["id"] for item in manifest["records"]] == ["one", "bad", "three"]
    assert [item["status"] for item in manifest["records"]] == [
        "captured",
        "error",
        "captured",
    ]


def test_redirects_are_errors_and_never_followed(tmp_path: Path) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    with reference_server(status=302) as (url, seen):
        manifest = capture_requests(requests, profile, url, tmp_path / "capture")
    assert manifest["complete"] is False
    assert [item["path"] for item in seen["get"]] == ["/version", "/v1/models"]
    record = loads((tmp_path / "capture" / manifest["records"][0]["path"]).read_bytes())
    assert record["http_status"] == 302
    assert record["error"]["kind"] == "http"


def test_nonempty_output_is_never_overwritten(tmp_path: Path) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    output = tmp_path / "capture"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("original")
    with reference_server() as (url, seen):
        with pytest.raises(ReplayError, match="must be empty"):
            capture_requests(requests, profile, url, output)
    assert sentinel.read_text() == "original"
    assert not seen["get"] and not seen["post"]


def test_metadata_failure_leaves_no_completion_manifest(tmp_path: Path) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    output = tmp_path / "capture"
    with reference_server(metadata_status=503) as (url, seen):
        with pytest.raises(ReplayError, match="metadata request failed"):
            capture_requests(requests, profile, url, output)
    assert not seen["post"]
    assert (output / "requests.jsonl").read_bytes() == requests.read_bytes()
    assert (output / ".capture-in-progress").exists()
    assert not (output / "capture.json").exists()


def test_environment_credentials_and_proxies_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login unwanted password unwanted\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with reference_server() as (url, seen):
        manifest = capture_requests(
            requests, profile, url, tmp_path / "capture", timeout=2
        )
    assert manifest["complete"] is True
    assert all(item["authorization"] is None for item in seen["get"] + seen["post"])


def test_explicit_loopback_auth_is_never_archived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    secret = "capture-unit-test-key"
    monkeypatch.setenv("CAPTURE_UNIT_KEY", secret)
    output = tmp_path / "capture"
    with reference_server() as (url, seen):
        manifest = capture_requests(
            requests, profile, url, output, api_key_env="CAPTURE_UNIT_KEY"
        )
    assert manifest["complete"] is True
    assert all(
        item["authorization"] == "Bearer " + secret
        for item in seen["get"] + seen["post"]
    )
    for path in output.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


@pytest.mark.parametrize("key", [None, "line\nbreak", "non-ascii-\u00e9"])
def test_empty_or_invalid_explicit_api_key_is_rejected_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    if key is None:
        monkeypatch.delenv("CAPTURE_UNIT_KEY", raising=False)
    else:
        monkeypatch.setenv("CAPTURE_UNIT_KEY", key)
    with reference_server() as (url, seen):
        with pytest.raises(ReplayError) as exc:
            capture_requests(
                requests,
                profile,
                url,
                tmp_path / "capture",
                api_key_env="CAPTURE_UNIT_KEY",
            )
        assert (
            main(
                [
                    "--requests",
                    str(requests),
                    "--profile",
                    str(profile),
                    "--url",
                    url,
                    "--output",
                    str(tmp_path / "capture"),
                    "--api-key-env",
                    "CAPTURE_UNIT_KEY",
                ]
            )
            == 2
        )
    assert not seen["get"] and not seen["post"]
    assert not (tmp_path / "capture").exists()
    if key:
        assert key not in str(exc.value)
        assert key not in capsys.readouterr().out


def test_credentials_require_tls_and_url_secrets_do_not_reach_cli_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    monkeypatch.setenv("CAPTURE_UNIT_KEY", "capture-unit-test-key")
    with pytest.raises(ReplayError, match="require HTTPS"):
        capture_requests(
            requests,
            profile,
            "http://example.com",
            tmp_path / "capture",
            api_key_env="CAPTURE_UNIT_KEY",
        )
    assert (
        main(
            [
                "--requests",
                str(requests),
                "--profile",
                str(profile),
                "--url",
                "http://user:fake-url-secret@localhost",
                "--output",
                str(tmp_path / "capture"),
            ]
        )
        == 2
    )
    assert "fake-url-secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": 0},
        {"concurrency": True},
        {"concurrency": 1.2},
        {"concurrency": 0},
    ],
)
def test_invalid_limits_fail_before_creating_output(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    requests, profile = inputs(tmp_path, [seed()])
    with pytest.raises(ReplayError):
        capture_requests(
            requests, profile, "http://localhost", tmp_path / "capture", **kwargs
        )
    assert not (tmp_path / "capture").exists()
