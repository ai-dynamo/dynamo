# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The convenience builder preserves evidence and publishes only complete corpora."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.frontend import test_trace_capture as http_fixtures
from tests.frontend.test_trace_prepare import setup as setup
from tests.frontend.trace_replay import build_corpus, prepare
from tests.frontend.trace_replay.contract import dumps, load_corpus, loads, write_json

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.timeout(20),
]


@pytest.fixture
def scenario(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    calls: list[Path] = []

    def acquire(profile: dict[str, Any], reference: dict[str, Any], cache: Path) -> Any:
        calls.append(cache)
        return SimpleNamespace(
            model_dir=setup["model"], encoder=setup["model"] / "encoding.py"
        )

    monkeypatch.setattr(build_corpus.assets, "resolve_cached_assets", acquire)
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda name: "0.21.4")
    monkeypatch.setattr(http_fixtures, "MODEL", setup["profile"]["model_name"])
    return {**setup, "downloads": calls, "cache": tmp_path / "cache"}


def build(
    scenario: dict[str, Any], url: str, output: Path, **kwargs: Any
) -> tuple[int, dict[str, Any]]:
    return build_corpus.build_corpus(
        scenario["seeds"],
        scenario["path"],
        url,
        output,
        cache_dir=scenario["cache"],
        **kwargs,
    )


def reply(
    scenario: dict[str, Any], payload: bytes, output: list[int] | None = None
) -> bytes:
    request = loads(payload)
    native = prepare.ReferenceEncoder(
        scenario["profile"], scenario["reference"], scenario["model"]
    )
    prompt, _ = native.prompt(request, captured=True)
    tokens = [9, 1] if output is None else output
    return dumps(
        {
            "prompt_token_ids": prompt,
            "choices": [
                {
                    "index": 0,
                    "token_ids": tokens,
                    "finish_reason": "stop",
                    "stop_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
        }
    ).encode()


def test_success_publishes_root_corpus_and_retains_frozen_evidence(
    scenario: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "built corpus"
    with http_fixtures.reference_server(
        factory=lambda payload: reply(scenario, payload)
    ) as (url, seen):
        code, result = build(scenario, url, output)
    assert code == 0 and result["status"] == "prepared"
    assert len(seen["post"]) == 1
    assert len(load_corpus(output).cases) == result["count"] == 1
    assert (output / "evidence/capture/capture.json").exists()
    assert (output / "evidence/inputs/requests.jsonl").read_bytes() == scenario[
        "seeds"
    ].read_bytes()
    assert (output / "evidence/inputs/profile.json").read_bytes() == scenario[
        "path"
    ].read_bytes()
    assert not (output / ".build-in-progress").exists()
    assert not (output / ".prepare-staging").exists()
    command = shlex.split(result["replay_command"])
    assert command[:4] == [
        "env",
        "PYTHONPATH=" + str(Path(build_corpus.__file__).resolve().parents[3]),
        "python",
        "-P",
    ]
    assert command[command.index("--corpus") + 1] == str(output.resolve())
    assert command[command.index("--model-dir") + 1] == str(scenario["model"].resolve())
    assert command[command.index("--output-dir") + 1] == str(
        output.resolve() / "replay-results"
    )


def test_exact_printed_replay_command_ignores_a_caller_checkout_decoy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intended = tmp_path / "intended checkout"
    decoy = tmp_path / "caller checkout"
    for root, label in ((intended, "intended"), (decoy, "decoy")):
        package = root / "tests/frontend/trace_replay"
        package.mkdir(parents=True)
        for directory in (root / "tests", root / "tests/frontend", package):
            (directory / "__init__.py").write_text("")
        (package / "__main__.py").write_text(
            "import json, sys\n"
            f"print(json.dumps({{'harness': {label!r}, 'argv': sys.argv[1:]}}))\n"
        )
    monkeypatch.setattr(
        build_corpus,
        "__file__",
        str(intended / "tests/frontend/trace_replay/build_corpus.py"),
    )
    output = tmp_path / "built corpus"
    model = tmp_path / "model cache"
    command = build_corpus.replay_command(output, model)
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }
    completed = subprocess.run(
        shlex.split(command),
        cwd=decoy,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    observed = loads(completed.stdout)
    assert observed["harness"] == "intended"
    assert observed["argv"] == [
        "run",
        "--corpus",
        str(output),
        "--model-dir",
        str(model),
        "--output-dir",
        str(output / "replay-results"),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "output",
        "url_auth",
        "timeout",
        "concurrency",
        "seed",
        "selector",
        "closed_request",
        "reserved_reference",
        "reviewed",
        "tokenizers",
    ],
)
def test_cheap_preflight_failures_do_not_download_or_contact_reference(
    scenario: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    output = tmp_path / "out"
    kwargs: dict[str, Any] = {}
    if failure == "output":
        output.mkdir()
        (output / "sentinel").write_text("keep")
    elif failure == "timeout":
        kwargs["timeout"] = float("nan")
    elif failure == "concurrency":
        kwargs["concurrency"] = True
    elif failure == "url_auth":
        monkeypatch.setenv("DYN_BUILD_TEST_KEY", "test-key")
        kwargs["api_key_env"] = "DYN_BUILD_TEST_KEY"
    elif failure == "reviewed":
        kwargs["reviewed_corpus"] = tmp_path / "missing-review"
    elif failure == "tokenizers":
        monkeypatch.setattr(
            prepare.importlib.metadata, "version", lambda name: "0.22.0"
        )
    else:
        seed = scenario["seed"]
        if failure == "seed":
            seed["request"]["model"] = "mismatched-alias"
        elif failure == "selector":
            seed["request"]["stream"] = True
        elif failure == "closed_request":
            seed["request"]["response_format"] = {}
        else:
            seed["reference"] = {"preparation": {}}
        scenario["seeds"].write_text(dumps(seed) + "\n")
    with http_fixtures.reference_server() as (url, seen):
        code, result = build(
            scenario,
            "http://example.com" if failure == "url_auth" else url,
            output,
            **kwargs,
        )
    assert code == 2 and result["stage"] == "preflight"
    assert not scenario["downloads"]
    assert not seen["get"] and not seen["post"]
    assert not (output / "manifest.json").exists()
    if failure == "output":
        assert (output / "sentinel").read_text() == "keep"
    else:
        assert not output.exists()


def test_prompt_preflight_vendor_exception_is_exit_two_without_secret_or_http(
    scenario: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code_bytes = b"def encode_messages(*args, **kwargs):\n    raise IndexError('private-vendor-message')\ndef render_message(*args, **kwargs):\n    return ''\n"
    (scenario["model"] / "encoding.py").write_bytes(code_bytes)
    document = loads(scenario["path"].read_bytes())
    document["reference"]["sha256"] = prepare._sha(code_bytes)
    write_json(scenario["path"], document)
    output = tmp_path / "out"
    with http_fixtures.reference_server() as (url, seen):
        status = build_corpus.main(
            [
                "--requests",
                str(scenario["seeds"]),
                "--profile",
                str(scenario["path"]),
                "--reference-url",
                url,
                "--output",
                str(output),
                "--cache-dir",
                str(scenario["cache"]),
            ]
        )
    printed = capsys.readouterr().out
    assert status == 2
    assert loads(printed)["error"] == "IndexError"
    assert "private-vendor-message" not in printed
    assert not seen["get"] and not seen["post"]
    assert not output.exists()


def test_captured_preflight_accepts_array_reasoning_and_tiny_valid_cap(
    scenario: dict[str, Any], tmp_path: Path
) -> None:
    seed = scenario["seed"]
    seed["request"]["messages"].insert(
        0,
        {
            "role": "assistant",
            "content": "prior",
            "reasoning_content": ["first", "", "second"],
        },
    )
    seed["request"]["max_tokens"] = 1
    scenario["seeds"].write_text(dumps(seed) + "\n")
    output = tmp_path / "out"
    with http_fixtures.reference_server(
        factory=lambda payload: reply(scenario, payload, [1])
    ) as (url, seen):
        code, _ = build(scenario, url, output)
    assert code == 0 and len(seen["post"]) == 1
    case = load_corpus(output).cases[0]
    assert case.request["messages"][0]["reasoning_content"] == ["first", "", "second"]
    assert case.worker_events[0]["token_ids"] == [1]


@pytest.mark.parametrize(
    "failure,expected", [("capture", 1), ("prompt", 2), ("publish", 2)]
)
def test_failures_retain_evidence_without_publishing_a_root_manifest(
    scenario: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: int,
) -> None:
    output = tmp_path / "out"
    if failure == "publish":
        real_link = os.link

        def fail_manifest(source: Path, destination: Path) -> None:
            if destination.name == "manifest.json":
                raise OSError("injected publication failure")
            real_link(source, destination)

        monkeypatch.setattr(build_corpus.os, "link", fail_manifest)

    def response(payload: bytes) -> bytes:
        if failure == "capture":
            return b"{}"
        raw = loads(reply(scenario, payload))
        if failure == "prompt":
            raw["prompt_token_ids"][0] += 1
        return dumps(raw).encode()

    with http_fixtures.reference_server(factory=response) as (url, _):
        code, _ = build(scenario, url, output)
    assert code == expected
    assert (output / "evidence/capture/capture.json").exists()
    assert (output / ".build-in-progress").exists()
    assert not (output / "manifest.json").exists()


def test_input_snapshots_survive_original_file_changes_during_acquisition(
    scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_seeds = scenario["seeds"].read_bytes()
    expected_profile = scenario["path"].read_bytes()
    acquire = build_corpus.assets.resolve_cached_assets

    def change_originals(
        profile: dict[str, Any], reference: dict[str, Any], cache: Path
    ) -> Any:
        scenario["seeds"].write_bytes(b"not JSON")
        scenario["path"].write_bytes(b"not JSON")
        return acquire(profile, reference, cache)

    monkeypatch.setattr(build_corpus.assets, "resolve_cached_assets", change_originals)
    output = tmp_path / "out"
    with http_fixtures.reference_server(
        factory=lambda payload: reply(scenario, payload)
    ) as (url, _):
        code, _ = build(scenario, url, output)
    assert code == 0
    assert (output / "evidence/inputs/requests.jsonl").read_bytes() == expected_seeds
    assert (output / "evidence/inputs/profile.json").read_bytes() == expected_profile
    assert (output / "evidence/capture/requests.jsonl").read_bytes() == expected_seeds


@pytest.mark.parametrize("when", ["assets", "capture", "prepare"])
def test_reviewed_identity_changes_block_publication(
    scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    reviewed = tmp_path / "reviewed"
    prepare.prepare_corpus(
        scenario["path"], scenario["model"], reviewed, requests=scenario["seeds"]
    )
    manifest = loads((reviewed / "manifest.json").read_bytes())
    manifest["provenance"]["golden_review"] = {
        "ids": [],
        "encoder": {
            key: scenario["reference"][key]
            for key in ("repo_id", "revision", "path", "sha256")
        },
        "tokenizers_version": "0.21.4",
        "approved_capture_identities": {
            "source": {
                "repo_id": scenario["reference"]["repo_id"],
                "revision": scenario["reference"]["revision"],
                "encoder_sha256": scenario["reference"]["sha256"],
                "tokenizer_sha256": scenario["profile"]["model"]["assets"][
                    "tokenizer.json"
                ],
            },
            "records": {},
        },
    }
    write_json(reviewed / "manifest.json", manifest)

    def mutate_review() -> None:
        manifest["provenance"]["changed"] = True
        write_json(reviewed / "manifest.json", manifest)

    if when == "assets":
        acquire = build_corpus.assets.resolve_cached_assets

        def change_during_acquisition(
            profile: dict[str, Any], reference: dict[str, Any], cache: Path
        ) -> Any:
            mutate_review()
            return acquire(profile, reference, cache)

        monkeypatch.setattr(
            build_corpus.assets, "resolve_cached_assets", change_during_acquisition
        )
    elif when == "prepare":
        original_prepare = prepare.prepare_corpus

        def change_before_prepare_reads(*args: Any, **kwargs: Any) -> dict[str, Any]:
            mutate_review()
            return original_prepare(*args, **kwargs)

        monkeypatch.setattr(prepare, "prepare_corpus", change_before_prepare_reads)

    def respond(payload: bytes) -> bytes:
        if when == "capture":
            mutate_review()
        return reply(scenario, payload)

    output = tmp_path / "out"
    with http_fixtures.reference_server(factory=respond) as (url, seen):
        code, result = build(scenario, url, output, reviewed_corpus=reviewed)
    assert code == 2
    assert result["stage"] == ("prompt_preflight" if when == "assets" else "prepare")
    assert not (output / "manifest.json").exists()
    assert len(seen["post"]) == (0 if when == "assets" else 1)


def test_postcommit_cleanup_failure_keeps_successful_loadable_corpus(
    scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    real_remove = build_corpus.shutil.rmtree

    def fail_staging(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == ".prepare-staging":
            raise OSError("injected cleanup failure")
        real_remove(path, *args, **kwargs)

    monkeypatch.setattr(build_corpus.shutil, "rmtree", fail_staging)
    with http_fixtures.reference_server(
        factory=lambda payload: reply(scenario, payload)
    ) as (url, _):
        code, result = build(scenario, url, output)
    assert code == 0 and result["status"] == "prepared"
    assert result["warnings"] == [{"stage": "cleanup", "error": "OSError"}]
    assert len(load_corpus(output).cases) == 1
    assert (output / ".prepare-staging").exists()
    assert (output / ".build-in-progress").exists()


@pytest.mark.parametrize("xdg", [None, "", "custom"])
def test_cli_default_cache_obeys_xdg_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str | None
) -> None:
    received: list[Path] = []
    if xdg is None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / xdg) if xdg else "")

    def capture_arguments(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        received.append(kwargs["cache_dir"])
        return 2, {"status": "test"}

    monkeypatch.setattr(build_corpus, "build_corpus", capture_arguments)
    assert (
        build_corpus.main(
            [
                "--requests",
                "unused",
                "--reference-url",
                "unused",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    base = tmp_path / "custom" if xdg else Path.home() / ".cache"
    assert received == [base / "dynamo/frontend-replay"]


def test_alias_wins_but_explicit_paths_and_corpus_directories_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deepseek-v41").write_text("decoy")
    assert build_corpus.resolve_profile("deepseek-v41").parent.name == "deepseek_v41"
    assert build_corpus.resolve_profile("./deepseek-v41") == tmp_path / "deepseek-v41"
    assert (
        build_corpus.resolve_profile(str(tmp_path / "deepseek-v41"))
        == tmp_path / "deepseek-v41"
    )
    directory = tmp_path / "profile"
    directory.mkdir()
    assert build_corpus.resolve_profile(str(directory)) == directory / "manifest.json"
