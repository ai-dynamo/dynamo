# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned asset acquisition preserves evidence and explicit network settings."""

from __future__ import annotations

import copy
import hashlib
import io
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.frontend.trace_replay import assets, runner
from tests.frontend.trace_replay.contract import ReplayError, dumps, loads, write_json

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.timeout(20),
]


@dataclass
class Remote:
    profile: dict[str, Any]
    reference: dict[str, Any]
    payloads: dict[str, bytes]
    seen: list[dict[str, Any]]
    routes: dict[str, tuple[int, bytes, dict[str, str]]]

    def url(self, name: str) -> str:
        model = self.profile["model"]
        return f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> Remote:
    fixture = Path(__file__).parent / "trace_replay/fixtures/deepseek_v41/manifest.json"
    manifest = loads(fixture.read_bytes())
    profile = manifest["profile"]
    reference = {
        **manifest["provenance"]["golden_review"]["encoder"],
        "tokenizers_version": "0.21.4",
    }
    payloads = {
        name: ("verified " + name).encode() for name in profile["model"]["assets"]
    }
    payloads[reference["path"]] = b"# verified encoder bytes\n"
    profile["model"]["assets"] = {
        name: hashlib.sha256(payloads[name]).hexdigest()
        for name in profile["model"]["assets"]
    }
    reference["sha256"] = hashlib.sha256(payloads[reference["path"]]).hexdigest()
    state = Remote(profile, reference, payloads, [], {})

    def send(
        adapter: requests.adapters.HTTPAdapter,
        request: requests.PreparedRequest,
        **kwargs: Any,
    ) -> requests.Response:
        state.seen.append(
            {"url": request.url, "headers": dict(request.headers), **kwargs}
        )
        if request.url in state.routes:
            status, body, headers = state.routes[request.url]
        else:
            matching = [
                payload
                for name, payload in payloads.items()
                if request.url == state.url(name)
            ]
            if not matching:
                raise AssertionError("unexpected asset URL")
            status, body, headers = 200, matching[0], {}
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response.request = request
        response.headers.update(headers)
        response.raw = io.BytesIO(body)
        response._content = body
        response._content_consumed = True
        response.connection = adapter
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return state


def test_cold_download_then_readonly_warm_cache_reuses_all_verified_bytes(
    remote: Remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = assets.resolve_cached_assets(
        remote.profile, remote.reference, tmp_path / "cache"
    )
    assert bundle.model_dir.name == assets.cache_identity(
        remote.profile, remote.reference
    )
    assert bundle.encoder == bundle.model_dir / remote.reference["path"]
    assert set(bundle.sha256) == set(remote.payloads)
    assert len(remote.seen) == 4
    for name, data in remote.payloads.items():
        assert (bundle.model_dir / name).read_bytes() == data
        assert stat.S_IMODE((bundle.model_dir / name).stat().st_mode) == 0o600
    remote.seen.clear()

    def no_write(*args: Any, **kwargs: Any) -> int:
        raise AssertionError(
            "a complete cache must not create locks or temporary files"
        )

    monkeypatch.setattr(assets.os, "open", no_write)
    repeated = assets.resolve_cached_assets(
        remote.profile, remote.reference, tmp_path / "cache"
    )
    assert repeated == bundle
    assert remote.seen == []


def test_cache_identity_binds_only_immutable_file_identities(remote: Remote) -> None:
    changed = copy.deepcopy(remote.profile)
    changed["name"] = "another suite"
    changed["model_name"] = "another served alias"
    changed["runtime_config"]["context_length"] += 1
    key = assets.cache_identity(remote.profile, remote.reference)
    assert assets.cache_identity(changed, remote.reference) == key
    changed["model"]["assets"]["config.json"] = "a" * 64
    assert assets.cache_identity(changed, remote.reference) != key
    assert (
        assets.cache_identity(remote.profile, {**remote.reference, "sha256": "b" * 64})
        != key
    )


def test_legacy_prepare_assets_is_native_only_and_in_place_with_hf_symlinks(
    remote: Remote, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    case = {
        "id": "one",
        "source": {"kind": "authored"},
        "request": {},
        "replay": {"worker_events": [{"token_ids": [1], "finish_reason": "stop"}]},
        "expect": {"preprocessed": {"token_ids": [0]}},
    }
    data = (dumps(case) + "\n").encode()
    (corpus / "cases.jsonl").write_bytes(data)
    write_json(
        corpus / "manifest.json",
        {
            "schema": "dynamo-frontend-replay/v1",
            "profile": remote.profile,
            "traces": [
                {
                    "path": "cases.jsonl",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "count": 1,
                }
            ],
        },
    )
    model = tmp_path / "existing-bind-mount"
    model.mkdir()
    (model / "unrelated").write_text("keep")
    blob = tmp_path / "hf-tokenizer-blob"
    blob.write_bytes(remote.payloads["tokenizer.json"])
    (model / "tokenizer.json").symlink_to(blob)
    result = runner.prepare_assets(corpus, model)
    assert result == remote.profile["model"]["assets"]
    assert (model / "unrelated").read_text() == "keep"
    assert (model / "tokenizer.json").is_symlink()
    assert not (model / "encoding").exists()
    assert len(remote.seen) == 2
    assert set(path.name for path in model.iterdir()) == set(result) | {
        "unrelated",
        ".frontend-replay-assets.lock",
    }


@pytest.mark.parametrize("existing", ["corrupt", "dangling"])
def test_corrupt_or_dangling_existing_assets_fail_without_network_or_overwrite(
    remote: Remote, tmp_path: Path, existing: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    destination = model / "config.json"
    if existing == "corrupt":
        destination.write_bytes(b"must not be replaced")
    else:
        destination.symlink_to(tmp_path / "missing-blob")
    with pytest.raises(ReplayError, match="existing model asset"):
        assets.ensure_assets(remote.profile, model)
    assert remote.seen == []
    assert (
        destination.is_symlink()
        if existing == "dangling"
        else destination.read_bytes() == b"must not be replaced"
    )


def test_download_hash_failure_keeps_destination_absent(
    remote: Remote, tmp_path: Path
) -> None:
    remote.routes[remote.url("config.json")] = (200, b"different bytes", {})
    with pytest.raises(ReplayError, match="downloaded model asset"):
        assets.ensure_assets(remote.profile, tmp_path / "model")
    assert not (tmp_path / "model/config.json").exists()
    assert not list((tmp_path / "model").glob(".asset-*"))


def test_publication_uses_advisory_ownership_without_hardlinks_and_preserves_late_corruption(
    remote: Remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_hardlinks(*args: Any, **kwargs: Any) -> None:
        raise OSError("filesystem does not support hardlinks")

    monkeypatch.setattr(assets.os, "link", no_hardlinks)
    assets.ensure_assets(remote.profile, tmp_path / "supported")
    original = assets._publish

    def late_winner(
        temporary: Path, destination: Path, expected: str, name: str
    ) -> None:
        destination.write_bytes(b"late corrupt winner")
        original(temporary, destination, expected, name)

    monkeypatch.setattr(assets, "_publish", late_winner)
    with pytest.raises(ReplayError, match="existing model asset"):
        assets.ensure_assets(remote.profile, tmp_path / "racing")
    assert (tmp_path / "racing/config.json").read_bytes() == b"late corrupt winner"


@pytest.mark.parametrize("bundle_variable", ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_https_redirects_preserve_proxy_ca_settings_but_strip_auth_and_ignore_netrc(
    remote: Remote,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_variable: str,
) -> None:
    for name in (
        "https_proxy",
        "HTTPS_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "https://explicit-proxy.test:9443")
    monkeypatch.setenv("NO_PROXY", "cdn.test")
    monkeypatch.setenv(bundle_variable, str(tmp_path / "explicit-ca.pem"))
    monkeypatch.setenv("HF_TOKEN", "explicit-test-token")
    netrc = tmp_path / "netrc"
    netrc.write_text(
        "machine huggingface.co login unexpected password secret\nmachine cdn.test login unexpected password secret\n"
    )
    monkeypatch.setenv("NETRC", str(netrc))

    def no_netrc(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("netrc credentials must never be read")

    monkeypatch.setattr(requests.sessions, "get_netrc_auth", no_netrc)
    relative = remote.url("config.json") + "?relative"
    cdn = "https://cdn.test/asset?signature=private-diagnostic-value"
    remote.routes[remote.url("config.json")] = (
        307,
        b"",
        {"Location": relative.split("huggingface.co", 1)[1]},
    )
    remote.routes[relative] = (302, b"", {"Location": cdn})
    remote.routes[cdn] = (200, remote.payloads["config.json"], {})
    assets.ensure_assets(remote.profile, tmp_path / "model")
    first, second, third = remote.seen[:3]
    assert first["headers"]["Authorization"] == "Bearer explicit-test-token"
    assert second["headers"]["Authorization"] == "Bearer explicit-test-token"
    assert "Authorization" not in third["headers"]
    assert (
        first["proxies"]["https"]
        == second["proxies"]["https"]
        == "https://explicit-proxy.test:9443"
    )
    assert third["proxies"] == {}
    assert all(
        call["verify"] == str(tmp_path / "explicit-ca.pem") for call in remote.seen
    )


def test_https_downgrade_redirect_is_rejected_before_sending_credentials(
    remote: Remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "explicit-test-token")
    remote.routes[remote.url("config.json")] = (
        302,
        b"",
        {"Location": "http://untrusted.test/asset"},
    )
    with pytest.raises(ReplayError, match="requires HTTPS"):
        assets.ensure_assets(remote.profile, tmp_path / "model")
    assert len(remote.seen) == 1


@pytest.mark.parametrize("token", ["secret\n", " secret", "secret\t", "secret\u03c0"])
def test_invalid_hf_token_is_rejected_without_exposing_it(
    remote: Remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv("HF_TOKEN", token)
    with pytest.raises(ReplayError, match="visible ASCII") as raised:
        assets.ensure_assets(remote.profile, tmp_path / "model")
    assert "secret" not in str(raised.value)
    assert remote.seen == []


def test_http_error_does_not_echo_signed_redirect_or_token(
    remote: Remote, tmp_path: Path
) -> None:
    signed = "https://cdn.test/asset?signature=private-diagnostic-value"
    remote.routes[remote.url("config.json")] = (302, b"", {"Location": signed})
    remote.routes[signed] = (403, b"forbidden", {})
    with pytest.raises(ReplayError) as raised:
        assets.ensure_assets(remote.profile, tmp_path / "model")
    assert "HTTPError" in str(raised.value)
    assert "private-diagnostic-value" not in str(raised.value)
    assert "cdn.test" not in str(raised.value)


def test_assets_import_needs_no_runtime_or_process_helpers() -> None:
    code = "from tests.frontend.trace_replay import assets; import sys; assert 'tests.frontend.trace_replay.runner' not in sys.modules; assert not any(name == 'dynamo' or name.startswith('dynamo.') for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
