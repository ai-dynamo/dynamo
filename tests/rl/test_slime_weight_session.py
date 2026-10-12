# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Slime example's SGLang weight-update session hook."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "rl"
    / "slime"
    / "sglang_weight_session.py"
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]


@pytest.fixture
def session_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_weight_session_under_test", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEngine:
    """Answers the SGLang 0.5.21 weight-update session contract."""

    def __init__(self, *, has_sessions: bool = True, fail_begin: bool = False):
        self.has_sessions = has_sessions
        self.fail_begin = fail_begin
        self.session_open = False
        self.calls: list[str] = []
        self._lock = threading.Lock()
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                status, body = engine.handle(self.path.lstrip("/"))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def handle(self, route: str) -> tuple[int, dict]:
        with self._lock:
            self.calls.append(route)
            if route in ("begin_weight_update", "end_weight_update"):
                if not self.has_sessions:
                    return 404, {"detail": "Not Found"}
                if route == "begin_weight_update":
                    if self.fail_begin or self.session_open:
                        return 400, {"success": False, "message": "nested begin"}
                    self.session_open = True
                    return 200, {"success": True, "message": ""}
                if not self.session_open:
                    return 400, {"success": False, "message": "end without begin"}
                self.session_open = False
                return 200, {"success": True, "message": ""}
            if route == "update_weights_from_distributed":
                if self.has_sessions and not self.session_open:
                    return 400, {
                        "success": False,
                        "message": "update_weights_from_distributed must run "
                        "between begin_weight_update() and end_weight_update()",
                    }
                return 200, {"success": True, "message": ""}
            return 404, {"detail": "Not Found"}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engines():
    created: list[_FakeEngine] = []

    def make(**kwargs) -> _FakeEngine:
        engine = _FakeEngine(**kwargs)
        created.append(engine)
        return engine

    yield make
    for engine in created:
        engine.close()


class _Updater:
    """Stands in for Slime's UpdateWeightFromDistributed."""

    def __init__(self, engine_urls, fail: bool = False):
        self.engine_urls = engine_urls
        self.fail = fail

    def _send_weights(self, pbar):
        for url in self.engine_urls:
            response = requests.post(
                f"{url}/update_weights_from_distributed", json={}, timeout=5
            )
            response.raise_for_status()
        if self.fail:
            raise RuntimeError("broadcast failed")
        return pbar


def _wrap(module, urls, owner=True):
    barriers: list[bool] = []
    wrapped = module.wrap_send_weights(
        _Updater._send_weights,
        urls,
        is_session_owner=lambda: owner,
        barrier=lambda: barriers.append(True),
    )
    return wrapped, barriers


def test_wrapper_opens_one_session_around_all_buckets(session_module, engines):
    first, second = engines(), engines()
    urls = [first.url, second.url]
    wrapped, barriers = _wrap(session_module, urls)

    assert wrapped(_Updater(urls), "pbar") == "pbar"
    assert barriers == [True]
    for engine in (first, second):
        assert engine.calls == [
            "begin_weight_update",
            "update_weights_from_distributed",
            "end_weight_update",
        ]
        assert not engine.session_open
    # A second policy update can open a new session.
    wrapped(_Updater(urls), None)
    assert first.calls.count("begin_weight_update") == 2


def test_wrapper_closes_session_after_failed_send(session_module, engines):
    engine = engines()
    wrapped, _ = _wrap(session_module, [engine.url])
    with pytest.raises(RuntimeError, match="broadcast failed"):
        wrapped(_Updater([engine.url], fail=True), None)
    assert engine.calls[-1] == "end_weight_update"
    assert not engine.session_open


def test_non_owner_rank_only_waits_on_barrier(session_module, engines):
    engine = engines()
    wrapped, barriers = _wrap(session_module, [engine.url], owner=False)
    wrapped(_Updater([]), None)
    assert barriers == [True]
    assert engine.calls == []


def test_legacy_engine_skips_end(session_module, engines):
    legacy = engines(has_sessions=False)
    current = engines()
    urls = [legacy.url, current.url]
    wrapped, _ = _wrap(session_module, urls)
    wrapped(_Updater(urls), None)
    assert legacy.calls == [
        "begin_weight_update",
        "update_weights_from_distributed",
    ]
    assert current.calls[-1] == "end_weight_update"


def test_begin_failure_closes_already_opened_sessions(session_module, engines):
    good = engines()
    bad = engines(fail_begin=True)
    with pytest.raises(RuntimeError, match="begin_weight_update failed"):
        session_module.begin_weight_update([good.url, bad.url])
    assert good.calls == ["begin_weight_update", "end_weight_update"]
    assert not good.session_open


@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        ("engine-0:30000", "http://engine-0:30000"),
        ("http://engine-0:30000/", "http://engine-0:30000"),
        ("[::1]:30000", "http://[::1]:30000"),
    ],
)
def test_engine_base_url(session_module, addr, expected):
    assert session_module.engine_base_url(addr) == expected


def test_engine_base_url_rejects_missing_port(session_module):
    with pytest.raises(ValueError):
        session_module.engine_base_url("engine-0")


def _fake_slime(monkeypatch):
    class UpdateWeightFromDistributed:
        def _send_weights(self, pbar):
            return pbar

    modules = {
        "slime": types.ModuleType("slime"),
        "slime.backends": types.ModuleType("slime.backends"),
        "slime.backends.megatron_utils": types.ModuleType(
            "slime.backends.megatron_utils"
        ),
        "slime.backends.megatron_utils.update_weight": types.ModuleType(
            "slime.backends.megatron_utils.update_weight"
        ),
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed": types.ModuleType(
            "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
        ),
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.distributed_utils": types.ModuleType(
            "slime.utils.distributed_utils"
        ),
    }
    modules[
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    ].UpdateWeightFromDistributed = UpdateWeightFromDistributed
    modules["slime.utils.distributed_utils"].get_gloo_group = lambda: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.get_rank = lambda: 1
    fake_dist.barrier = lambda group=None: None
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_dist
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    return UpdateWeightFromDistributed


def _args(**overrides):
    values = {
        "rollout_external": True,
        "rollout_external_engine_addrs": ["engine-0:30000"],
        "update_weight_transport": "nccl",
        "update_weight_mode": "full",
        "colocate": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_install_wraps_nccl_updater_once(session_module, monkeypatch):
    updater_cls = _fake_slime(monkeypatch)
    original = updater_cls._send_weights

    session_module.install(_args())
    wrapped = updater_cls._send_weights
    assert wrapped is not original
    session_module.install(_args())
    assert updater_cls._send_weights is wrapped
    # The fake rank is not 0, so no HTTP call is made.
    assert updater_cls()._send_weights("pbar") == "pbar"


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_external": False},
        {"update_weight_transport": "disk"},
        {"update_weight_mode": "delta"},
        {"colocate": True},
    ],
)
def test_install_skips_paths_without_distributed_updates(
    session_module, monkeypatch, overrides
):
    updater_cls = _fake_slime(monkeypatch)
    original = updater_cls._send_weights
    session_module.install(_args(**overrides))
    assert updater_cls._send_weights is original
