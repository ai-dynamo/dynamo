# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shell retries without a Kubernetes cluster or real retry delays."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.fixture(autouse=True, params=[False, True], ids=["clean", "bash-startup"])
def shell_startup_environment(request, tmp_path, monkeypatch):
    for name in ("BASH_ENV", "ENV"):
        monkeypatch.delenv(name, raising=False)
    if request.param:
        startup = tmp_path / "bashrc"
        startup.write_text('echo "${PS1:?unexpected shell startup}"\nexit 99\n')
        monkeypatch.delenv("PS1", raising=False)
        monkeypatch.setenv("BASH_ENV", str(startup))
        monkeypatch.setenv("ENV", str(startup))


def shell_environment():
    # CI images may source interactive configuration through BASH_ENV. These
    # subprocesses test standalone scripts and must not execute that startup code.
    return {k: v for k, v in os.environ.items() if k not in {"BASH_ENV", "ENV"}}


@pytest.fixture
def kubectl_stub(tmp_path, monkeypatch):
    executable = tmp_path / "kubectl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "state = pathlib.Path(os.environ['CALLS'])\n"
        "calls = json.loads(state.read_text()) if state.exists() else []\n"
        "args = sys.argv[1:]\n"
        "payload = pathlib.Path(args[args.index('-f') + 1]).read_text() if '-f' in args else None\n"
        "calls.append({'args': args, 'payload': payload})\n"
        "state.write_text(json.dumps(calls))\n"
        "if len(calls) <= int(os.environ['FAILURES']):\n"
        "    print(os.environ['ERROR'], file=sys.stderr)\n"
        "    sys.exit(7)\n"
        "print('ok')\n"
    )
    executable.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$DELAYS"\n')
    sleep.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CALLS", str(tmp_path / "calls.json"))
    monkeypatch.setenv("DELAYS", str(tmp_path / "delays"))
    return tmp_path


def run_retry(*args):
    script = Path(__file__).resolve().parents[2] / ".github/scripts/retry_kubectl.sh"
    return subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'source "$1"; shift; retry_kubectl "$@"',
            "test",
            str(script),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=5,
        env=shell_environment(),
    )


@pytest.mark.parametrize(
    "error",
    [
        "Unable to connect to the server: EOF",
        "The connection to the server 127.0.0.1:8443 was refused - did you specify the right host or port?",
        "Unable to connect to the server: read: connection reset by peer",
        "Unable to connect to the server: net/http: TLS handshake timeout",
    ],
)
def test_recovers_connection_failure(kubectl_stub, monkeypatch, error):
    monkeypatch.setenv("ERROR", error)
    monkeypatch.setenv("FAILURES", "1")
    manifest = kubectl_stub / "manifest.yaml"
    manifest.write_text("kind: Namespace\n")

    result = run_retry("apply", "-f", str(manifest))

    assert result.returncode == 0, result.stderr
    calls = json.loads((kubectl_stub / "calls.json").read_text())
    assert len(calls) == 2
    assert all(call["payload"] == manifest.read_text() for call in calls)
    assert all("--request-timeout=30s" in call["args"] for call in calls)
    assert (kubectl_stub / "delays").read_text() == "5\n"


def test_stops_after_three_retries(kubectl_stub, monkeypatch):
    monkeypatch.setenv("ERROR", "Unable to connect to the server: EOF")
    monkeypatch.setenv("FAILURES", "10")

    result = run_retry("get", "namespaces")

    assert result.returncode == 7
    assert len(json.loads((kubectl_stub / "calls.json").read_text())) == 4
    assert (kubectl_stub / "delays").read_text() == "5\n5\n5\n"


@pytest.mark.parametrize(
    "error",
    [
        "Error from server (Forbidden): namespaces is forbidden",
        "error: error parsing manifest: unexpected EOF",
        "error: timed out waiting for the condition",
    ],
)
def test_does_not_retry_non_connection_errors(kubectl_stub, monkeypatch, error):
    monkeypatch.setenv("ERROR", error)
    monkeypatch.setenv("FAILURES", "10")

    result = run_retry("get", "namespaces")

    assert result.returncode == 7
    assert len(json.loads((kubectl_stub / "calls.json").read_text())) == 1
    assert not (kubectl_stub / "delays").exists()


def test_collects_logs_without_live_tunnel(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".vcluster-port-forward.log").write_text("tunnel lost\n")
    (workspace / ".vcluster-port-forward-watchdog.log").write_text("restart\n")
    (workspace / ".kubeconfig-vcluster").write_text("credentials must not be uploaded")
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 7\n")
    curl.chmod(0o755)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    output = tmp_path / "diagnostics"
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/scripts/collect_vcluster_diagnostics.sh"
    )

    result = subprocess.run(
        ["bash", str(script), str(output)],
        capture_output=True,
        text=True,
        timeout=5,
        env=shell_environment(),
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "vcluster-port-forward-status.log",
        "vcluster-port-forward-watchdog.log",
        "vcluster-port-forward.log",
    ]
    assert (output / "vcluster-port-forward.log").read_text() == "tunnel lost\n"
    assert (
        "healthz_unreachable=true"
        in (output / "vcluster-port-forward-status.log").read_text()
    )
