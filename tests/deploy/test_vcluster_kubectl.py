# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise vCluster tunnel shell helpers without a Kubernetes cluster or real delays."""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
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
def sleep_stub(tmp_path, monkeypatch):
    sleep = tmp_path / "sleep"
    sleep.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$DELAYS"\n')
    sleep.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DELAYS", str(tmp_path / "delays"))
    return tmp_path


@pytest.fixture
def kubectl_stub(sleep_stub, monkeypatch):
    executable = sleep_stub / "kubectl"
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
    monkeypatch.setenv("CALLS", str(sleep_stub / "calls.json"))
    return sleep_stub


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


# Stands in for the port-forward that connect-vcluster hands to the watchdog.
# It reports ready once its signal disposition is in place.
PORT_FORWARD = """\
import signal, sys
if "ignore-sigterm" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
while True:
    signal.pause()
"""

# Each port-forward the watchdog starts records its PID and arguments.
WATCHDOG_KUBECTL = """\
import json, os, signal, sys
with open(os.environ["STARTS"], "a") as starts:
    starts.write(json.dumps({"pid": os.getpid(), "args": sys.argv[1:]}) + "\\n")
while True:
    signal.pause()
"""

# Answers each health probe with the next outcome in OUTCOMES, then holds the
# watchdog until the test stops it.
WATCHDOG_CURL = """\
import json, os, pathlib, signal, sys
args = sys.argv[1:]
state = pathlib.Path(os.environ["PROBES"])
probes = json.loads(state.read_text()) if state.exists() else []
probes.append(args)
state.write_text(json.dumps(probes))
outcomes = json.loads(os.environ["OUTCOMES"])
if len(probes) > len(outcomes):
    pathlib.Path(os.environ["DONE"]).touch()
    while True:
        signal.pause()
outcome = outcomes[len(probes) - 1]
if outcome == "slow":
    # The TLS handshake through the loaded tunnel takes 3s.
    outcome = "ok" if float(args[args.index("--max-time") + 1]) > 3 else "timeout"
errors = {
    "timeout": (28, "Operation timed out"),
    "refused": (7, "Failed to connect to 127.0.0.1 port 8443: Connection refused"),
}
if outcome in errors:
    code, message = errors[outcome]
    print(f"curl: ({code}) {message}", file=sys.stderr)
    sys.exit(code)
"""


@pytest.fixture
def watchdog_stubs(sleep_stub, monkeypatch):
    for name, source in (("kubectl", WATCHDOG_KUBECTL), ("curl", WATCHDOG_CURL)):
        executable = sleep_stub / name
        executable.write_text(f"#!{sys.executable}\n{source}")
        executable.chmod(0o755)
    monkeypatch.setenv("STARTS", str(sleep_stub / "starts"))
    monkeypatch.setenv("PROBES", str(sleep_stub / "probes.json"))
    monkeypatch.setenv("DONE", str(sleep_stub / "done"))
    return sleep_stub


@pytest.fixture
def port_forward(watchdog_stubs):
    processes = []

    def start(*args):
        process = subprocess.Popen(
            [sys.executable, "-c", PORT_FORWARD, *args],
            stdout=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        assert process.stdout.readline() == "ready\n"
        (watchdog_stubs / "port-forward.pid").write_text(f"{process.pid}\n")
        return process

    yield start
    for process in processes:
        process.kill()
        process.communicate()


def wait_for(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def port_forward_starts(stubs):
    starts = stubs / "starts"
    if not starts.exists():
        return []
    return [json.loads(line) for line in starts.read_text().splitlines()]


def is_zombie(pid):
    return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def run_watchdog(stubs, outcomes, **settings):
    """Run the watchdog until curl has answered every scripted probe."""
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/actions/connect-vcluster/port-forward-watchdog.sh"
    )
    pid_file = stubs / "port-forward.pid"
    initial_pid = int(pid_file.read_text())
    env = shell_environment()
    env["OUTCOMES"] = json.dumps(outcomes)
    # The stub sleep returns at once, so give a port-forward that honors
    # SIGTERM plenty of polls to exit before the watchdog escalates to SIGKILL.
    env["VCLUSTER_WATCHDOG_STOP_TIMEOUT"] = "1000"
    env.update(settings)

    def settled():
        if watchdog.poll() is not None:
            return True
        if not (stubs / "done").exists():
            return False
        # A replacement port-forward records its start in the background.
        pid = int(pid_file.read_text())
        return pid == initial_pid or pid in {
            start["pid"] for start in port_forward_starts(stubs)
        }

    with open(stubs / "watchdog.log", "w") as log:
        watchdog = subprocess.Popen(
            ["bash", str(script), str(pid_file), "vcluster-ns", "vcluster"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            finished = wait_for(settled)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(watchdog.pid, signal.SIGKILL)
            watchdog.wait()
    output = (stubs / "watchdog.log").read_text()
    # The watchdog runs until it is killed.
    assert finished and watchdog.returncode == -signal.SIGKILL, output
    return output


@pytest.mark.parametrize(
    "outcomes, failures",
    [
        # A probe slower than the former 1s connect timeout passes, and a
        # single failed probe leaves a running port-forward alone.
        (["slow", "timeout", "ok"], ["1/3"]),
        # A successful probe resets the count of consecutive failures.
        (
            ["timeout", "timeout", "ok", "timeout", "timeout"],
            ["1/3", "2/3", "1/3", "2/3"],
        ),
    ],
    ids=["slow-probe", "intermediate-success"],
)
def test_watchdog_keeps_running_port_forward(
    port_forward, watchdog_stubs, outcomes, failures
):
    running = port_forward()

    log = run_watchdog(watchdog_stubs, outcomes)

    assert re.findall(r"Probe failed \((\d+/\d+)\)", log) == failures
    assert "Watchdog restarting" not in log
    assert port_forward_starts(watchdog_stubs) == []
    assert running.poll() is None
    assert (watchdog_stubs / "port-forward.pid").read_text() == f"{running.pid}\n"
    probes = json.loads((watchdog_stubs / "probes.json").read_text())
    assert all(
        probe[probe.index("--max-time") + 1] == "10"
        and probe[-1] == "https://127.0.0.1:8443/healthz"
        for probe in probes
    )
    assert (watchdog_stubs / "delays").read_text().split() == ["3"] * len(outcomes)


@pytest.mark.parametrize("reaped", [True, False], ids=["reaped", "zombie"])
def test_watchdog_restarts_exited_port_forward(port_forward, watchdog_stubs, reaped):
    exited = port_forward()
    exited.kill()
    if reaped:
        exited.wait()
    else:
        # Once the connect-vcluster step ends, its port-forward may go unreaped.
        assert wait_for(lambda: is_zombie(exited.pid))

    log = run_watchdog(watchdog_stubs, ["refused", "ok"])

    assert (
        f"Watchdog restarting vCluster API port-forward: pid {exited.pid} exited" in log
    )
    assert "Probe failed" not in log
    [start] = port_forward_starts(watchdog_stubs)
    assert start["args"] == [
        "port-forward",
        "-n",
        "vcluster-ns",
        "svc/vcluster",
        "8443:443",
    ]
    assert (watchdog_stubs / "port-forward.pid").read_text() == f"{start['pid']}\n"


@pytest.mark.parametrize(
    "args, settings, stopped_by",
    [
        ((), {}, signal.SIGTERM),
        # Escalate when the port-forward outlasts the bounded wait after SIGTERM.
        (("ignore-sigterm",), {"VCLUSTER_WATCHDOG_STOP_TIMEOUT": "20"}, signal.SIGKILL),
    ],
    ids=["sigterm", "sigkill"],
)
def test_watchdog_replaces_unresponsive_port_forward(
    port_forward, watchdog_stubs, args, settings, stopped_by
):
    unresponsive = port_forward(*args)

    log = run_watchdog(watchdog_stubs, ["timeout"] * 3 + ["ok"], **settings)

    assert re.findall(r"Probe failed \((\d+/\d+)\)", log) == ["1/3", "2/3", "3/3"]
    assert (
        "Watchdog restarting vCluster API port-forward: "
        f"3 consecutive failed probes with pid {unresponsive.pid} running"
    ) in log
    assert ("sending SIGKILL" in log) == (stopped_by == signal.SIGKILL)
    # The replacement started only after the old port-forward had exited.
    assert "starting a replacement anyway" not in log
    assert unresponsive.poll() == -stopped_by
    [start] = port_forward_starts(watchdog_stubs)
    assert (watchdog_stubs / "port-forward.pid").read_text() == f"{start['pid']}\n"
