# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("watchdog.sh")


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pf_pid_file = self.root / "port-forward.pid"
        self.pf_log = self.root / "port-forward.log"
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        self.temp_dir.cleanup()

    def run_shell(self, body, extra_env=None, timeout=5):
        env = dict(
            os.environ,
            PF_PID_FILE=str(self.pf_pid_file),
            PF_LOG=str(self.pf_log),
            NAMESPACE="test-namespace",
            VCLUSTER_NAME="test-vcluster",
            FAILURE_THRESHOLD="3",
            STOP_WAIT_ATTEMPTS="2",
            PORT_RELEASE_ATTEMPTS="2",
            WAIT_INTERVAL="0",
        )
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", "-c", f'source "{SCRIPT}"\n{body}'],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def make_executable(self, name, body):
        executable = self.root / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/usr/bin/env bash\n" + body)
        executable.chmod(0o755)
        return executable

    def wait_for_path(self, path, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return True
            time.sleep(0.02)
        return path.exists()

    def make_fake_flock(self):
        return self.make_executable(
            "bin/flock",
            """exec python3 - "$@" <<'PY'
import fcntl
import sys

try:
    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(1)
PY
""",
        )

    def watchdog_env(self, watchdog_pid_file, lock_file):
        return dict(
            os.environ,
            PF_PID_FILE=str(self.pf_pid_file),
            PF_LOG=str(self.pf_log),
            WATCHDOG_LOG=str(self.root / "watchdog.log"),
            WATCHDOG_PID_FILE=str(watchdog_pid_file),
            WATCHDOG_LOCK_FILE=str(lock_file),
            NAMESPACE="test-namespace",
            VCLUSTER_NAME="test-vcluster",
            STOP_WAIT_ATTEMPTS="10",
            WAIT_INTERVAL="0.1",
            WATCHDOG_INTERVAL="0.1",
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
        )

    def start_process(self, name, *args, body=None):
        executable = self.make_executable(
            name,
            body or "trap 'exit 0' TERM\nwhile :; do sleep 1; done\n",
        )
        process = subprocess.Popen([str(executable), *args])
        self.processes.append(process)
        time.sleep(0.05)
        return process

    def test_pid_must_be_an_integer_greater_than_one(self):
        result = self.run_shell(
            """
for value in "" 0 1 -1 12x "2 3"; do
    if valid_pid "${value}"; then
        echo "accepted:${value}"
        exit 1
    fi
done
valid_pid 2
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_process_identity_rejects_a_reused_pid(self):
        unrelated = self.start_process("unrelated")
        result = self.run_shell(
            f"""
if process_is_port_forward {unrelated.pid}; then
    echo "accepted unrelated process"
    exit 1
fi
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIsNone(unrelated.poll())

    def test_process_identity_accepts_the_expected_port_forward(self):
        port_forward = self.start_process(
            "kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        result = self.run_shell(f"process_is_port_forward {port_forward.pid}")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_process_identity_accepts_linux_ps_without_executable_path(self):
        port_forward = self.start_process("linux-ps-process")
        self.make_executable(
            "bin/ps",
            (
                'echo "kubectl port-forward -n test-namespace '
                'svc/test-vcluster 8443:443"\n'
            ),
        )
        result = self.run_shell(
            f"process_is_port_forward {port_forward.pid}",
            {"PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"]},
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_success_resets_the_consecutive_failure_count(self):
        result = self.run_shell(
            """
failure_count=0
record_probe_result 28 || true
printf '%s\n' "${failure_count}"
record_probe_result 0 || true
printf '%s\n' "${failure_count}"
record_probe_result 28 || true
printf '%s\n' "${failure_count}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["1", "0", "1"])

    def test_replacement_requires_three_consecutive_failures(self):
        result = self.run_shell(
            """
failure_count=0
record_probe_result 28 && exit 1
record_probe_result 28 && exit 1
record_probe_result 28
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_pid_file_change_prevents_stopping_the_observed_process(self):
        observed = self.start_process(
            "observed/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        replacement = self.start_process(
            "replacement/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        self.pf_pid_file.write_text(f"{replacement.pid}\n")

        result = self.run_shell(f"stop_port_forward {observed.pid}")

        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(observed.poll())
        self.assertIsNone(replacement.poll())

    def test_pid_file_change_prevents_sigkill_escalation(self):
        term_ready = self.root / "escalation-term-ready"
        observed = self.start_process(
            "escalation-observed/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
            body=f"trap '' TERM\ntouch '{term_ready}'\nwhile :; do sleep 1; done\n",
        )
        replacement = self.start_process(
            "escalation-replacement/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        self.assertTrue(self.wait_for_path(term_ready))
        self.pf_pid_file.write_text(f"{observed.pid}\n")
        result = self.run_shell(
            f"""
(sleep 0.1; printf '%s\n' {replacement.pid} > "${{PF_PID_FILE}}") &
stop_port_forward {observed.pid}
""",
            {"STOP_WAIT_ATTEMPTS": "3", "WAIT_INTERVAL": "0.1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(observed.poll())
        self.assertIsNone(replacement.poll())

    def test_dead_candidate_is_not_ready_when_another_listener_is_healthy(self):
        candidate = self.start_process(
            "candidate/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        candidate.terminate()
        candidate.wait(timeout=2)

        result = self.run_shell(
            f"""
declare -F wait_for_candidate_ready >/dev/null || exit 99
probe_health() {{ return 0; }}
if wait_for_candidate_ready {candidate.pid} 0; then
    echo "dead candidate reported ready"
    exit 1
fi
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_unknown_listener_is_not_killed_or_replaced(self):
        kubectl_started = self.root / "kubectl-started"
        self.make_executable(
            "bin/kubectl",
            f'echo "$$" > "{kubectl_started}"\nexit 1\n',
        )
        env = {"PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"]}

        result = self.run_shell(
            """
declare -F recover_unmanaged_port_forward >/dev/null || exit 99
probe_health() { return 22; }
port_is_free() { return 1; }
if recover_unmanaged_port_forward; then
    echo "unknown listener reported recovered"
    exit 1
fi
""",
            env,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertFalse(kubectl_started.exists())

    def test_readiness_timeout_cleans_candidate_without_publishing_it(self):
        kubectl_started = self.root / "kubectl-started"
        self.make_executable(
            "bin/kubectl",
            f"""
echo "$$" > "{kubectl_started}"
trap 'exit 0' TERM
while :; do sleep 1; done
""",
        )
        env = {
            "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"],
            "READINESS_ATTEMPTS": "2",
            "WAIT_INTERVAL": "1",
        }

        result = self.run_shell(
            """
declare -F replace_port_forward >/dev/null || exit 99
probe_health() { return 28; }
if replace_port_forward; then
    echo "unready candidate reported recovered"
    exit 1
fi
""",
            env,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        candidate_pid = int(kubectl_started.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(candidate_pid, 0)
        self.assertFalse(self.pf_pid_file.exists())

    def test_address_conflict_cleans_a_still_running_candidate(self):
        kubectl_started = self.root / "kubectl-started"
        self.make_executable(
            "bin/kubectl",
            f"""
echo "$$" > "{kubectl_started}"
echo "Unable to listen on port 8443: address already in use"
trap 'exit 0' TERM
while :; do sleep 1; done
""",
        )
        env = {
            "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"],
            "READINESS_ATTEMPTS": "3",
            "WAIT_INTERVAL": "1",
        }

        result = self.run_shell(
            """
declare -F replace_port_forward >/dev/null || exit 99
probe_health() { return 0; }
if replace_port_forward; then
    echo "conflicting candidate reported ready"
    exit 1
fi
""",
            env,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        candidate_pid = int(kubectl_started.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(candidate_pid, 0)
        self.assertFalse(self.pf_pid_file.exists())

    def test_candidate_pid_is_published_only_after_readiness(self):
        kubectl_started = self.root / "kubectl-started"
        ready = self.root / "ready"
        self.make_executable(
            "bin/kubectl",
            f"""
echo "$$" > "{kubectl_started}"
trap 'exit 0' TERM
while :; do sleep 1; done
""",
        )
        env = dict(
            os.environ,
            PF_PID_FILE=str(self.pf_pid_file),
            PF_LOG=str(self.pf_log),
            NAMESPACE="test-namespace",
            VCLUSTER_NAME="test-vcluster",
            READINESS_ATTEMPTS="50",
            STOP_WAIT_ATTEMPTS="2",
            WAIT_INTERVAL="0.2",
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
            READY_FILE=str(ready),
        )
        body = f"""
source "{SCRIPT}"
declare -F replace_port_forward >/dev/null || exit 99
probe_health() {{ [ -f "${{READY_FILE}}" ]; }}
replace_port_forward
"""
        replacement = subprocess.Popen(
            ["bash", "-c", body],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.processes.append(replacement)
        self.assertTrue(self.wait_for_path(kubectl_started))
        candidate_pid = int(kubectl_started.read_text())
        self.assertFalse(self.pf_pid_file.exists())

        ready.touch()
        stdout, stderr = replacement.communicate(timeout=12)
        self.assertEqual(replacement.returncode, 0, stderr + stdout)
        self.assertEqual(self.pf_pid_file.read_text().strip(), str(candidate_pid))
        self.assertEqual(list(self.root.glob("port-forward.pid.tmp.*")), [])
        os.kill(candidate_pid, 15)

    def test_candidate_is_rechecked_immediately_before_pid_publication(self):
        candidate = self.start_process(
            "publish-race/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        result = self.run_shell(
            f"""
start_candidate() {{
    candidate_pid={candidate.pid}
    candidate_log_start=0
}}
wait_for_candidate_ready() {{
    kill -TERM {candidate.pid}
    while kill -0 {candidate.pid} 2>/dev/null; do
        state=$(ps -p {candidate.pid} -o stat= 2>/dev/null || true)
        case "${{state}}" in *Z*) break ;; esac
    done
    return 0
}}
if replace_port_forward; then
    echo "dead candidate was published"
    exit 1
fi
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        candidate.wait(timeout=2)
        self.assertFalse(self.pf_pid_file.exists())

    def test_term_timeout_escalates_only_for_the_same_process(self):
        term_ready = self.root / "term-ready"
        port_forward = self.start_process(
            "stubborn/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
            body=f"trap '' TERM\ntouch '{term_ready}'\nwhile :; do sleep 1; done\n",
        )
        self.assertTrue(self.wait_for_path(term_ready))
        self.pf_pid_file.write_text(f"{port_forward.pid}\n")

        result = self.run_shell(f"stop_port_forward {port_forward.pid}")

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        port_forward.wait(timeout=2)
        self.assertEqual(port_forward.returncode, -9)

    def test_live_port_forward_restarts_only_after_three_failed_probes(self):
        port_forward = self.start_process(
            "managed/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        self.pf_pid_file.write_text(f"{port_forward.pid}\n")

        result = self.run_shell(
            """
declare -F watchdog_iteration >/dev/null || exit 99
probe_health() { return 28; }
stop_port_forward() { echo "stop:$1"; }
wait_for_port_release() { return 0; }
replace_port_forward() { echo "replace"; }
watchdog_iteration
watchdog_iteration
watchdog_iteration
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(
            result.stdout.splitlines(),
            [f"stop:{port_forward.pid}", "replace"],
        )

    def test_successful_probe_resets_the_live_port_forward_threshold(self):
        port_forward = self.start_process(
            "managed-reset/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        self.pf_pid_file.write_text(f"{port_forward.pid}\n")

        result = self.run_shell(
            """
declare -F watchdog_iteration >/dev/null || exit 99
probe_number=0
probe_health() {
    probe_number=$((probe_number + 1))
    case "${probe_number}" in
        2) return 0 ;;
        *) return 28 ;;
    esac
}
stop_port_forward() { echo "unexpected stop"; return 1; }
watchdog_iteration
watchdog_iteration
watchdog_iteration
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stdout, "")
        self.assertIsNone(port_forward.poll())

    def test_dead_managed_pid_recovers_without_waiting_for_probe_threshold(self):
        port_forward = self.start_process(
            "dead/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        port_forward.terminate()
        port_forward.wait(timeout=2)
        self.pf_pid_file.write_text(f"{port_forward.pid}\n")

        result = self.run_shell(
            """
declare -F watchdog_iteration >/dev/null || exit 99
probe_health() { return 7; }
port_is_free() { return 0; }
replace_port_forward() { echo "replace"; }
watchdog_iteration
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stdout.splitlines(), ["replace"])

    def test_reused_pid_is_not_signaled_before_recovery(self):
        unrelated = self.start_process("reused/unrelated")
        self.pf_pid_file.write_text(f"{unrelated.pid}\n")

        result = self.run_shell(
            """
declare -F watchdog_iteration >/dev/null || exit 99
probe_health() { return 7; }
port_is_free() { return 0; }
replace_port_forward() { echo "replace"; }
watchdog_iteration
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stdout.splitlines(), ["replace"])
        self.assertIsNone(unrelated.poll())

    def test_stopping_watchdog_leaves_sibling_port_forward_for_teardown(self):
        watchdog_pid_file = self.root / "watchdog.pid"
        lock_file = self.root / "watchdog.lock"
        self.make_executable("bin/curl", "exit 0\n")
        self.make_fake_flock()
        port_forward = self.start_process(
            "sibling/kubectl",
            "port-forward",
            "-n",
            "test-namespace",
            "svc/test-vcluster",
            "8443:443",
        )
        self.pf_pid_file.write_text(f"{port_forward.pid}\n")
        env = self.watchdog_env(watchdog_pid_file, lock_file)
        watchdog = subprocess.Popen([str(SCRIPT), "run"], env=env)
        self.processes.append(watchdog)
        self.assertTrue(self.wait_for_path(watchdog_pid_file))

        result = subprocess.run(
            [str(SCRIPT), "stop-watchdog"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        watchdog.wait(timeout=2)
        self.assertIsNone(port_forward.poll())
        self.assertEqual(self.pf_pid_file.read_text().strip(), str(port_forward.pid))

        result = subprocess.run(
            [str(SCRIPT), "stop-port-forward"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        port_forward.wait(timeout=2)
        self.assertFalse(self.pf_pid_file.exists())

    def test_stopping_watchdog_reaps_its_published_replacement(self):
        watchdog_pid_file = self.root / "watchdog.pid"
        lock_file = self.root / "watchdog.lock"
        health_count = self.root / "health-count"
        self.make_executable(
            "bin/curl",
            f"""
case " $* " in
    *" -fsk "*)
        count=0
        [ ! -f "{health_count}" ] || count=$(cat "{health_count}")
        count=$((count + 1))
        echo "${{count}}" > "{health_count}"
        if [ "${{count}}" -eq 1 ]; then
            exit 7
        fi
        ;;
    *) exit 7 ;;
esac
""",
        )
        self.make_executable(
            "bin/kubectl",
            "trap 'exit 0' TERM\nwhile :; do sleep 1; done\n",
        )
        self.make_fake_flock()
        env = self.watchdog_env(watchdog_pid_file, lock_file)
        env["READINESS_TIMEOUT"] = "10"
        watchdog = subprocess.Popen([str(SCRIPT), "run"], env=env)
        self.processes.append(watchdog)
        self.assertTrue(self.wait_for_path(self.pf_pid_file))
        candidate_pid = int(self.pf_pid_file.read_text())

        result = subprocess.run(
            [str(SCRIPT), "stop-watchdog"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        watchdog.wait(timeout=2)
        with self.assertRaises(ProcessLookupError):
            os.kill(candidate_pid, 0)
        self.assertFalse(self.pf_pid_file.exists())

    def test_only_one_watchdog_acquires_the_lock(self):
        watchdog_pid_file = self.root / "watchdog.pid"
        lock_file = self.root / "watchdog.lock"
        curl = self.make_executable("bin/curl", "exit 0\n")
        self.make_fake_flock()
        env = self.watchdog_env(watchdog_pid_file, lock_file)
        env["PATH"] = str(curl.parent) + os.pathsep + os.environ["PATH"]
        first = subprocess.Popen([str(SCRIPT), "run"], env=env)
        second = subprocess.Popen([str(SCRIPT), "run"], env=env)
        self.processes.extend([first, second])

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if watchdog_pid_file.exists() and (
                first.poll() is not None or second.poll() is not None
            ):
                break
            time.sleep(0.02)

        self.assertTrue(watchdog_pid_file.exists())
        owner = int(watchdog_pid_file.read_text())
        self.assertIn(owner, [first.pid, second.pid])
        self.assertEqual(sum(process.poll() is None for process in [first, second]), 1)

    def test_watchdog_exits_when_its_pid_cannot_be_published(self):
        self.make_executable("bin/curl", "exit 0\n")
        self.make_fake_flock()
        missing = self.root / "missing"
        env = self.watchdog_env(missing / "watchdog.pid", self.root / "lock")
        env["WATCHDOG_MAX_ITERATIONS"] = "1"

        result = subprocess.run(
            [str(SCRIPT), "run"],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((missing / "watchdog.pid").exists())


if __name__ == "__main__":
    unittest.main()
