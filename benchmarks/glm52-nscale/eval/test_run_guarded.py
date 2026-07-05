#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Sequence

from runtime_binding import BindingError, canonical_sha256, validate_continuity


EVAL_DIR = Path(__file__).resolve().parent
SCRIPT = EVAL_DIR / "run-guarded.sh"
REMOTE_DRIVER = EVAL_DIR / "remote-command-driver.py"
FIXTURE = EVAL_DIR / "fixtures" / "runtime-binding.json"

FAKE_KUBECTL = r"""#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


remote_root = Path(os.environ["FAKE_REMOTE_ROOT"])
state_path = Path(os.environ["FAKE_KUBE_STATE"])
call_state_path = Path(os.environ["FAKE_KUBE_CALL_STATE"])
args = sys.argv[1:]
if len(args) >= 2 and args[0] == "--context":
    args = args[2:]
if args and args[0].startswith("--request-timeout="):
    args = args[1:]


def remote(path):
    if not path.startswith("/"):
        raise RuntimeError(f"expected absolute remote path, got {path!r}")
    return remote_root / path.removeprefix("/")


def state():
    return json.loads(state_path.read_text())


if args[:2] == ["get", "pod"]:
    call_state = json.loads(call_state_path.read_text())
    call_state["pod_get_calls"] += 1
    call_state_path.write_text(json.dumps(call_state))
    if call_state["fail_pod_get_on_call"] == call_state["pod_get_calls"]:
        raise SystemExit(1)
    print("{}")
    raise SystemExit(0)
if args[:2] == ["get", "pods"]:
    print(json.dumps(state()["pods"]))
    raise SystemExit(0)
if len(args) >= 2 and args[0] == "get" and args[1] in {
    "deployment",
    "dynamographdeployment",
}:
    print(json.dumps(state()["controller"]))
    raise SystemExit(0)
if not args or args[0] != "exec" or "--" not in args:
    raise RuntimeError(f"unsupported fake kubectl invocation: {args!r}")

command = args[args.index("--") + 1 :]
if command[:2] == ["python3", "/workspace/eval/remote-command-driver.py"]:
    driver_args = command[2:]
    operation = driver_args[0]
    state_index = driver_args.index("--state-dir") + 1
    driver_args[state_index] = str(remote(driver_args[state_index]))
    completed = subprocess.run(
        [sys.executable, os.environ["FAKE_REMOTE_DRIVER"], *driver_args],
        text=True,
        capture_output=True,
        env=os.environ,
        check=False,
    )
    call_state = json.loads(call_state_path.read_text())
    if operation == "acquire":
        call_state["acquire_calls"] += 1
        if call_state["drop_acquire_after_apply"] > 0:
            call_state["drop_acquire_after_apply"] -= 1
            call_state_path.write_text(json.dumps(call_state))
            raise SystemExit(255)
    if operation == "start":
        call_state["start_calls"] += 1
        if call_state["drop_start_after_launch"] > 0:
            call_state["drop_start_after_launch"] -= 1
            call_state_path.write_text(json.dumps(call_state))
            raise SystemExit(255)
    if operation == "status" and call_state["status_failures"] > 0:
        call_state["status_failures"] -= 1
        call_state_path.write_text(json.dumps(call_state))
        raise SystemExit(255)
    if operation == "status" and call_state["status_malformed"] > 0:
        call_state["status_malformed"] -= 1
        call_state_path.write_text(json.dumps(call_state))
        print("{")
        raise SystemExit(0)
    call_state_path.write_text(json.dumps(call_state))
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    raise SystemExit(completed.returncode)
if command[:2] == ["python3", "-c"] or command[:2] == ["docker", "ps"]:
    raise SystemExit(0)
if command[:2] == ["tmux", "list-sessions"]:
    raise SystemExit(1)
if command[:2] == ["/bin/bash", "-c"]:
    owner = remote("/artifacts/glm52-nscale/.campaign-run.lock/owner.json")
    if owner.is_file():
        sys.stdout.write(owner.read_text())
    raise SystemExit(0)
if command[:1] == ["mkdir"]:
    parents = command[1:2] == ["-p"]
    target = remote(command[-1])
    try:
        target.mkdir(parents=parents, exist_ok=parents)
    except FileExistsError:
        raise SystemExit(1)
    raise SystemExit(0)
if command[:2] == ["rm", "-rf"]:
    shutil.rmtree(remote(command[2]), ignore_errors=True)
    raise SystemExit(0)
if command[:2] == ["rm", "-f"]:
    remote(command[2]).unlink(missing_ok=True)
    raise SystemExit(0)
if command[:1] == ["cat"]:
    sys.stdout.write(remote(command[1]).read_text())
    raise SystemExit(0)
if command[:1] == ["sha256sum"]:
    import hashlib
    path = remote(command[1])
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {command[1]}")
    raise SystemExit(0)
if command[:1] == ["mv"]:
    source = remote(command[1])
    destination = remote(command[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    raise SystemExit(0)
if command[:2] == ["test", "-e"]:
    raise SystemExit(0 if remote(command[2]).exists() else 1)
if command[:3] == ["test", "!", "-e"]:
    raise SystemExit(1 if remote(command[3]).exists() else 0)
if command[:2] == ["/bin/bash", "-eu"]:
    target = remote(command[-1])
    payload = sys.stdin.read()
    if "owner.json" in command[3]:
        target.mkdir(parents=True, exist_ok=True)
        (target / "owner.json").write_text(payload)
    elif "pre.json" in command[3]:
        target.mkdir(parents=True, exist_ok=True)
        (target / "pre.json").write_text(payload)
    else:
        json.loads(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(payload)
        temporary.replace(target)
        call_state = json.loads(call_state_path.read_text())
        if call_state["drop_attestation_after_write"] > 0:
            call_state["drop_attestation_after_write"] -= 1
            call_state_path.write_text(json.dumps(call_state))
            raise SystemExit(255)
    raise SystemExit(0)
raise RuntimeError(f"unsupported fake remote command: {command!r}")
"""


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class GuardedRunIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        kubectl = fake_bin / "kubectl"
        kubectl.write_text(FAKE_KUBECTL)
        kubectl.chmod(0o755)
        self.state_path = self.root / "state.json"
        self.call_state_path = self.root / "call-state.json"
        self.call_state_path.write_text(
            json.dumps(
                {
                    "drop_acquire_after_apply": 0,
                    "drop_attestation_after_write": 0,
                    "drop_start_after_launch": 0,
                    "status_failures": 0,
                    "status_malformed": 0,
                    "acquire_calls": 0,
                    "fail_pod_get_on_call": 0,
                    "pod_get_calls": 0,
                    "start_calls": 0,
                }
            )
        )
        self.marker = self.root / "command-ran"
        self.binding = self._make_binding()
        binding_path = (
            self.remote
            / "artifacts/glm52-nscale/runtime-bindings/dynamo-vllm/active.json"
        )
        binding_path.parent.mkdir(parents=True)
        binding_path.write_text(json.dumps(self.binding))
        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{self.env['PATH']}",
                "FAKE_REMOTE_ROOT": str(self.remote),
                "FAKE_KUBE_STATE": str(self.state_path),
                "FAKE_KUBE_CALL_STATE": str(self.call_state_path),
                "FAKE_COMMAND_MARKER": str(self.marker),
                "FAKE_REMOTE_DRIVER": str(REMOTE_DRIVER),
                "KUBE_CONTEXT": "synthetic-context",
                "NAMESPACE": "synthetic-namespace",
                "EVAL_RUNNER_POD": "synthetic-runner",
                "GLM52_GUARD_POLL_SECONDS": "0.01",
                "GLM52_GUARD_RETRY_SECONDS": "0.01",
                "GLM52_GUARD_REQUEST_TIMEOUT": "5s",
                "GLM52_REMOTE_COMMAND_ROOT": str(
                    self.remote / "artifacts/glm52-nscale/.campaign-run.lock"
                ),
                "TMPDIR": str(self.root),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_binding(self) -> dict[str, object]:
        controller_uid = "private-controller-uid"
        frontend = {
            "role": "frontend",
            "name": "private-frontend-pod",
            "uid": "private-frontend-uid",
            "node": "private-cpu-node",
            "image_id": (
                "nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly@sha256:"
                "c3336583c830ea5c3cf4bd5cc92cb57200b8f558398a18c3ac0f473f9b74dd1d"
            ),
            "container_id": "containerd://" + "1" * 64,
        }
        worker = {
            "role": "worker",
            "name": "private-worker-pod",
            "uid": "private-worker-uid",
            "node": "private-gpu-node",
            "image_id": (
                "nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly@sha256:"
                "67c4e55999cfd1f79cb5b5d59fcc20def55e6f465f819f4dbdb69c08613a6b4c"
            ),
            "container_id": "containerd://" + "2" * 64,
        }

        def pod(value: dict[str, str]) -> dict[str, object]:
            return {
                "metadata": {
                    "name": value["name"],
                    "uid": value["uid"],
                    "labels": {"glm52.nvidia.com/role": value["role"]},
                },
                "spec": {
                    "nodeName": value["node"],
                    "containers": [{"name": "main"}],
                },
                "status": {
                    "containerStatuses": [
                        {
                            "name": "main",
                            "imageID": value["image_id"],
                            "containerID": value["container_id"],
                            "restartCount": 0,
                        }
                    ]
                },
            }

        state = {
            "controller": {
                "metadata": {
                    "name": "glm52-dynamo-vllm",
                    "uid": controller_uid,
                    "generation": 1,
                }
            },
            "pods": {"items": [pod(frontend), pod(worker)]},
        }
        self.state_path.write_text(json.dumps(state))
        binding = json.loads(FIXTURE.read_text())
        binding["controller"]["uid_sha256"] = digest(controller_uid)
        for runtime in (frontend, worker):
            role = runtime["role"]
            binding["pods"][role].update(
                {
                    "name_sha256": digest(runtime["name"]),
                    "uid_sha256": digest(runtime["uid"]),
                    "node_name_sha256": digest(runtime["node"]),
                    "image_id": runtime["image_id"],
                }
            )
        return binding

    def _set_call_state(self, **changes: int) -> None:
        value = json.loads(self.call_state_path.read_text())
        value.update(changes)
        self.call_state_path.write_text(json.dumps(value))

    def _run(
        self,
        command: Sequence[str],
        *,
        phase: str = "ab",
        suffix: str = "case",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        attestation = (
            f"/artifacts/glm52-nscale/integration/{suffix}/runtime-continuity.json"
        )
        result = subprocess.run(
            [
                str(SCRIPT),
                "dynamo-vllm",
                "--phase",
                phase,
                "--attestation",
                attestation,
                "--",
                *command,
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
        )
        return result, self.remote / attestation.removeprefix("/")

    def test_success_attests_canonical_private_stable_runtime(self) -> None:
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"], suffix="success"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        continuity = json.loads(path.read_text())
        validate_continuity(continuity, self.binding)
        self.assertEqual(
            continuity["deployment_sha256"], canonical_sha256(self.binding)
        )
        self.assertLessEqual(
            continuity["pre_captured_at"], continuity["post_captured_at"]
        )
        serialized = path.read_text()
        state = json.loads(self.state_path.read_text())
        raw_values = [state["controller"]["metadata"]["uid"]]
        for pod in state["pods"]["items"]:
            raw_values.extend(
                [
                    pod["metadata"]["name"],
                    pod["metadata"]["uid"],
                    pod["spec"]["nodeName"],
                    pod["status"]["containerStatuses"][0]["containerID"],
                ]
            )
        for raw_value in raw_values:
            self.assertNotIn(raw_value, serialized)

    def test_lost_start_response_retries_without_duplicate_execution(self) -> None:
        marker = self.root / "detached-launches"
        self._set_call_state(drop_start_after_launch=1)
        result, path = self._run(
            [
                sys.executable,
                "-c",
                "import sys, time; "
                "open(sys.argv[1], 'a', encoding='utf-8').write('launch\\n'); "
                "time.sleep(0.2)",
                str(marker),
            ],
            suffix="lost-start-response",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        validate_continuity(json.loads(path.read_text()), self.binding)
        self.assertEqual(marker.read_text().splitlines(), ["launch"])
        call_state = json.loads(self.call_state_path.read_text())
        self.assertGreaterEqual(call_state["start_calls"], 2)

    def test_lost_lock_acquisition_response_retries_same_owner(self) -> None:
        self._set_call_state(drop_acquire_after_apply=1)
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"],
            suffix="lost-lock-response",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        validate_continuity(json.loads(path.read_text()), self.binding)
        call_state = json.loads(self.call_state_path.read_text())
        self.assertGreaterEqual(call_state["acquire_calls"], 2)

    def test_transient_status_failures_preserve_remote_exit_code(self) -> None:
        self._set_call_state(status_failures=1, status_malformed=1)
        result, path = self._run(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(0.2); raise SystemExit(7)",
            ],
            suffix="status-retry",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        continuity = json.loads(path.read_text())
        self.assertEqual(continuity["command_exit_code"], 7)
        validate_continuity(continuity, self.binding, require_success=False)

    def test_remote_exit_255_is_not_a_transport_failure(self) -> None:
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(255)"],
            suffix="remote-exit-255",
        )
        self.assertEqual(result.returncode, 255, result.stderr)
        continuity = json.loads(path.read_text())
        self.assertEqual(continuity["command_exit_code"], 255)
        validate_continuity(continuity, self.binding, require_success=False)

    def test_attestation_response_loss_retains_terminal_state(self) -> None:
        self._set_call_state(drop_attestation_after_write=1)
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"],
            suffix="attestation-response-loss",
        )
        self.assertNotEqual(result.returncode, 0)
        lock = self.remote / "artifacts/glm52-nscale/.campaign-run.lock"
        self.assertTrue(lock.is_dir())
        self.assertTrue(path.is_file())
        owner = json.loads((lock / "owner.json").read_text())
        driver_status = json.loads((lock / "command/status.json").read_text())
        self.assertEqual(driver_status["exit_code"], 0)

        released = subprocess.run(
            [
                sys.executable,
                str(REMOTE_DRIVER),
                "release",
                "--state-dir",
                str(lock / "command"),
                "--invocation-id",
                owner["invocation_id"],
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        self.assertEqual(json.loads(released.stdout)["state"], "released")

    def test_postflight_runner_failure_retains_terminal_state(self) -> None:
        self._set_call_state(fail_pod_get_on_call=2)
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"],
            suffix="postflight-runner-failure",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Evaluation runner", result.stderr)
        self.assertFalse(path.exists())
        lock = self.remote / "artifacts/glm52-nscale/.campaign-run.lock"
        self.assertTrue(lock.is_dir())
        owner = json.loads((lock / "owner.json").read_text())
        released = subprocess.run(
            [
                sys.executable,
                str(REMOTE_DRIVER),
                "release",
                "--state-dir",
                str(lock / "command"),
                "--invocation-id",
                owner["invocation_id"],
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        self.assertEqual(json.loads(released.stdout)["state"], "released")

    def test_terminated_local_guard_retains_lock_and_remote_command(self) -> None:
        marker = self.root / "remote-command-started"
        attestation = (
            "/artifacts/glm52-nscale/integration/killed-guard/runtime-continuity.json"
        )
        command = [
            sys.executable,
            "-c",
            "import sys, time; from pathlib import Path; "
            "Path(sys.argv[1]).touch(); time.sleep(60)",
            str(marker),
        ]
        guard = subprocess.Popen(
            [
                str(SCRIPT),
                "dynamo-vllm",
                "--phase",
                "ab",
                "--attestation",
                attestation,
                "--",
                *command,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        lock = self.remote / "artifacts/glm52-nscale/.campaign-run.lock"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        self.assertTrue(marker.exists(), "detached command did not start")
        guard.terminate()
        guard.communicate(timeout=10)

        self.assertTrue(lock.is_dir())
        self.assertFalse((self.remote / attestation.removeprefix("/")).exists())
        blocked, _ = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"],
            suffix="blocked-by-killed-guard",
        )
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("Evaluation runner is active", blocked.stderr)

        owner = json.loads((lock / "owner.json").read_text())
        probed = subprocess.run(
            [
                sys.executable,
                str(REMOTE_DRIVER),
                "status",
                "--state-dir",
                str(lock / "command"),
                "--invocation-id",
                owner["invocation_id"],
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        self.assertEqual(json.loads(probed.stdout)["state"], "running")
        completed = subprocess.run(
            [
                sys.executable,
                str(REMOTE_DRIVER),
                "terminate",
                "--state-dir",
                str(lock / "command"),
                "--invocation-id",
                owner["invocation_id"],
                "--timeout",
                "5",
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout)["state"], "finished")
        shutil.rmtree(lock)

    def test_explicit_phase_mismatch_fails_before_command(self) -> None:
        result, path = self._run(
            [
                sys.executable,
                "-c",
                "import os; from pathlib import Path; "
                "Path(os.environ['FAKE_COMMAND_MARKER']).touch()",
            ],
            phase="ba",
            suffix="wrong-phase",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("campaign_phase", result.stderr)
        self.assertFalse(self.marker.exists())
        self.assertFalse(path.exists())

    def test_failed_command_gets_structurally_valid_attestation(self) -> None:
        result, path = self._run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            suffix="command-failure",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        continuity = json.loads(path.read_text())
        validate_continuity(continuity, self.binding, require_success=False)
        with self.assertRaisesRegex(BindingError, "did not exit zero"):
            validate_continuity(continuity, self.binding)

    def test_retry_archives_failed_attestation_but_never_overwrites_success(
        self,
    ) -> None:
        failed, path = self._run(
            [sys.executable, "-c", "raise SystemExit(7)"], suffix="retry"
        )
        self.assertEqual(failed.returncode, 7, failed.stderr)
        failed_payload = path.read_bytes()
        failed_digest = hashlib.sha256(failed_payload).hexdigest()

        retried, same_path = self._run(
            [sys.executable, "-c", "raise SystemExit(0)"], suffix="retry"
        )
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertEqual(same_path, path)
        validate_continuity(json.loads(path.read_text()), self.binding)
        archived = (
            path.with_suffix("").with_name(f"{path.stem}.failures")
            / f"{failed_digest}.json"
        )
        self.assertEqual(archived.read_bytes(), failed_payload)

        refused, _ = self._run(
            [
                sys.executable,
                "-c",
                "import os; from pathlib import Path; "
                "Path(os.environ['FAKE_COMMAND_MARKER']).touch()",
            ],
            suffix="retry",
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("successful runtime attestation", refused.stderr)
        self.assertFalse(self.marker.exists())

    def test_runtime_restart_fails_without_publishing_attestation(self) -> None:
        result, path = self._run(
            [
                sys.executable,
                "-c",
                "import json, os; from pathlib import Path; "
                "path=Path(os.environ['FAKE_KUBE_STATE']); "
                "document=json.loads(path.read_text()); "
                "status=document['pods']['items'][1]['status']['containerStatuses'][0]; "
                "status['restartCount']=1; "
                "status['containerID']='containerd://'+'9'*64; "
                "path.write_text(json.dumps(document))",
            ],
            suffix="restart",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("zero restarts after", result.stderr)
        self.assertFalse(path.exists())

    def test_container_replacement_fails_without_publishing_attestation(self) -> None:
        result, path = self._run(
            [
                sys.executable,
                "-c",
                "import json, os; from pathlib import Path; "
                "path=Path(os.environ['FAKE_KUBE_STATE']); "
                "document=json.loads(path.read_text()); "
                "status=document['pods']['items'][1]['status']['containerStatuses'][0]; "
                "status['containerID']='containerd://'+'8'*64; "
                "path.write_text(json.dumps(document))",
            ],
            suffix="replacement",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime identity changed", result.stderr)
        self.assertFalse(path.exists())

    def test_dot_component_attestation_path_is_rejected(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "dynamo-vllm",
                "--phase",
                "ab",
                "--attestation",
                "/artifacts/glm52-nscale/integration/../runtime-continuity.json",
                "--",
                "touch-marker",
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
