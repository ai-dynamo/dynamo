#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from runtime_binding import BindingError, canonical_sha256, validate_continuity


EVAL_DIR = Path(__file__).resolve().parent
SCRIPT = EVAL_DIR / "run-guarded.sh"
FIXTURE = EVAL_DIR / "fixtures" / "runtime-binding.json"

FAKE_KUBECTL = r"""#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path


remote_root = Path(os.environ["FAKE_REMOTE_ROOT"])
state_path = Path(os.environ["FAKE_KUBE_STATE"])
args = sys.argv[1:]
if len(args) >= 2 and args[0] == "--context":
    args = args[2:]


def remote(path):
    if not path.startswith("/"):
        raise RuntimeError(f"expected absolute remote path, got {path!r}")
    return remote_root / path.removeprefix("/")


def state():
    return json.loads(state_path.read_text())


if args[:2] == ["get", "pod"]:
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
    else:
        json.loads(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(payload)
        temporary.replace(target)
    raise SystemExit(0)
if command == ["succeed"]:
    raise SystemExit(0)
if command == ["fail-seven"]:
    raise SystemExit(7)
if command == ["touch-marker"]:
    Path(os.environ["FAKE_COMMAND_MARKER"]).touch()
    raise SystemExit(0)
if command == ["mutate-runtime"]:
    document = state()
    status = document["pods"]["items"][1]["status"]["containerStatuses"][0]
    status["restartCount"] = 1
    status["containerID"] = "containerd://" + "9" * 64
    state_path.write_text(json.dumps(document))
    raise SystemExit(0)
if command == ["replace-container"]:
    document = state()
    status = document["pods"]["items"][1]["status"]["containerStatuses"][0]
    status["containerID"] = "containerd://" + "8" * 64
    state_path.write_text(json.dumps(document))
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
                "FAKE_COMMAND_MARKER": str(self.marker),
                "KUBE_CONTEXT": "synthetic-context",
                "NAMESPACE": "synthetic-namespace",
                "EVAL_RUNNER_POD": "synthetic-runner",
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

    def _run(
        self,
        command: str,
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
                command,
            ],
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
        )
        return result, self.remote / attestation.removeprefix("/")

    def test_success_attests_canonical_private_stable_runtime(self) -> None:
        result, path = self._run("succeed", suffix="success")
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

    def test_explicit_phase_mismatch_fails_before_command(self) -> None:
        result, path = self._run("touch-marker", phase="ba", suffix="wrong-phase")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("campaign_phase", result.stderr)
        self.assertFalse(self.marker.exists())
        self.assertFalse(path.exists())

    def test_failed_command_gets_structurally_valid_attestation(self) -> None:
        result, path = self._run("fail-seven", suffix="command-failure")
        self.assertEqual(result.returncode, 7, result.stderr)
        continuity = json.loads(path.read_text())
        validate_continuity(continuity, self.binding, require_success=False)
        with self.assertRaisesRegex(BindingError, "did not exit zero"):
            validate_continuity(continuity, self.binding)

    def test_retry_archives_failed_attestation_but_never_overwrites_success(
        self,
    ) -> None:
        failed, path = self._run("fail-seven", suffix="retry")
        self.assertEqual(failed.returncode, 7, failed.stderr)
        failed_payload = path.read_bytes()
        failed_digest = hashlib.sha256(failed_payload).hexdigest()

        retried, same_path = self._run("succeed", suffix="retry")
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertEqual(same_path, path)
        validate_continuity(json.loads(path.read_text()), self.binding)
        archived = (
            path.with_suffix("").with_name(f"{path.stem}.failures")
            / f"{failed_digest}.json"
        )
        self.assertEqual(archived.read_bytes(), failed_payload)

        refused, _ = self._run("touch-marker", suffix="retry")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("successful runtime attestation", refused.stderr)
        self.assertFalse(self.marker.exists())

    def test_runtime_restart_fails_without_publishing_attestation(self) -> None:
        result, path = self._run("mutate-runtime", suffix="restart")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("zero restarts after", result.stderr)
        self.assertFalse(path.exists())

    def test_container_replacement_fails_without_publishing_attestation(self) -> None:
        result, path = self._run("replace-container", suffix="replacement")
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
