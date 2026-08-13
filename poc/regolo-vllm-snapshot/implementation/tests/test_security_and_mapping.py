import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = ROOT / "implementation/lib/snapshot_poc.py"
AUDIT = ROOT / "implementation/bin/audit-no-secrets"


def load_library():
    spec = importlib.util.spec_from_file_location("snapshot_poc", LIB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointIdentityTests(unittest.TestCase):
    def test_checkpoint_id_is_label_safe_and_deterministic(self):
        value = "a" * 64
        result = load_library().checkpoint_id(value)
        self.assertEqual(result, "h-" + "a" * 61)
        self.assertEqual(len(result), 63)

    def test_checkpoint_id_rejects_noncanonical_hashes(self):
        function = load_library().checkpoint_id
        for value in (None, "a" * 63, "a" * 65, "A" * 64, "g" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                function(value)

    def test_create_checkpoint_rejects_mismatched_locator_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = root / "pod.json"
            manifest.write_text("{}\n")
            result = subprocess.run(
                [
                    str(ROOT / "implementation/bin/create-checkpoint"),
                    "--snapshotctl", str(root / "must-not-run"),
                    "--manifest", str(manifest),
                    "--namespace", "test",
                    "--compatibility-hash", "a" * 64,
                    "--checkpoint-id", "h-" + "b" * 61,
                    "--output", str(root / "result.json"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)
            self.assertFalse((root / "result.json").exists())

    def test_create_checkpoint_passes_safe_single_gpu_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log = root / "commands.log"
            manifest = root / "pod.json"
            manifest.write_text("{}\n")
            snapshotctl = root / "snapshotctl"
            snapshotctl.write_text(
                "#!/bin/sh\n"
                "printf 'snapshotctl %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
                "printf '%s\\n' 'checkpoint_location=/checkpoints/test/versions/1'\n"
            )
            kubectl = root / "kubectl"
            kubectl.write_text(
                "#!/bin/sh\n"
                "printf 'kubectl %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
                "case \"$*\" in\n"
                "  *'get pod'*) printf '%s' 'snapshot-agent-test' ;;\n"
                "  *'exec snapshot-agent-test'*) printf '%s\\n' '43886233878 /checkpoints/test/versions/1' ;;\n"
                "esac\n"
            )
            for executable in (snapshotctl, kubectl):
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            compatibility_hash = "a" * 64
            locator = "h-" + "a" * 61
            env = os.environ.copy()
            env.update(PATH=f"{root}:{env['PATH']}", COMMAND_LOG=str(log))
            result = subprocess.run(
                [
                    str(ROOT / "implementation/bin/create-checkpoint"),
                    "--snapshotctl", str(snapshotctl),
                    "--manifest", str(manifest),
                    "--namespace", "test",
                    "--compatibility-hash", compatibility_hash,
                    "--checkpoint-id", locator,
                    "--output", str(root / "result.json"),
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            first = log.read_text().splitlines()[0]
            self.assertIn(f"--checkpoint-id {locator}", first)
            self.assertIn("--disable-cuda-checkpoint-job-file", first)
            self.assertIn("--timeout 45m", first)
            metadata = json.loads((root / "result.json").read_text())
            self.assertEqual(metadata["compatibility_hash"], compatibility_hash)
            self.assertEqual(metadata["checkpoint_id"], locator)

    def test_campaign_rejects_mismatched_checkpoint_metadata_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = {name: root / name for name in ("campaign", "plan", "key", "checkpoint")}
            paths["campaign"].write_text(json.dumps({"compatibility_hash": "a" * 64}))
            paths["plan"].write_text("[]")
            paths["key"].write_text(json.dumps({"A": "cold", "B": "restore"}))
            paths["checkpoint"].write_text(
                json.dumps({"compatibility_hash": "b" * 64, "checkpoint_id": "h-" + "b" * 61})
            )
            result = subprocess.run(
                [
                    str(ROOT / "implementation/bin/run-campaign"),
                    "--campaign", str(paths["campaign"]),
                    "--run-plan", str(paths["plan"]),
                    "--key", str(paths["key"]),
                    "--checkpoint-metadata", str(paths["checkpoint"]),
                    "--output-dir", str(root / "runs"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match checkpoint metadata", result.stderr)
            self.assertFalse((root / "runs").exists())

    def test_campaign_rejects_noncanonical_locator_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            compatibility_hash = "a" * 64
            campaign = root / "campaign.json"
            checkpoint = root / "checkpoint.json"
            plan = root / "plan.json"
            key = root / "key.json"
            campaign.write_text(
                json.dumps({"compatibility_hash": compatibility_hash, "checkpoint_id": "h-" + "b" * 61})
            )
            checkpoint.write_text(
                json.dumps({"compatibility_hash": compatibility_hash, "checkpoint_id": "h-" + "a" * 61})
            )
            plan.write_text("[]")
            key.write_text(json.dumps({"A": "cold", "B": "restore"}))
            result = subprocess.run(
                [
                    str(ROOT / "implementation/bin/run-campaign"),
                    "--campaign", str(campaign),
                    "--run-plan", str(plan),
                    "--key", str(key),
                    "--checkpoint-metadata", str(checkpoint),
                    "--output-dir", str(root / "runs"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not the Kubernetes-safe compatibility mapping", result.stderr)
            self.assertFalse((root / "runs").exists())


class PodSecurityContractTests(unittest.TestCase):
    def test_template_disables_token_and_uses_http_readiness_and_shared_cache(self):
        manifest = json.loads((ROOT / "implementation/k8s/pod.template.json").read_text())
        self.assertIs(manifest["spec"]["automountServiceAccountToken"], False)
        self.assertFalse(any("projected" in volume for volume in manifest["spec"]["volumes"]))
        container = manifest["spec"]["containers"][0]
        self.assertEqual(container["readinessProbe"]["httpGet"], {"path": "/health", "port": 8000})
        mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}
        self.assertEqual(mounts["model-cache"], "/root/.cache/huggingface")
        volumes = {volume["name"]: volume for volume in manifest["spec"]["volumes"]}
        self.assertEqual(volumes["model-cache"]["hostPath"]["type"], "Directory")


class PlaceholderPatchTests(unittest.TestCase):
    def test_patcher_rewrites_pinned_upstream_shape(self):
        patcher = ROOT / "implementation/bin/patch-placeholder-dockerfile"
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = pathlib.Path(directory) / "Dockerfile"
            dockerfile.write_text(
                "FROM ubuntu:24.04 AS criu-builder\n"
                "RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint\n"
                "RUN x && make DESTDIR=/criu-install install-criu install-lib install-cuda_plugin\n"
                "FROM ${BASE_IMAGE} AS placeholder\n"
                "RUN apt-get install libgnutls30t64\n"
            )
            result = subprocess.run(
                [str(patcher), str(dockerfile), "0" * 40], text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            body = dockerfile.read_text()
            self.assertIn("FROM ${BASE_IMAGE} AS criu-builder", body)
            self.assertIn("git fetch --depth 1 origin " + "0" * 40, body)
            self.assertIn("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python", body)
            self.assertIn("libgnutls30", body)
            self.assertNotIn("libgnutls30t64", body)

    def test_patcher_fails_closed_on_upstream_drift(self):
        patcher = ROOT / "implementation/bin/patch-placeholder-dockerfile"
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = pathlib.Path(directory) / "Dockerfile"
            dockerfile.write_text("FROM changed-upstream\n")
            result = subprocess.run(
                [str(patcher), str(dockerfile), "0" * 40], text=True, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected upstream", result.stderr)


class SecretAuditTests(unittest.TestCase):
    def run_audit(self, automount=False, history="safe", inspect="ok"):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = root / "pod.json"
            spec = {"containers": [{"name": "server", "image": "example@sha256:" + "a" * 64}]}
            if automount is not None:
                spec["automountServiceAccountToken"] = automount
            manifest.write_text(json.dumps({"spec": spec}))
            docker = root / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                "if [ \"$1 $2\" = \"image inspect\" ]; then\n"
                "  [ \"$FAKE_INSPECT\" = fail ] && exit 7\n"
                "  printf '%s\\n' '[{\"Id\":\"sha256:test\",\"Config\":{\"Env\":[\"PATH=/bin\"]}}]'\n"
                "  exit 0\n"
                "fi\n"
                "if [ \"$1\" = history ]; then\n"
                "  [ \"$FAKE_HISTORY\" = fail ] && exit 8\n"
                "  [ \"$FAKE_HISTORY\" = secret ] && printf '%s\\n' 'RUN API_KEY=exposed build'\n"
                "  [ \"$FAKE_HISTORY\" = safe ] && printf '%s\\n' 'RUN install tokenizer'\n"
                "  exit 0\n"
                "fi\n"
                "exit 9\n"
            )
            docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env.update(PATH=f"{root}:{env['PATH']}", FAKE_HISTORY=history, FAKE_INSPECT=inspect)
            result = subprocess.run(
                [str(AUDIT), "--manifest", str(manifest), "--image", "example@sha256:" + "a" * 64],
                text=True,
                capture_output=True,
                env=env,
            )
            return result, json.loads(result.stdout)

    def test_clean_manifest_and_successful_history_pass(self):
        result, report = self.run_audit()
        self.assertEqual(result.returncode, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["image"], "example@sha256:" + "a" * 64)
        self.assertTrue(report["docker_inspect_passed"])
        self.assertTrue(report["docker_history_passed"])

    def test_automount_must_be_explicitly_false(self):
        for automount in (None, True):
            with self.subTest(automount=automount):
                result, report = self.run_audit(automount=automount)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(any("automountServiceAccountToken" in item[0] for item in report["findings"]))

    def test_docker_inspect_and_history_fail_closed(self):
        inspect_result, inspect_report = self.run_audit(inspect="fail")
        self.assertNotEqual(inspect_result.returncode, 0)
        self.assertFalse(inspect_report["docker_inspect_passed"])
        history_result, history_report = self.run_audit(history="fail")
        self.assertNotEqual(history_result.returncode, 0)
        self.assertTrue(history_report["docker_inspect_passed"])
        self.assertFalse(history_report["docker_history_passed"])

    def test_history_secret_assignment_fails_but_name_only_text_does_not(self):
        secret_result, secret_report = self.run_audit(history="secret")
        self.assertNotEqual(secret_result.returncode, 0)
        self.assertTrue(any(item[0] == "image.history" for item in secret_report["findings"]))
        safe_result, safe_report = self.run_audit(history="safe")
        self.assertEqual(safe_result.returncode, 0)
        self.assertTrue(safe_report["passed"])


if __name__ == "__main__":
    unittest.main()
