"""Offline contract for the production-only V2 run wiring CLI."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _support import load_harness


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
V2RUN = V2_ROOT / "harness" / "v2run.py"
AGENT = "snapshot-agent"


def load_v2run():
    if not V2RUN.is_file():
        raise FileNotFoundError(f"missing required V2 production runner: {V2RUN}")
    spec = importlib.util.spec_from_file_location("v2run_under_test", V2RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V2RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V2RunTests(unittest.TestCase):
    def setUp(self):
        self.module = load_v2run()
        self.harness = load_harness()
        self.lane = json.loads((V2_ROOT / "lane.json").read_text())
        self.plan = self.harness.make_paired_blinded_plan(self.lane)

    @staticmethod
    def write_json(root, name, value):
        path = pathlib.Path(root) / name
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return path

    def inputs(self, directory):
        root = pathlib.Path(directory)
        lane = self.write_json(root, "lane.json", self.lane)
        schedule, key = self.harness.seal_plan(self.plan, root / "plan")
        authorization = self.write_json(root, "authorization.json", {"execution_authorized": True})
        snapshotctl = root / "snapshotctl"
        snapshotctl.write_bytes(b"pinned snapshotctl\n")
        checkpoint_id = "h-" + self.lane["identity"]["compatibility_hash"][:61]
        inventory_list = [
            {"path": "manifest.yaml", "size": 8740},
            {"path": "pages-12.img", "size": 300},
            {"path": "rootfs-diff.tar", "size": 80},
        ]
        checkpoint = {
            "checkpoint": {
                "id": checkpoint_id,
                "compatibility_hash": self.lane["identity"]["compatibility_hash"],
                "location": "/checkpoints/" + checkpoint_id + "/versions/1",
                "total_size_bytes": 9120,
                "pages_12_size_bytes": 300,
                "rootfs_size_bytes": 80,
                "metadata_size_bytes": 8740,
                "manifest_sha256": "b" * 64,
                "inventory": {
                    "regular_file_count": 3,
                    "regular_file_size_bytes": 9120,
                    "inventory_sha256": hashlib.sha256(json.dumps(inventory_list, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                },
            },
            "agent": {
                "namespace": "v2-live-test", "name": AGENT, "uid": "agent-uid",
                "image": "registry.example/snapshot-agent@sha256:" + "c" * 64,
                "node": self.lane["identity"]["node"],
            },
            "pvc": {"name": "checkpoint-pvc", "uid": "pvc-uid", "pv": "checkpoint-pv"},
            "pv": {
                "uid": "pv-uid", "local_path": "/mnt/regolo-vllm-snapshot-luks/checkpoints",
                "claim_uid": "pvc-uid", "node": self.lane["identity"]["node"], "reclaim_policy": "Retain",
            },
        }
        checkpoint_path = self.write_json(root, "checkpoint-attestation.json", checkpoint)
        campaign = {
            "namespace": "v2-live-test", "node": self.lane["identity"]["node"],
            "snapshotctl": str(snapshotctl), "snapshotctl_sha256": hashlib.sha256(snapshotctl.read_bytes()).hexdigest(),
            "checkpoint": {
                "checkpoint_id": checkpoint_id,
                "compatibility_hash": self.lane["identity"]["compatibility_hash"],
                "attestation_path": str(checkpoint_path),
                "attestation_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            },
        }
        campaign_path = self.write_json(root, "campaign.json", campaign)
        cluster = {
            "namespace": "v2-live-test", "node": self.lane["identity"]["node"],
            "agent": {
                "name": AGENT, "uid": "agent-uid",
                "image": "registry.example/snapshot-agent@sha256:" + "c" * 64,
                "node": self.lane["identity"]["node"],
            },
            "reserves": [
                {"name": "gpu-reserve-1", "uid": "reserve-uid-1", "container": "server",
                 "image": "registry.example/reserve@sha256:" + "1" * 64,
                 "node": self.lane["identity"]["node"], "gpu_uuid": "GPU-1"},
                {"name": "gpu-reserve-2", "uid": "reserve-uid-2", "container": "server",
                 "image": "registry.example/reserve@sha256:" + "2" * 64,
                 "node": self.lane["identity"]["node"], "gpu_uuid": "GPU-2"},
                {"name": "gpu-reserve-3", "uid": "reserve-uid-3", "container": "server",
                 "image": "registry.example/reserve@sha256:" + "3" * 64,
                 "node": self.lane["identity"]["node"], "gpu_uuid": "GPU-3"},
            ],
        }
        cluster_path = self.write_json(root, "cluster-attestation.json", cluster)
        ledger = root / "results.jsonl"
        ledger.touch(mode=0o600)
        return {
            "--lane": lane, "--auth": authorization, "--schedule": schedule, "--key": key,
            "--ledger": ledger, "--campaign": campaign_path, "--artifact-dir": root / "artifacts",
            "--cluster-attestation": cluster_path,
            "--cluster-attestation-sha": hashlib.sha256(cluster_path.read_bytes()).hexdigest(),
            "--agent": AGENT, "--limit": "2",
        }

    @staticmethod
    def argv(values, *, dry_run=False):
        result = [item for pair in values.items() for item in (pair[0], str(pair[1]))]
        if dry_run:
            result.append("--dry-run")
        return result

    def test_wires_validated_inputs_to_production_dependencies_and_runs_limited_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.inputs(directory)
            transport = mock.Mock(name="transport")
            validator = mock.Mock(name="validator")
            preflight = mock.Mock(name="preflight")
            collector = mock.Mock(name="collector")
            advisor = mock.Mock(name="advisor")
            runner = mock.Mock(name="runner")
            runner.run.return_value = {"completed": ["v2-01-01", "v2-01-02"]}
            transport_cls = mock.Mock(return_value=transport)
            validator_cls = mock.Mock(return_value=validator)
            preflight_cls = mock.Mock(return_value=preflight)
            collector_cls = mock.Mock(return_value=collector)
            advisor_cls = mock.Mock(return_value=advisor)
            runner_cls = mock.Mock(return_value=runner)
            stdout = io.StringIO()
            with mock.patch.multiple(
                self.module,
                SubprocessTransport=transport_cls, ProductionCheckpointValidator=validator_cls,
                ProductionClusterPreflight=preflight_cls, ProductionCollector=collector_cls,
                CacheAdvisor=advisor_cls, LiveRunner=runner_cls,
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(self.module.main(self.argv(values)), 0)
            transport_cls.assert_called_once_with(timeout_s=1800)
            validator_cls.assert_called_once()
            preflight_cls.assert_called_once()
            collector_cls.assert_called_once()
            self.assertEqual(collector_cls.call_args.kwargs["timeout_s"], 1800)
            advisor_cls.assert_called_once()
            live_kwargs = runner_cls.call_args.kwargs
            self.assertIs(live_kwargs["command_runner"], transport)
            self.assertIs(live_kwargs["checkpoint_validator"], validator)
            self.assertIs(live_kwargs["cluster_preflight"], preflight)
            self.assertIs(live_kwargs["collector"], collector)
            self.assertIs(live_kwargs["fadvise"], advisor.advise_inventory)
            self.assertEqual(live_kwargs["artifact_dir"], values["--artifact-dir"])
            self.assertRegex(live_kwargs["execution_digest"], r"^[0-9a-f]{64}$")
            self.assertEqual(live_kwargs.get("checkpoint_files", ()), ())
            self.assertEqual(
                collector_cls.call_args.args[3],
                {
                    "checkpoint_id": "h-" + self.lane["identity"]["compatibility_hash"][:61],
                    "compatibility_hash": self.lane["identity"]["compatibility_hash"],
                    "checkpoint_size_bytes": 9120,
                    "pages_12_size_bytes": 300,
                    "rootfs_size_bytes": 80,
                    "metadata_size_bytes": 8740,
                    "checkpoint_inventory": {
                        "regular_file_count": 3,
                        "regular_file_size_bytes": 9120,
                        "inventory_sha256": hashlib.sha256(json.dumps([
                            {"path": "manifest.yaml", "size": 8740},
                            {"path": "pages-12.img", "size": 300},
                            {"path": "rootfs-diff.tar", "size": 80},
                        ], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    },
                },
            )
            runner.run.assert_called_once_with(limit=2)
            self.assertEqual(
                stdout.getvalue(),
                json.dumps(runner.run.return_value, sort_keys=True, separators=(",", ":")) + "\n",
            )

    def test_runtime_failure_returns_one_with_concise_v2run_stderr_and_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.inputs(directory)
            runner = mock.Mock(name="runner")
            runner.run.side_effect = RuntimeError("collector failed after start")
            stderr = io.StringIO()
            with mock.patch.multiple(
                self.module,
                SubprocessTransport=mock.Mock(return_value=mock.Mock()),
                ProductionCheckpointValidator=mock.Mock(), ProductionClusterPreflight=mock.Mock(),
                ProductionCollector=mock.Mock(), CacheAdvisor=mock.Mock(), LiveRunner=mock.Mock(return_value=runner),
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(self.module.main(self.argv(values)), 1)
            self.assertEqual(stderr.getvalue(), "v2run: collector failed after start\n")
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_rejects_malformed_digest_mismatched_missing_or_symlinked_local_inputs_before_transport(self):
        for mutation in ("malformed", "digest", "missing", "symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                values = self.inputs(directory)
                target = values["--cluster-attestation"]
                if mutation == "malformed":
                    target.write_text("[]")
                elif mutation == "digest":
                    values["--cluster-attestation-sha"] = "0" * 64
                elif mutation == "missing":
                    target.unlink()
                else:
                    outside = pathlib.Path(directory) / "outside.json"
                    outside.write_text(target.read_text())
                    target.unlink()
                    os.symlink(outside, target)
                with mock.patch.object(self.module, "SubprocessTransport") as transport:
                    self.assertNotEqual(self.module.main(self.argv(values)), 0)
                transport.assert_not_called()

    def test_dry_run_does_not_invoke_the_external_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.inputs(directory)
            transport = mock.Mock(name="transport")
            runner = mock.Mock(name="runner")
            runner.run.return_value = {"commands": []}
            runner_cls = mock.Mock(return_value=runner)
            with mock.patch.multiple(
                self.module,
                SubprocessTransport=mock.Mock(return_value=transport),
                ProductionCheckpointValidator=mock.Mock(), ProductionClusterPreflight=mock.Mock(),
                ProductionCollector=mock.Mock(), CacheAdvisor=mock.Mock(), LiveRunner=runner_cls,
            ):
                self.assertEqual(self.module.main(self.argv(values, dry_run=True)), 0)
            transport.assert_not_called()
            self.assertTrue(runner_cls.call_args.kwargs["dry_run"])
            runner.run.assert_called_once_with(limit=2)

    def test_sha256s_seal_tamper_aborts_before_transport_and_execution_digest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.inputs(directory)
            seal_root = pathlib.Path(directory) / "sealed-repository"
            sealed_file = seal_root / "verification" / "v2" / "README.md"
            sealed_file.parent.mkdir(parents=True)
            shutil.copyfile(V2_ROOT / "README.md", sealed_file)
            line = next(line for line in (V2_ROOT / "SHA256SUMS").read_text().splitlines() if line.endswith("  verification/v2/README.md"))
            (seal_root / "verification" / "v2" / "SHA256SUMS").write_text(line + "\n")
            sealed_file.write_text("tampered after sealing\n")
            transport_cls = mock.Mock(return_value=mock.Mock())
            with mock.patch.object(self.module, "SEAL_ROOT", seal_root, create=True), mock.patch.multiple(
                self.module,
                SubprocessTransport=transport_cls, ProductionCheckpointValidator=mock.Mock(),
                ProductionClusterPreflight=mock.Mock(), ProductionCollector=mock.Mock(), CacheAdvisor=mock.Mock(),
                LiveRunner=mock.Mock(return_value=mock.Mock(run=mock.Mock(return_value={}))),
            ):
                self.assertNotEqual(self.module.main(self.argv(values)), 0)
            transport_cls.assert_not_called()

        def execution_digest():
            with tempfile.TemporaryDirectory() as directory:
                values = self.inputs(directory)
                runner = mock.Mock(run=mock.Mock(return_value={}))
                runner_cls = mock.Mock(return_value=runner)
                with mock.patch.multiple(
                    self.module,
                    SubprocessTransport=mock.Mock(return_value=mock.Mock()), ProductionCheckpointValidator=mock.Mock(),
                    ProductionClusterPreflight=mock.Mock(), ProductionCollector=mock.Mock(), CacheAdvisor=mock.Mock(),
                    LiveRunner=runner_cls,
                ):
                    self.assertEqual(self.module.main(self.argv(values)), 0)
                return runner_cls.call_args.kwargs["execution_digest"]

        self.assertEqual(execution_digest(), execution_digest())


if __name__ == "__main__":
    unittest.main()
