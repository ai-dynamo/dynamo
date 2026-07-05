# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))
RUNTIME_FIXTURE = EVAL_ROOT / "fixtures" / "runtime-binding.json"

from evaluate_pro import track_evaluator, wait_with_cleanup  # noqa: E402
from endpoint_preflight import build_evidence  # noqa: E402
from capture_task_images import (  # noqa: E402
    inspect_image,
    install_docker_image_guard,
    task_image_ref,
)
from prediction_digest import prediction_digest  # noqa: E402
from runtime_binding import make_wrapper  # noqa: E402
from source_provenance import build_source_provenance  # noqa: E402


class CompletenessGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset.jsonl"
        self.dataset_ids = ["alpha-1", "beta-1", "alpha-2"]
        self.dataset.write_text(
            "".join(
                json.dumps({"instance_id": instance_id}) + "\n"
                for instance_id in self.dataset_ids
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_script(
        self, script: str, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, arguments)],
            check=check,
            text=True,
            capture_output=True,
        )

    def prepare_scope(
        self, *, instance_filter: str = "", instance_slice: str = ""
    ) -> Path:
        scope = self.root / f"scope-{len(list(self.root.glob('scope-*.json')))}.json"
        self.run_script(
            "manage_scope.py",
            "prepare",
            "--dataset",
            self.dataset,
            "--expected",
            "3",
            "--output",
            scope,
            "--filter",
            instance_filter,
            "--slice",
            instance_slice,
        )
        return scope

    @staticmethod
    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def canonical_sha256(value: object) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def task_image_ref(instance_id: str) -> str:
        return f"docker.io/swebench/sweb.eval.x86_64.{instance_id}:latest"

    def effective_config(self) -> dict:
        return {
            "agent": {
                "system_template": "system",
                "instance_template": "task",
                "step_limit": 250,
                "cost_limit": 3.0,
                "wall_time_limit_seconds": 14400,
                "max_consecutive_format_errors": 3,
                "output_path": None,
            },
            "environment": {
                "cwd": "/testbed",
                "env": {},
                "forward_env": [],
                "timeout": 900,
                "executable": "docker",
                "run_args": ["--rm"],
                "container_timeout": "4h",
                "pull_timeout": 1800,
                "interpreter": ["bash", "-c"],
            },
            "model": {
                "model_name": "zai-org/GLM-5.2",
                "model_kwargs": {
                    "api_base": "http://glm52-dynamo-vllm-frontend:8000/v1",
                    "custom_llm_provider": "openai",
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 32768,
                    "timeout": 1800,
                    "drop_params": True,
                    "parallel_tool_calls": True,
                },
                "litellm_model_registry": None,
                "set_cache_control": None,
                "cost_tracking": "ignore_errors",
                "format_error_template": "error",
                "observation_template": "observation",
                "multimodal_regex": "",
            },
        }

    def write_run_evidence(
        self, scope: Path, *, omit_task_images: set[str] | None = None
    ) -> tuple[Path, Path]:
        omit_task_images = omit_task_images or set()
        effective_config = self.effective_config()
        endpoint_evidence = {
            "schema_version": 1,
            "requested_model": "zai-org/GLM-5.2",
            "expected_context_window": 409600,
            "selected_model_response": {
                "id": "zai-org/GLM-5.2",
                "context_window": 409600,
            },
            "full_response": {
                "data": [{"id": "zai-org/GLM-5.2", "context_window": 409600}]
            },
        }
        deployment = json.loads(RUNTIME_FIXTURE.read_text())
        deployment["campaign_phase"] = "validation"
        endpoint_payload = (
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n"
        ).encode()
        campaign_source = {
            "schema_version": 1,
            "source_commit": deployment["recipe"]["source_commit"],
            "source_clean": True,
            "source_changed_path_count": 0,
            "bundle_sha256": "1" * 64,
            "source_tree_sha256": "2" * 64,
            "eval_tree_sha256": "3" * 64,
            "campaign_env_sha256": "4" * 64,
            "source_file_count": 10,
            "eval_file_count": 9,
        }
        evaluator = {
            "deployment_source_sha256": self.canonical_sha256(deployment),
            "runtime_family": "vllm",
            "runtime_source_revision": "c" * 40,
            "dynamo_enabled": True,
            "tensor_parallel_size": 4,
            "generation": {"workers": 16, "batch_size": 8},
            "evaluation": {
                "workers": 8,
                "timeout_seconds": 3600,
                "backend": "official-swebench-docker",
                "docker_platform": None,
            },
            "effective_config_sha256": "a" * 64,
            "effective_config_content_sha256": self.canonical_sha256(effective_config),
            "effective_config": effective_config,
            "endpoint_evidence": {
                "file_sha256": hashlib.sha256(endpoint_payload).hexdigest(),
                "content_sha256": self.canonical_sha256(endpoint_evidence),
                "content": endpoint_evidence,
            },
            "campaign_source": campaign_source,
        }
        wrapper = make_wrapper(
            deployment,
            evaluator=evaluator,
            variant="dynamo-vllm",
            campaign_phase="validation",
            endpoint="http://glm52-dynamo-vllm-frontend:8000/v1",
        )
        runtime_binding = wrapper["content"]
        (self.root / "runtime-binding.json").write_text(
            json.dumps(runtime_binding, indent=2, sort_keys=True) + "\n"
        )
        metadata = {
            "schema_version": 3,
            "run_name": "test-run",
            "suite": "verified",
            "campaign_phase": "validation",
            "endpoint": "http://glm52-dynamo-vllm-frontend:8000/v1",
            "model": "zai-org/GLM-5.2",
            "scope_sha256": self.sha256_file(scope),
            "dataset": {"evaluator_jsonl_sha256": self.sha256_file(self.dataset)},
            "python_environment": {
                "constraints_lock_sha256": "1" * 64,
                "freeze_sha256": "2" * 64,
                "normalized_freeze_sha256": "3" * 64,
                "normalized_requirement_count": 101,
            },
            "campaign_source": campaign_source,
            "runtime_binding": wrapper,
        }
        metadata_path = self.root / "run-metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        scope_payload = json.loads(scope.read_text())
        images = {}
        for index, instance_id in enumerate(scope_payload["target_ids"]):
            if instance_id in omit_task_images:
                continue
            content = {
                "image_id": f"sha256:{index + 1:064x}",
                "repo_digests": [f"repo/task@sha256:{index + 101:064x}"],
            }
            images[instance_id] = {
                "requested_ref": self.task_image_ref(instance_id),
                **content,
                "content_identity_sha256": self.canonical_sha256(content),
            }
        task_evidence = {
            "schema_version": 1,
            "suite": "verified",
            "dataset_sha256": self.sha256_file(self.dataset),
            "scope_sha256": self.sha256_file(scope),
            "images": images,
        }
        task_path = self.root / "task-images.json"
        task_path.write_text(json.dumps(task_evidence))
        return metadata_path, task_path

    def write_predictions(
        self,
        predictions: dict[str, str],
        *,
        exit_status: str = "Submitted",
        exception: str = "",
        omit_trajectories: set[str] | None = None,
    ) -> Path:
        agent_dir = self.root / "agent"
        agent_dir.mkdir(exist_ok=True)
        omit_trajectories = omit_trajectories or set()
        payload = {
            instance_id: {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "zai-org/GLM-5.2",
            }
            for instance_id, patch in predictions.items()
        }
        (agent_dir / "preds.json").write_text(json.dumps(payload))
        for instance_id, patch in predictions.items():
            if instance_id in omit_trajectories:
                continue
            effective_config = json.loads(json.dumps(self.effective_config()))
            effective_config["environment"]["image"] = self.task_image_ref(instance_id)
            info = {
                "mini_version": "2.4.4",
                "exit_status": exit_status,
                "submission": patch,
                "model_stats": {"api_calls": 2},
                "config": {
                    **effective_config,
                    "agent_type": "minisweagent.run.benchmarks.utils.common.ProgressTrackingAgent",
                    "environment_type": "minisweagent.environments.docker.DockerEnvironment",
                    "model_type": "minisweagent.models.litellm_model.LitellmModel",
                },
            }
            final_extra = {"exit_status": exit_status, "submission": patch}
            if exception:
                info["exception_str"] = exception
                info["traceback"] = "traceback"
                final_extra["exception_str"] = exception
                final_extra["traceback"] = "traceback"
            trajectory = {
                "trajectory_format": "mini-swe-agent-1.1",
                "instance_id": instance_id,
                "info": info,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "exit", "content": patch, "extra": final_extra},
                ],
            }
            trajectory_dir = agent_dir / instance_id
            trajectory_dir.mkdir()
            (trajectory_dir / f"{instance_id}.traj.json").write_text(
                json.dumps(trajectory)
            )
        return agent_dir

    def generation_summary(
        self,
        scope: Path,
        agent_dir: Path,
        *,
        check: bool = True,
        omit_task_images: set[str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = self.root / "generation-summary.json"
        metadata, task_images = self.write_run_evidence(
            scope, omit_task_images=omit_task_images
        )
        result = self.run_script(
            "summarize_generation.py",
            "--agent-dir",
            agent_dir,
            "--dataset",
            self.dataset,
            "--expected",
            "3",
            "--scope",
            scope,
            "--run-metadata",
            metadata,
            "--task-images",
            task_images,
            "--output",
            output,
            "--require-complete",
            check=check,
        )
        return result, json.loads(output.read_text())

    def standard_raw(
        self,
        *,
        submitted: list[str],
        resolved: list[str],
        unresolved: list[str],
        empty: list[str],
        errors: list[str],
        incomplete: list[str],
        completed: list[str],
        total: int | None = None,
    ) -> Path:
        raw = (
            self.root
            / f"raw-standard-{len(list(self.root.glob('raw-standard-*.json')))}.json"
        )
        raw.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "total_instances": (
                        len(self.dataset_ids) if total is None else total
                    ),
                    "submitted_instances": len(submitted),
                    "completed_instances": len(completed),
                    "resolved_instances": len(resolved),
                    "unresolved_instances": len(unresolved),
                    "empty_patch_instances": len(empty),
                    "error_instances": len(errors),
                    "submitted_ids": submitted,
                    "completed_ids": completed,
                    "resolved_ids": resolved,
                    "unresolved_ids": unresolved,
                    "empty_patch_ids": empty,
                    "error_ids": errors,
                    "incomplete_ids": incomplete,
                }
            )
        )
        return raw

    def score_summary(
        self,
        *,
        kind: str,
        scope: Path,
        raw: Path,
        status_dir: Path | None = None,
        check: bool = True,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = self.root / f"score-{kind}.json"
        arguments = [
            "--kind",
            kind,
            "--suite",
            "verified" if kind == "swebench" else "pro",
            "--expected",
            "3",
            "--dataset",
            str(self.dataset),
            "--scope",
            str(scope),
            "--raw",
            str(raw),
            "--output",
            str(output),
            "--require-complete",
        ]
        if status_dir is not None:
            arguments.extend(("--status-dir", str(status_dir)))
        result = self.run_script("summarize_score.py", *arguments, check=check)
        return result, json.loads(output.read_text())

    def test_scope_applies_filter_then_slice_and_rejects_drift(self) -> None:
        scope = self.prepare_scope(instance_filter="^alpha", instance_slice="1:")
        payload = json.loads(scope.read_text())
        self.assertEqual(payload["scope"], "smoke")
        self.assertEqual(payload["target_ids"], ["alpha-2"])

        reused = self.run_script(
            "manage_scope.py",
            "prepare",
            "--dataset",
            self.dataset,
            "--expected",
            "3",
            "--output",
            scope,
            "--reuse-existing-if-unselected",
        )
        self.assertEqual(reused.returncode, 0)
        drift = self.run_script(
            "manage_scope.py",
            "prepare",
            "--dataset",
            self.dataset,
            "--expected",
            "3",
            "--output",
            scope,
            "--slice",
            "0:1",
            check=False,
        )
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("scope differs", drift.stderr)

    def test_full_generation_rejects_equal_count_with_wrong_ids(self) -> None:
        scope = self.prepare_scope()
        agent_dir = self.write_predictions(
            {"alpha-1": "patch", "beta-1": "patch", "foreign": "patch"}
        )
        result, summary = self.generation_summary(scope, agent_dir, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["missing_prediction_ids"], ["alpha-2"])
        self.assertEqual(summary["unexpected_prediction_ids"], ["foreign"])

    def test_prediction_digest_is_canonical_and_patch_sensitive(self) -> None:
        first = self.root / "predictions-first.json"
        reordered = self.root / "predictions-reordered.json"
        changed = self.root / "predictions-changed.json"
        first.write_text(
            json.dumps({"alpha-1": {"model_patch": "one", "instance_id": "alpha-1"}})
        )
        reordered.write_text(
            '{"alpha-1":{"instance_id":"alpha-1","model_patch":"one"}}'
        )
        changed.write_text(
            json.dumps({"alpha-1": {"model_patch": "two", "instance_id": "alpha-1"}})
        )
        self.assertEqual(prediction_digest(first), prediction_digest(reordered))
        self.assertNotEqual(prediction_digest(first), prediction_digest(changed))

    def test_smoke_generation_uses_slice_denominator(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions({"alpha-1": "patch"})
        _, summary = self.generation_summary(scope, agent_dir)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["scope"], "smoke")
        self.assertEqual(summary["target_instances"], 1)
        self.assertEqual(summary["excluded_dataset_instances"], 2)

    def test_generation_rejects_prediction_without_trajectory(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions(
            {"alpha-1": ""}, omit_trajectories={"alpha-1"}
        )
        result, summary = self.generation_summary(scope, agent_dir, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["infrastructure_error_ids"], ["alpha-1"])
        self.assertIn(
            "trajectory file is missing",
            summary["instances"][0]["validation_failures"],
        )

    def test_generation_rejects_api_exception_disguised_as_empty_patch(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions(
            {"alpha-1": ""},
            exit_status="APIConnectionError",
            exception="connection refused",
        )
        result, summary = self.generation_summary(scope, agent_dir, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["empty_patches"], 1)
        self.assertEqual(summary["valid_model_empty_patches"], 0)
        self.assertEqual(summary["infrastructure_error_ids"], ["alpha-1"])

    def test_generation_accepts_evidenced_model_empty_patch(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions(
            {"alpha-1": ""}, exit_status="LimitsExceeded"
        )
        _, summary = self.generation_summary(scope, agent_dir)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["valid_model_empty_patches"], 1)
        self.assertEqual(summary["infrastructure_error_ids"], [])

    def test_generation_rejects_effective_config_drift(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions({"alpha-1": "patch"})
        trajectory_path = agent_dir / "alpha-1" / "alpha-1.traj.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["info"]["config"]["model"]["model_kwargs"]["max_tokens"] = 8192
        trajectory_path.write_text(json.dumps(trajectory))
        result, summary = self.generation_summary(scope, agent_dir, check=False)
        self.assertNotEqual(result.returncode, 0)
        failures = summary["instances"][0]["validation_failures"]
        self.assertTrue(any("model_kwargs.max_tokens" in item for item in failures))

    def test_generation_rejects_endpoint_and_cross_trajectory_config_drift(
        self,
    ) -> None:
        scope = self.prepare_scope(instance_slice="0:2")
        agent_dir = self.write_predictions({"alpha-1": "one", "beta-1": "two"})
        trajectory_path = agent_dir / "beta-1" / "beta-1.traj.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["info"]["config"]["model"]["model_kwargs"]["api_base"] = (
            "http://wrong.example/v1"
        )
        trajectory_path.write_text(json.dumps(trajectory))
        result, summary = self.generation_summary(scope, agent_dir, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(summary["trajectory_effective_config_sha256s"]), 2)
        self.assertTrue(
            any("configs are inconsistent" in item for item in summary["gate_failures"])
        )

    def test_generation_rejects_missing_task_image_identity(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        agent_dir = self.write_predictions({"alpha-1": "patch"})
        result, summary = self.generation_summary(
            scope,
            agent_dir,
            check=False,
            omit_task_images={"alpha-1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(summary["missing_task_image_ids"], ["alpha-1"])

    def test_endpoint_preflight_requires_exact_409600_context(self) -> None:
        good = {"data": [{"id": "test-model", "context_window": 409600}]}
        evidence = build_evidence(good, "test-model", 409600)
        self.assertEqual(evidence["selected_model_response"]["context_window"], 409600)
        self.assertEqual(evidence["full_response"], good)

        native_vllm = {
            "data": [
                {
                    "id": "test-model",
                    "max_model_len": 409600,
                    "owned_by": "vllm",
                }
            ]
        }
        evidence = build_evidence(native_vllm, "test-model", 409600)
        self.assertEqual(
            evidence["selected_model_response"],
            {
                "id": "test-model",
                "context_window": 409600,
                "owned_by": "vllm",
            },
        )
        self.assertEqual(evidence["full_response"], native_vllm)

        both = {
            "data": [
                {
                    "id": "test-model",
                    "context_window": 409600,
                    "max_model_len": 409600,
                }
            ]
        }
        evidence = build_evidence(both, "test-model", 409600)
        self.assertEqual(
            evidence["selected_model_response"],
            {"id": "test-model", "context_window": 409600},
        )

        nullable_alias = {
            "data": [
                {
                    "id": "test-model",
                    "context_window": None,
                    "max_model_len": 409600,
                }
            ]
        }
        evidence = build_evidence(nullable_alias, "test-model", 409600)
        self.assertEqual(
            evidence["selected_model_response"],
            {"id": "test-model", "context_window": 409600},
        )

        with self.assertRaisesRegex(ValueError, "expected 409600"):
            build_evidence(
                {"data": [{"id": "test-model", "context_window": 262144}]},
                "test-model",
                409600,
            )
        with self.assertRaisesRegex(ValueError, "expected 409600"):
            build_evidence(
                {"data": [{"id": "test-model", "max_model_len": 262144}]},
                "test-model",
                409600,
            )

    def test_endpoint_preflight_rejects_missing_or_conflicting_context_aliases(
        self,
    ) -> None:
        for entry in (
            {"id": "test-model"},
            {"id": "test-model", "context_window": None},
            {
                "id": "test-model",
                "context_window": None,
                "max_model_len": None,
            },
        ):
            with (
                self.subTest(entry=entry),
                self.assertRaisesRegex(ValueError, "no non-null context field"),
            ):
                build_evidence({"data": [entry]}, "test-model", 409600)

        with self.assertRaisesRegex(ValueError, "conflicting context aliases"):
            build_evidence(
                {
                    "data": [
                        {
                            "id": "test-model",
                            "context_window": 409600,
                            "max_model_len": 262144,
                        }
                    ]
                },
                "test-model",
                409600,
            )

        with self.assertRaisesRegex(ValueError, "expected 409600"):
            build_evidence(
                {
                    "data": [
                        {
                            "id": "test-model",
                            "context_window": 262144,
                            "max_model_len": 262144,
                        }
                    ]
                },
                "test-model",
                409600,
            )

    def test_task_image_identity_requires_repo_digest_and_image_id(self) -> None:
        self.assertEqual(
            task_image_ref("owner__repo-1", {}),
            "docker.io/swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
        )
        docker_output = [
            {
                "Id": f"sha256:{'1' * 64}",
                "RepoDigests": [f"repo/task@sha256:{'2' * 64}"],
            }
        ]
        completed = subprocess.CompletedProcess(
            ["docker"], 0, stdout=json.dumps(docker_output), stderr=""
        )
        with mock.patch("capture_task_images.subprocess.run", return_value=completed):
            identity = inspect_image("repo/task:latest")
        self.assertEqual(identity["image_id"], f"sha256:{'1' * 64}")
        self.assertEqual(identity["repo_digests"], [f"repo/task@sha256:{'2' * 64}"])

    def test_evaluator_image_guard_rejects_mutated_tag_content(self) -> None:
        reference = self.task_image_ref("alpha-1")

        class Image:
            def __init__(self, image_id: str) -> None:
                self.attrs = {
                    "Id": image_id,
                    "RepoDigests": [f"repo/task@sha256:{'2' * 64}"],
                }

        class ImageCollection:
            image_id = f"sha256:{'9' * 64}"

            def get(self, _name: str) -> Image:
                return Image(self.image_id)

            def pull(self, _name: str, *args: object, **kwargs: object) -> Image:
                return Image(self.image_id)

        class Images:
            pass

        class Models:
            images = Images()

        Models.images.ImageCollection = ImageCollection

        class Docker:
            models = Models()

        content = {
            "image_id": f"sha256:{'1' * 64}",
            "repo_digests": [f"repo/task@sha256:{'2' * 64}"],
        }
        evidence = self.root / "guard-task-images.json"
        evidence.write_text(
            json.dumps(
                {
                    "images": {
                        "alpha-1": {
                            "requested_ref": reference,
                            **content,
                            "content_identity_sha256": self.canonical_sha256(content),
                        }
                    }
                }
            )
        )
        install_docker_image_guard(Docker, evidence)
        with self.assertRaisesRegex(RuntimeError, "differs from generation"):
            ImageCollection().get(reference)
        ImageCollection.image_id = content["image_id"]
        unqualified = reference.removeprefix("docker.io/")
        self.assertIsInstance(ImageCollection().get(unqualified), Image)

    def test_standard_full_accepts_empty_patch_as_scored_failure(self) -> None:
        scope = self.prepare_scope()
        raw = self.standard_raw(
            submitted=self.dataset_ids,
            resolved=["alpha-1"],
            unresolved=["beta-1"],
            empty=["alpha-2"],
            errors=[],
            incomplete=[],
            completed=["alpha-1", "beta-1"],
        )
        _, summary = self.score_summary(kind="swebench", scope=scope, raw=raw)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["failed_ids"], ["alpha-2", "beta-1"])
        self.assertEqual(summary["benchmark_score"], 1 / 3)

    def test_standard_smoke_scores_only_its_recorded_scope(self) -> None:
        scope = self.prepare_scope(instance_slice="0:1")
        raw = self.standard_raw(
            submitted=["alpha-1"],
            resolved=[],
            unresolved=["alpha-1"],
            empty=[],
            errors=[],
            incomplete=[],
            completed=["alpha-1"],
            total=1,
        )
        _, summary = self.score_summary(kind="swebench", scope=scope, raw=raw)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["scope"], "smoke")
        self.assertEqual(summary["target_instances"], 1)
        self.assertIsNone(summary["benchmark_score"])

    def test_standard_full_rejects_incomplete_and_error_evaluations(self) -> None:
        scope = self.prepare_scope()
        cases = (
            self.standard_raw(
                submitted=["alpha-1", "beta-1"],
                resolved=["alpha-1"],
                unresolved=["beta-1"],
                empty=[],
                errors=[],
                incomplete=["alpha-2"],
                completed=["alpha-1", "beta-1"],
            ),
            self.standard_raw(
                submitted=self.dataset_ids,
                resolved=["alpha-1"],
                unresolved=["beta-1"],
                empty=[],
                errors=["alpha-2"],
                incomplete=[],
                completed=self.dataset_ids,
            ),
        )
        for raw in cases:
            result, summary = self.score_summary(
                kind="swebench", scope=scope, raw=raw, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["benchmark_score"])

    def test_pro_statuses_distinguish_failures_from_evaluator_errors(self) -> None:
        smoke_scope = self.prepare_scope(instance_slice="0:1")
        smoke_raw = self.root / "raw-pro-smoke.json"
        smoke_raw.write_text(json.dumps({"alpha-1": False}))
        smoke_status = self.root / "status-smoke"
        smoke_status.mkdir()
        (smoke_status / "one.json").write_text(
            json.dumps({"instance_id": "alpha-1", "status": "completed"})
        )
        _, smoke_summary = self.score_summary(
            kind="pro",
            scope=smoke_scope,
            raw=smoke_raw,
            status_dir=smoke_status,
        )
        self.assertTrue(smoke_summary["complete"])
        self.assertIsNone(smoke_summary["benchmark_score"])

        full_scope = self.prepare_scope()
        full_raw = self.root / "raw-pro-full.json"
        full_raw.write_text(json.dumps(dict.fromkeys(self.dataset_ids, False)))
        full_status = self.root / "status-full"
        full_status.mkdir()
        for index, instance_id in enumerate(self.dataset_ids):
            status = "error" if instance_id == "alpha-2" else "completed"
            (full_status / f"{index}.json").write_text(
                json.dumps({"instance_id": instance_id, "status": status})
            )
        result, full_summary = self.score_summary(
            kind="pro",
            scope=full_scope,
            raw=full_raw,
            status_dir=full_status,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(full_summary["complete"])
        self.assertEqual(full_summary["evaluation_error_ids"], ["alpha-2"])

    def test_pro_tracker_records_valid_output_and_infrastructure_errors(self) -> None:
        completed_dir = self.root / "tracker-completed"
        completed_dir.mkdir()
        output = {"tests": [{"name": "test_a", "status": "FAILED"}]}
        tracked_success = track_evaluator(lambda _patch, _sample: output, completed_dir)
        sample = {
            "instance_id": "alpha-1",
            "fail_to_pass": "['test_a']",
            "pass_to_pass": "[]",
        }
        self.assertEqual(tracked_success("patch", sample), output)
        completed = json.loads(next(completed_dir.glob("*.json")).read_text())
        self.assertEqual(completed["status"], "completed")

        error_dir = self.root / "tracker-error"
        error_dir.mkdir()
        tracked_error = track_evaluator(lambda _patch, _sample: None, error_dir)
        self.assertIsNone(tracked_error("patch", {"instance_id": "beta-1"}))
        error = json.loads(next(error_dir.glob("*.json")).read_text())
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["error"], "missing valid test output")

        malformed_dir = self.root / "tracker-malformed"
        malformed_dir.mkdir()
        tracked_malformed = track_evaluator(
            lambda _patch, _sample: {"tests": [{"name": "test_a"}]},
            malformed_dir,
        )
        tracked_malformed("patch", sample)
        malformed = json.loads(next(malformed_dir.glob("*.json")).read_text())
        self.assertEqual(malformed["status"], "error")

    def test_pro_timeout_kills_container_and_records_replayable_error(self) -> None:
        class Container:
            def __init__(self) -> None:
                self.killed = False
                self.removed = False

            def kill(self) -> None:
                self.killed = True

            def remove(self, *, force: bool) -> None:
                self.removed = force

        container = Container()

        def wait(_container: Container, *, timeout: int) -> None:
            self.assertEqual(timeout, 17)
            raise TimeoutError("test timeout")

        status_dir = self.root / "timeout-status"
        status_dir.mkdir()

        def upstream(_patch: str, _sample: dict) -> None:
            try:
                wait_with_cleanup(container, wait, 17)
            except TimeoutError:
                return None

        tracked = track_evaluator(upstream, status_dir)
        self.assertIsNone(tracked("patch", {"instance_id": "alpha-1"}))
        self.assertTrue(container.killed)
        self.assertTrue(container.removed)
        status = json.loads(next(status_dir.glob("*.json")).read_text())
        self.assertEqual(status["status"], "error")
        self.assertIn("17s hard timeout", status["error"])

    def test_pro_prompt_renders_the_configured_working_directory(self) -> None:
        config = (ROOT / "config" / "glm52.yaml").read_text()
        template = config.split("instance_template: |", 1)[1].split(
            "\nenvironment:", 1
        )[0]
        self.assertIn("files in {{cwd}}", template)
        self.assertNotIn("/testbed", template)
        run_wrapper = (ROOT / "run.sh").read_text()
        self.assertIn('config/pro.yaml"', run_wrapper)
        pro_config = (ROOT / "config" / "pro.yaml").read_text()
        self.assertIn("cwd: /app", pro_config)
        self.assertIn('run_args: ["--rm", "--entrypoint="]', pro_config)


class ImmutableRunMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "run-metadata.json"
        self.predictions = self.root / "preds.json"
        self.dataset = self.root / "verified.jsonl"
        self.dataset.write_text('{"instance_id":"alpha-1"}\n')
        self.provenance = self.root / "provenance.json"
        self.provenance.write_text(
            json.dumps(
                {
                    "split": "test",
                    "datasets": {
                        "verified": {"repo": "test/verified", "revision": "abc"}
                    },
                }
            )
        )
        self.pins = self.root / "pins.env"
        self.pins.write_text(
            "MINI_SWE_AGENT_VERSION=2.4.4\n"
            "SWEBENCH_VERSION=4.1.0\n"
            "TERMINAL_BENCH_TASKS=89\n"
        )
        self.base_config = self.root / "base.yaml"
        self.base_config.write_text("agent: {}\n")
        self.delta_config = self.root / "delta.yaml"
        self.delta_config.write_text("model: {}\n")
        self.scope = self.root / "scope.json"
        self.scope.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": "full",
                    "full_run": True,
                    "target_ids": ["alpha-1"],
                }
            )
        )
        self.repos: dict[str, Path] = {}
        commits: dict[str, str] = {}
        for name in ("mini_swe_agent", "swebench", "swebench_pro"):
            repo = self.root / name
            repo.mkdir()
            subprocess.run(["git", "init", "-q", repo], check=True)
            (repo / "README").write_text(name)
            subprocess.run(["git", "-C", repo, "add", "README"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "initial",
                ],
                check=True,
            )
            self.repos[name] = repo
            commits[name] = subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        self.source_lock = self.root / "source-lock.json"
        self.source_lock.write_text(
            json.dumps(
                {
                    "mini_swe_agent": {
                        "version": "2.4.4",
                        "commit": commits["mini_swe_agent"],
                    },
                    "swebench": {"version": "4.1.0", "commit": commits["swebench"]},
                    "swebench_pro": {"commit": commits["swebench_pro"]},
                }
            )
        )
        self.provenance.write_text(
            json.dumps(
                {
                    "split": "test",
                    "datasets": {
                        "verified": {
                            "repo": "test/verified",
                            "revision": "dataset-commit",
                            "expected": 1,
                            "rows": 1,
                        }
                    },
                }
            )
        )
        self.pins.write_text(
            "MINI_SWE_AGENT_VERSION=2.4.4\n"
            f"MINI_SWE_AGENT_COMMIT={commits['mini_swe_agent']}\n"
            "SWEBENCH_VERSION=4.1.0\n"
            f"SWEBENCH_COMMIT={commits['swebench']}\n"
            f"SWEBENCH_PRO_COMMIT={commits['swebench_pro']}\n"
            "SWEBENCH_VERIFIED_REVISION=dataset-commit\n"
            "SWEBENCH_VERIFIED_CASES=1\n"
        )
        self.effective_config = self.root / "effective-config.json"
        self.endpoint_evidence = self.root / "endpoint-models.json"
        self.runtime_binding = self.root / "runtime-binding.json"
        self.deployment_binding = self.root / "active.json"
        self.constraints_lock = self.root / "constraints.lock"
        self.constraints_lock.write_text("dependency==1.0\n")
        self.environment_freeze = self.root / "environment.freeze.txt"
        self.environment_freeze.write_text(
            "dependency==1.0\n-e file:///tmp/mini-swe-agent\n-e file:///tmp/SWE-bench\n"
        )
        self.normalized_freeze = self.root / "environment.normalized.freeze.txt"
        self.normalized_freeze.write_text("dependency==1.0\n")
        self.campaign_source_root = self.root / "workspace"
        (self.campaign_source_root / "eval").mkdir(parents=True)
        (self.campaign_source_root / "campaign.env").write_text(
            "MAX_MODEL_LEN=409600\n"
        )
        (self.campaign_source_root / "eval/run.sh").write_text("#!/bin/sh\n")
        self.campaign_source_metadata = (
            self.campaign_source_root / "source-provenance.json"
        )
        self.campaign_source_metadata.write_text(
            json.dumps(
                build_source_provenance(
                    self.campaign_source_root,
                    source_commit="c" * 40,
                    source_branch="rmccormick/glm52",
                    bundle_sha256="b" * 64,
                )
            )
            + "\n"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(
        self,
        endpoint: str = "http://glm52-dynamo-vllm-frontend:8000/v1",
        generation_workers: str = "16",
        generation_batch_size: str = "8",
        evaluator_workers: str = "8",
        evaluator_timeout: str = "3600",
    ) -> subprocess.CompletedProcess[str]:
        self.deployment_binding.write_text(RUNTIME_FIXTURE.read_text())
        effective_config = {
            "agent": {
                "wall_time_limit_seconds": 14400,
                "step_limit": 250,
                "cost_limit": 3.0,
            },
            "environment": {
                "environment_class": "docker",
                "cwd": "/testbed",
                "timeout": 900,
                "pull_timeout": 1800,
                "container_timeout": "4h",
                "interpreter": ["bash", "-c"],
                "run_args": ["--rm"],
            },
            "model": {
                "model_name": "openai/zai-org/GLM-5.2",
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "api_base": endpoint,
                    "custom_llm_provider": "openai",
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 32768,
                    "timeout": 1800,
                    "drop_params": True,
                    "parallel_tool_calls": True,
                },
            },
        }
        self.effective_config.write_text(json.dumps(effective_config))
        self.endpoint_evidence.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "requested_model": "zai-org/GLM-5.2",
                    "expected_context_window": 409600,
                    "selected_model_response": {
                        "id": "zai-org/GLM-5.2",
                        "context_window": 409600,
                    },
                    "full_response": {
                        "data": [
                            {
                                "id": "zai-org/GLM-5.2",
                                "context_window": 409600,
                            }
                        ]
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        command = [
            sys.executable,
            str(ROOT / "manage_run.py"),
            "prepare",
            "--output",
            str(self.output),
            "--run-name",
            "dynamo-vllm-ab",
            "--suite",
            "verified",
            "--endpoint",
            endpoint,
            "--model",
            "zai-org/GLM-5.2",
            "--variant",
            "dynamo-vllm",
            "--campaign-phase",
            "ab",
            "--deployment-binding",
            str(self.deployment_binding),
            "--campaign-source-metadata",
            str(self.campaign_source_metadata),
            "--campaign-source-root",
            str(self.campaign_source_root),
            "--model-id",
            "nvidia/GLM-5.2-NVFP4",
            "--model-revision",
            "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
            "--context-window",
            "409600",
            "--tensor-parallel-size",
            "4",
            "--runtime-family",
            "vllm",
            "--runtime-image",
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260701-5245c0f@sha256:c3336583c830ea5c3cf4bd5cc92cb57200b8f558398a18c3ac0f473f9b74dd1d",
            "--runtime-source-revision",
            "c" * 40,
            "--dynamo-enabled",
            "true",
            "--generation-workers",
            generation_workers,
            "--generation-batch-size",
            generation_batch_size,
            "--evaluator-workers",
            evaluator_workers,
            "--evaluator-timeout",
            evaluator_timeout,
            "--pro-eval-backend",
            "local",
            "--docker-platform",
            "",
            "--effective-config",
            str(self.effective_config),
            "--endpoint-evidence",
            str(self.endpoint_evidence),
            "--runtime-binding-output",
            str(self.runtime_binding),
            "--config",
            f"upstream={self.base_config}",
            "--config",
            f"delta={self.delta_config}",
            "--dataset",
            str(self.dataset),
            "--dataset-provenance",
            str(self.provenance),
            "--pins",
            str(self.pins),
            "--source-lock",
            str(self.source_lock),
            "--constraints-lock",
            str(self.constraints_lock),
            "--environment-freeze",
            str(self.environment_freeze),
            "--normalized-environment-freeze",
            str(self.normalized_freeze),
            "--scope",
            str(self.scope),
            "--predictions",
            str(self.predictions),
        ]
        for name, repo in self.repos.items():
            command.extend(("--source-repo", f"{name}={repo}"))
        return subprocess.run(command, check=False, capture_output=True, text=True)

    def test_run_identity_reuses_exact_inputs_and_rejects_drift(self) -> None:
        created = self.prepare()
        self.assertEqual(created.returncode, 0, created.stderr)
        identity = json.loads(self.output.read_text())
        self.assertEqual(identity["schema_version"], 3)
        self.assertEqual(
            identity["endpoint"],
            "http://glm52-dynamo-vllm-frontend:8000/v1",
        )
        self.assertEqual(identity["suite"], "verified")
        self.assertEqual(
            set(identity["source"]["repositories"]),
            {"mini_swe_agent", "swebench", "swebench_pro"},
        )
        self.assertEqual(
            identity["runtime_binding"]["content"]["deployment"]["max_model_len"],
            409600,
        )
        self.assertEqual(
            identity["runtime_binding"]["content"]["evaluator"]["endpoint_evidence"][
                "content"
            ]["selected_model_response"]["context_window"],
            409600,
        )
        self.assertEqual(
            identity["python_environment"]["normalized_requirement_count"], 1
        )
        self.assertEqual(
            identity["campaign_source"],
            identity["runtime_binding"]["content"]["evaluator"]["campaign_source"],
        )
        self.assertEqual(identity["campaign_source"]["source_commit"], "c" * 40)
        runtime = identity["runtime_binding"]
        self.assertEqual(
            runtime["deployment_sha256"],
            CompletenessGateTest.canonical_sha256(runtime["content"]["deployment"]),
        )
        self.assertEqual(
            runtime["content_sha256"],
            CompletenessGateTest.canonical_sha256(runtime["content"]),
        )
        self.assertEqual(
            runtime["deployment_sha256"],
            runtime["content"]["evaluator"]["deployment_source_sha256"],
        )
        self.assertEqual(self.prepare().returncode, 0)

        endpoint_drift = self.prepare("http://other.example/v1")
        self.assertNotEqual(endpoint_drift.returncode, 0)
        self.assertIn("endpoint differs", endpoint_drift.stderr)

        self.delta_config.write_text("model: {temperature: 0}\n")
        config_drift = self.prepare()
        self.assertNotEqual(config_drift.returncode, 0)
        self.assertIn("configuration", config_drift.stderr)

    def test_run_identity_refuses_to_adopt_legacy_predictions(self) -> None:
        self.predictions.write_text("{}")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("predate immutable run metadata", result.stderr)

    def test_run_identity_rejects_environment_freeze_drift(self) -> None:
        self.assertEqual(self.prepare().returncode, 0)
        self.environment_freeze.write_text(
            "dependency==2.0\n-e file:///tmp/mini-swe-agent\n-e file:///tmp/SWE-bench\n"
        )
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from constraints lock", result.stderr)

    def test_run_identity_rejects_untracked_source_files(self) -> None:
        (self.repos["mini_swe_agent"] / "shadow.py").write_text("raise SystemExit\n")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source repository is not clean", result.stderr)

    def test_run_identity_rejects_campaign_evaluator_source_drift(self) -> None:
        (self.campaign_source_root / "eval/run.sh").write_text("changed\n")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("current campaign source differs", result.stderr)

    def test_full_run_rejects_generation_concurrency_override(self) -> None:
        result = self.prepare(generation_workers="15")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 16 generation workers", result.stderr)

        result = self.prepare(generation_batch_size="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("generation batch size 8", result.stderr)

        result = self.prepare(evaluator_workers="7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 8 evaluator workers", result.stderr)

        result = self.prepare(evaluator_timeout="3599")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("evaluator timeout 3600", result.stderr)

    def test_run_identity_rejects_published_deployment_phase_drift(self) -> None:
        self.assertEqual(self.prepare().returncode, 0)
        binding = json.loads(self.deployment_binding.read_text())
        binding["campaign_phase"] = "ba"
        self.deployment_binding.write_text(json.dumps(binding))
        arguments = [
            sys.executable,
            str(EVAL_ROOT / "runtime_binding.py"),
            str(self.deployment_binding),
            "--variant",
            "dynamo-vllm",
            "--phase",
            "ab",
            "--endpoint",
            "http://glm52-dynamo-vllm-frontend:8000/v1",
        ]
        result = subprocess.run(arguments, check=False, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("campaign_phase", result.stderr)


if __name__ == "__main__":
    unittest.main()
