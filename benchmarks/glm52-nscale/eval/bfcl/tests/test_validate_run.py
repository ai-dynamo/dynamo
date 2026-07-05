# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_run.py"
SPEC = importlib.util.spec_from_file_location("validate_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validate_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_run
SPEC.loader.exec_module(validate_run)

CAPTURE_SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_metadata.py"
CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "capture_metadata", CAPTURE_SCRIPT
)
assert CAPTURE_SPEC is not None and CAPTURE_SPEC.loader is not None
capture_metadata = importlib.util.module_from_spec(CAPTURE_SPEC)
sys.modules[CAPTURE_SPEC.name] = capture_metadata
CAPTURE_SPEC.loader.exec_module(capture_metadata)

from bfcl_endpoint import EndpointModelError, canonical_endpoint_model  # noqa: E402
from source_provenance import SourceProvenanceError, build_source_provenance  # noqa: E402


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def synthetic_population():
    return validate_run.ExpectedPopulation(
        requested_categories=("synthetic",),
        expanded_categories=("cat", "memory_kv"),
        generated_by_category={
            "cat": frozenset(("cat_0", "cat_1")),
            "memory_kv_prereq": frozenset(("memory_kv_prereq_0",)),
        },
        scored_by_category={"cat": frozenset(("cat_0", "cat_1"))},
    )


class ValidateRunTest(unittest.TestCase):
    def valid_metadata(self) -> dict:
        patch = (
            Path(__file__).parents[1]
            / "patches"
            / "0001-glm52-openai-chat-completions.patch"
        )
        deployment = json.loads(
            (
                Path(__file__).parents[2] / "fixtures" / "runtime-binding.json"
            ).read_text()
        )
        campaign_source = {
            "schema_version": 1,
            "source_commit": deployment["recipe"]["source_commit"],
            "source_clean": True,
            "source_changed_path_count": 0,
            "bundle_sha256": "1" * 64,
            "source_tree_sha256": "2" * 64,
            "eval_tree_sha256": "3" * 64,
            "campaign_env_sha256": "4" * 64,
            "source_file_count": 11,
            "eval_file_count": 10,
        }
        binding = capture_metadata.make_wrapper(
            deployment,
            evaluator={"harness": "bfcl-v4", "campaign_source": campaign_source},
            variant="dynamo-vllm",
            campaign_phase="ab",
            endpoint="http://glm52-dynamo-vllm-frontend:8000/v1",
        )
        return {
            "schema_version": 6,
            "bfcl_gorilla_commit": "commit",
            "mode": "full",
            "variant": "dynamo-vllm",
            "campaign_phase": "ab",
            "run_name": "bfcl-full-ab-test",
            "categories": ["all_scoring"],
            "model_registry_name": "zai-org/GLM-5.2-FC",
            "served_model_name": "zai-org/GLM-5.2",
            "endpoint": "http://glm52-dynamo-vllm-frontend:8000/v1",
            "bfcl_patch_sha256": validate_run.file_sha256(patch),
            "bfcl_source_identity": {
                "head": "commit",
                "status": validate_run.EXPECTED_SOURCE_STATUS,
                "tracked_diff_sha256": validate_run.EXPECTED_TRACKED_DIFF_SHA256,
                "new_handler_sha256": validate_run.EXPECTED_NEW_HANDLER_SHA256,
            },
            "endpoint_models_sha256": "a" * 64,
            "endpoint_model": {
                "id": "zai-org/GLM-5.2",
                "object": "model",
                "owned_by": "nvidia",
                "context_window": 409600,
            },
            "temperature": 0.0,
            "max_tokens": 64000,
            "num_threads": 16,
            "include_input_log": True,
            "glm52_openai_extra_body": None,
            "glm52_openai_default_headers_sha256": None,
            "runtime_binding": binding,
            "campaign_source": campaign_source,
            "python_environment": {
                "schema_version": 1,
                "constraints_sha256": validate_run.EXPECTED_CONSTRAINTS_SHA256,
                "freeze_sha256": validate_run.EXPECTED_FREEZE_SHA256,
                "package_count": validate_run.EXPECTED_PACKAGE_COUNT,
                "python": "3.12.11",
            },
        }

    def test_adapter_patch_contains_complete_handler(self) -> None:
        patch = (
            Path(__file__).parents[1]
            / "patches"
            / "0001-glm52-openai-chat-completions.patch"
        )
        result = subprocess.run(
            ["git", "apply", "--numstat", str(patch)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "93\t0\tberkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/glm52_openai.py",
            result.stdout,
        )

    def test_endpoint_context_aliases_canonicalize(self) -> None:
        base = {
            "id": "zai-org/GLM-5.2",
            "object": "model",
            "owned_by": "nvidia",
        }
        expected = {**base, "context_window": 409600}
        accepted = (
            {"context_window": 409600},
            {"max_model_len": 409600},
            {"context_window": 409600, "max_model_len": 409600},
            {"context_window": None, "max_model_len": 409600},
            {"context_window": 409600, "max_model_len": None},
        )
        for aliases in accepted:
            with self.subTest(aliases=aliases):
                self.assertEqual(
                    canonical_endpoint_model({**base, **aliases}, 409600), expected
                )

    def test_endpoint_context_aliases_reject_missing_conflicting_and_wrong(
        self,
    ) -> None:
        rejected = (
            ({}, "non-null"),
            ({"context_window": None, "max_model_len": None}, "non-null"),
            ({"context_window": 262144}, "!= campaign"),
            ({"max_model_len": 262144}, "!= campaign"),
            (
                {"context_window": 409600, "max_model_len": 262144},
                "conflict",
            ),
            ({"max_model_len": 409600.0}, "must be integers"),
        )
        for aliases, message in rejected:
            with self.subTest(aliases=aliases):
                with self.assertRaisesRegex(EndpointModelError, message):
                    canonical_endpoint_model(aliases, 409600)

    def test_configured_population_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cases.json"
            path.write_text(
                json.dumps({"simple_python": ["simple_python_0", "simple_python_1"]})
            )
            population = validate_run.build_configured_population(path)
            self.assertEqual(
                population.generated_ids,
                frozenset(("simple_python_0", "simple_python_1")),
            )

            path.write_text(json.dumps({"simple_python": ["parallel_0"]}))
            with self.assertRaisesRegex(validate_run.ContractError, "does not belong"):
                validate_run.build_configured_population(path)

    def test_perfect_score_gate_rejects_partial_score(self) -> None:
        scores = {
            "expected_count": 5,
            "categories": {
                "simple_python": {"correct_count": 2},
                "parallel": {"correct_count": 1},
                "irrelevance": {"correct_count": 1},
                "multi_turn_base": {"correct_count": 0},
            },
        }
        self.assertEqual(
            validate_run.perfect_score_errors(scores),
            ["Smoke score gate requires 5/5, got 4/5"],
        )
        scores["categories"]["multi_turn_base"]["correct_count"] = 1
        self.assertEqual(validate_run.perfect_score_errors(scores), [])

    def test_metadata_requires_campaign_output_cap(self) -> None:
        metadata = {
            "bfcl_gorilla_commit": "commit",
            "mode": "smoke",
            "variant": "dynamo-vllm",
            "max_tokens": 32768,
        }
        errors = validate_run.verify_metadata(
            metadata,
            "commit",
            "dynamo-vllm",
            False,
            "smoke",
        )
        self.assertIn("Expected max_tokens 64000, got 32768", errors)
        metadata["max_tokens"] = 64000
        self.assertNotIn(
            "Expected max_tokens",
            "\n".join(
                validate_run.verify_metadata(
                    metadata,
                    "commit",
                    "dynamo-vllm",
                    False,
                    "smoke",
                )
            ),
        )

    def test_metadata_rejects_generation_provenance_drift(self) -> None:
        cases = {
            "schema_version": (1, "schema_version 6"),
            "served_model_name": ("other", "served_model_name"),
            "bfcl_patch_sha256": ("0" * 64, "adapter patch digest mismatch"),
            "temperature": (1.0, "temperature 0"),
            "max_tokens": (32000, "max_tokens 64000"),
            "num_threads": (8, "num_threads 16"),
            "include_input_log": (False, "include_input_log true"),
            "glm52_openai_extra_body": (
                '{"chat_template_kwargs":{"enable_thinking":false}}',
                "requires glm52_openai_extra_body to be unset",
            ),
            "glm52_openai_default_headers_sha256": (
                "a" * 64,
                "GLM52_OPENAI_DEFAULT_HEADERS",
            ),
            "endpoint_model": (
                {"id": "zai-org/GLM-5.2", "context_window": 262144},
                "context_window 409600",
            ),
            "runtime_binding": (None, "runtime_binding is missing"),
            "campaign_source": (None, "campaign_source identity"),
            "endpoint": ("http://host:8000/completions", "OpenAI /v1 endpoint"),
        }
        for field, (value, expected_error) in cases.items():
            with self.subTest(field=field):
                metadata = self.valid_metadata()
                metadata[field] = value
                errors = validate_run.verify_metadata(
                    metadata, "commit", "dynamo-vllm", True, "full", "ab"
                )
                self.assertIn(expected_error, "\n".join(errors))

    def test_valid_full_metadata_has_stable_immutable_identity(self) -> None:
        metadata = self.valid_metadata()
        self.assertEqual(
            validate_run.verify_metadata(
                metadata, "commit", "dynamo-vllm", True, "full", "ab"
            ),
            [],
        )
        identity = validate_run.immutable_metadata(metadata)
        self.assertEqual(set(identity), set(validate_run.IMMUTABLE_METADATA_FIELDS))
        self.assertEqual(identity["bfcl_patch_sha256"], metadata["bfcl_patch_sha256"])

    def test_patched_checkout_identity_rejects_unrelated_changes(self) -> None:
        identity = {
            "head": capture_metadata.EXPECTED_BFCL_COMMIT,
            "status": capture_metadata.EXPECTED_SOURCE_STATUS,
            "tracked_diff_sha256": capture_metadata.EXPECTED_TRACKED_DIFF_SHA256,
            "new_handler_sha256": capture_metadata.EXPECTED_NEW_HANDLER_SHA256,
        }
        capture_metadata.verify_source_identity(
            identity, capture_metadata.EXPECTED_BFCL_COMMIT
        )
        for field, value in (
            ("head", "0" * 40),
            ("status", [*capture_metadata.EXPECTED_SOURCE_STATUS, "?? stray.py"]),
            ("tracked_diff_sha256", "0" * 64),
            ("new_handler_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                tampered = dict(identity)
                tampered[field] = value
                with self.assertRaisesRegex(RuntimeError, "differs from pinned HEAD"):
                    capture_metadata.verify_source_identity(
                        tampered, capture_metadata.EXPECTED_BFCL_COMMIT
                    )

    def test_full_capture_rejects_default_headers(self) -> None:
        self.assertIsNone(capture_metadata.default_headers_sha256("full", None))
        with self.assertRaisesRegex(RuntimeError, "forbids"):
            capture_metadata.default_headers_sha256("full", '{"x-route":"alternate"}')
        self.assertEqual(
            capture_metadata.default_headers_sha256("smoke", '{"x-route":"alternate"}'),
            "40a95a4556742ebcc17f5a949ee9253eb43741b4bb88ec765657218b43525f59",
        )

    def test_runtime_binding_embeds_verified_campaign_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "eval").mkdir()
            (root / "campaign.env").write_text("MAX_MODEL_LEN=409600\n")
            (root / "eval/run.sh").write_text("#!/bin/sh\n")
            deployment_path = root / "runtime-binding.json"
            deployment_path.write_text(
                (
                    Path(__file__).parents[2] / "fixtures/runtime-binding.json"
                ).read_text()
            )
            deployment = json.loads(deployment_path.read_text())
            provenance = root / "source-provenance.json"
            provenance.write_text(
                json.dumps(
                    build_source_provenance(
                        root,
                        source_commit=deployment["recipe"]["source_commit"],
                        source_branch="rmccormick/glm52",
                        bundle_sha256="b" * 64,
                    )
                )
            )
            wrapper, campaign_source = capture_metadata.runtime_binding(
                deployment_path,
                variant="dynamo-vllm",
                campaign_phase="ab",
                endpoint="http://glm52-dynamo-vllm-frontend:8000/v1",
                evaluator={"harness": "bfcl-v4"},
                campaign_source_metadata=provenance,
                campaign_source_root=root,
            )
            self.assertEqual(
                wrapper["content"]["evaluator"]["campaign_source"], campaign_source
            )
            (root / "eval/run.sh").write_text("changed\n")
            with self.assertRaises(SourceProvenanceError):
                capture_metadata.runtime_binding(
                    deployment_path,
                    variant="dynamo-vllm",
                    campaign_phase="ab",
                    endpoint="http://glm52-dynamo-vllm-frontend:8000/v1",
                    evaluator={"harness": "bfcl-v4"},
                    campaign_source_metadata=provenance,
                    campaign_source_root=root,
                )

    def test_complete_generation_and_scores_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            write_jsonl(
                run_dir / "result/model/non_live/BFCL_v4_cat_result.json",
                [{"id": "cat_0", "result": []}, {"id": "cat_1", "result": []}],
            )
            write_jsonl(
                run_dir / "result/model/agentic/BFCL_v4_memory_kv_prereq_result.json",
                [{"id": "memory_kv_prereq_0", "result": []}],
            )
            write_jsonl(
                run_dir / "score/model/non_live/BFCL_v4_cat_score.json",
                [
                    {"accuracy": 0.5, "correct_count": 1, "total_count": 2},
                    {"id": "cat_1", "error_type": "wrong_tool"},
                ],
            )

            generation = validate_run.validate_generation(
                run_dir, "model", synthetic_population()
            )
            scores = validate_run.validate_scores(
                run_dir, "model", synthetic_population()
            )

            self.assertEqual(generation["status"], "pass")
            self.assertEqual(generation["actual_count"], 3)
            self.assertEqual(scores["status"], "pass")
            self.assertEqual(scores["scored_count"], 2)

    def test_missing_duplicate_and_inference_error_fail_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            error_entry = {
                "id": "cat_0",
                "result": "Error during inference: timeout",
                "traceback": "TimeoutError",
            }
            write_jsonl(
                run_dir / "result/model/non_live/BFCL_v4_cat_result.json",
                [error_entry, error_entry],
            )
            write_jsonl(
                run_dir / "result/model/agentic/BFCL_v4_memory_kv_prereq_result.json",
                [{"id": "memory_kv_prereq_0", "result": []}],
            )

            generation = validate_run.validate_generation(
                run_dir, "model", synthetic_population()
            )

            self.assertEqual(generation["status"], "fail")
            self.assertEqual(generation["missing_ids"], ["cat_1"])
            self.assertEqual(list(generation["duplicate_ids"]), ["cat_0"])
            self.assertEqual(generation["inference_error_ids"], ["cat_0"])

    def test_partial_score_header_fails_expected_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            write_jsonl(
                run_dir / "score/model/non_live/BFCL_v4_cat_score.json",
                [{"accuracy": 1.0, "correct_count": 1, "total_count": 1}],
            )

            scores = validate_run.validate_scores(
                run_dir, "model", synthetic_population()
            )

            self.assertEqual(scores["status"], "fail")
            self.assertIn("total_count 1 != expected 2", "\n".join(scores["errors"]))
            self.assertIn(
                "Scored population 0 != expected 2", "\n".join(scores["errors"])
            )


if __name__ == "__main__":
    unittest.main()
