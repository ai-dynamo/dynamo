#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from runtime_binding import (
    BindingError,
    canonical_sha256,
    make_wrapper,
    validate_continuity,
    validate_deployment,
)


FIXTURE = Path(__file__).with_name("fixtures") / "runtime-binding.json"
CONTINUITY_FIXTURE = Path(__file__).with_name("fixtures") / "runtime-continuity.json"


class RuntimeBindingContractTests(unittest.TestCase):
    def test_canonical_fixture_wraps(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        wrapper = make_wrapper(
            deployment,
            evaluator=None,
            variant="dynamo-vllm",
            campaign_phase="ab",
            endpoint="http://glm52-dynamo-vllm-frontend:8000/v1",
        )
        self.assertEqual(wrapper["file"], "runtime-binding.json")
        self.assertEqual(
            wrapper["deployment_sha256"],
            "8a8278c1f49eea2c5574d285d192747966a78ad3d88d8091396df865066185d5",
        )
        self.assertEqual(len(wrapper["content_sha256"]), 64)

    def test_fixture_and_continuity_use_the_same_canonical_digest(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        self.assertEqual(continuity["deployment_sha256"], canonical_sha256(deployment))

    def test_raw_identifiers_and_schema_drift_are_rejected(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        deployment["controller"]["uid"] = "raw-cluster-uid"
        with self.assertRaisesRegex(BindingError, "controller fields differ"):
            validate_deployment(deployment)

        deployment = json.loads(FIXTURE.read_text())
        deployment["pods"]["worker"]["node_name"] = "raw-node-name"
        with self.assertRaisesRegex(BindingError, "worker fields differ"):
            validate_deployment(deployment)

        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        continuity["pre"]["pods"]["worker"]["container_id"] = "raw-container-id"
        with self.assertRaisesRegex(BindingError, "worker fields differ"):
            validate_continuity(continuity, json.loads(FIXTURE.read_text()))

    def test_public_contract_contains_no_raw_runtime_identifier_fields(self) -> None:
        forbidden = {"uid", "node", "node_name", "pod_name", "container_id"}

        def walk(value: object) -> None:
            if isinstance(value, dict):
                self.assertTrue(forbidden.isdisjoint(value))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(json.loads(FIXTURE.read_text()))
        walk(json.loads(CONTINUITY_FIXTURE.read_text()))

    def test_variant_phase_and_hardware_are_bound(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        with self.assertRaisesRegex(BindingError, "campaign_phase"):
            validate_deployment(deployment, campaign_phase="ba")

        changed = copy.deepcopy(deployment)
        changed["hardware"]["gpu_count"] = 3
        with self.assertRaisesRegex(BindingError, "4x NVIDIA B200"):
            validate_deployment(changed)

    def test_observed_parent_image_index_is_preserved(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        spec_digest = deployment["image"].rsplit("@", 1)[1]
        worker_digest = deployment["pods"]["worker"]["image_id"].rsplit("@", 1)[1]
        self.assertNotEqual(worker_digest, spec_digest)
        validate_deployment(deployment)

        deployment["pods"]["worker"]["image_id"] = "sha256:" + "a" * 64
        with self.assertRaisesRegex(BindingError, "image_id is invalid"):
            validate_deployment(deployment)

    def test_schema_versions_and_control_plane_are_canonical(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        deployment["schema_version"] = True
        with self.assertRaisesRegex(BindingError, "schema_version"):
            validate_deployment(deployment)

        deployment = json.loads(FIXTURE.read_text())
        deployment["control_plane"]["dynamo_operator_image_digests"] *= 2
        with self.assertRaisesRegex(BindingError, "sorted and unique"):
            validate_deployment(deployment)

    def test_continuity_fixture_is_bound_and_stable(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        validate_continuity(continuity, deployment)

        changed = copy.deepcopy(continuity)
        changed["post"]["pods"]["worker"]["restart_count"] = 1
        with self.assertRaisesRegex(BindingError, "restart_count"):
            validate_continuity(changed, deployment)

        changed = copy.deepcopy(continuity)
        changed["post"]["pods"]["worker"]["restart_count"] = False
        with self.assertRaisesRegex(BindingError, "restart_count"):
            validate_continuity(changed, deployment)

    def test_command_failure_is_structural_but_not_successful(self) -> None:
        deployment = json.loads(FIXTURE.read_text())
        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        continuity["command_exit_code"] = 7
        validate_continuity(continuity, deployment, require_success=False)
        with self.assertRaisesRegex(BindingError, "did not exit zero"):
            validate_continuity(continuity, deployment)

        continuity["command_exit_code"] = False
        with self.assertRaisesRegex(BindingError, "command_exit_code"):
            validate_continuity(continuity, deployment, require_success=False)

    def test_continuity_markers_and_timestamps_cannot_contradict_snapshots(
        self,
    ) -> None:
        deployment = json.loads(FIXTURE.read_text())
        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        continuity["stable"] = False
        with self.assertRaisesRegex(BindingError, "stable marker"):
            validate_continuity(continuity, deployment, require_success=False)

        continuity = json.loads(CONTINUITY_FIXTURE.read_text())
        continuity["post_captured_at"] = "2026-07-05T00:59:59Z"
        with self.assertRaisesRegex(BindingError, "out of order"):
            validate_continuity(continuity, deployment)


if __name__ == "__main__":
    unittest.main()
