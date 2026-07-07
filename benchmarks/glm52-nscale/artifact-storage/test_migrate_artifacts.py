# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("migrate_artifacts.py")
SPEC = importlib.util.spec_from_file_location("migrate_artifacts", MODULE_PATH)
assert SPEC and SPEC.loader
migrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migrate)


class MigrateArtifactsTest(unittest.TestCase):
    def test_copy_verify_and_idempotent_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination_parent = root / "destination"
            (source / "nested" / "empty").mkdir(parents=True)
            (source / "nested" / "result.json").write_text('{"score": 0.872}\n')
            (source / "link").symlink_to("nested/result.json")

            required_hash = migrate.file_sha256(source / "nested" / "result.json")
            requirements = {
                "required_absent": ("swebench/results/vllm-serve-ab-r2/verified",),
                "required_files": {"nested/result.json": required_hash},
            }
            first = migrate.copy_and_verify(
                source, destination_parent, "pvc-uid", **requirements
            )
            second = migrate.copy_and_verify(
                source, destination_parent, "pvc-uid", **requirements
            )

            self.assertEqual(first["state"], "complete")
            self.assertEqual(second["state"], "already_complete")
            self.assertEqual(first["marker_sha256"], second["marker_sha256"])
            self.assertEqual(first["marker"], second["marker"])
            destination = destination_parent / migrate.DESTINATION_NAME
            self.assertEqual(
                migrate.tree_identity(source),
                migrate.tree_identity(destination, ignore_marker=True),
            )

    def test_rejects_unmarked_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination" / migrate.DESTINATION_NAME
            source.mkdir()
            destination.mkdir(parents=True)
            (destination / "partial").write_text("unsafe")
            with self.assertRaisesRegex(ValueError, "unmarked destination"):
                migrate.copy_and_verify(source, root / "destination", "pvc-uid")

    def test_recheck_rejects_source_or_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination_parent = root / "destination"
            source.mkdir()
            (source / "result.json").write_text(json.dumps({"value": 1}))
            migrate.copy_and_verify(source, destination_parent, "pvc-uid")

            with self.assertRaisesRegex(ValueError, "source PVC differs"):
                migrate.copy_and_verify(source, destination_parent, "other-pvc")
            (source / "result.json").write_text(json.dumps({"value": 2}))
            with self.assertRaisesRegex(ValueError, "source tree differs"):
                migrate.copy_and_verify(source, destination_parent, "pvc-uid")

    def test_rejects_dirty_replay_path_or_required_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            target = source / "swebench" / "results" / "vllm-serve-ab-r2" / "verified"
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "clean replay path exists"):
                migrate.copy_and_verify(
                    source,
                    root / "destination",
                    "pvc-uid",
                    required_absent=("swebench/results/vllm-serve-ab-r2/verified",),
                )
            target.rmdir()
            artifact = source / "task-images.json"
            artifact.write_text("manifest")
            with self.assertRaisesRegex(ValueError, "required artifact hash differs"):
                migrate.copy_and_verify(
                    source,
                    root / "destination",
                    "pvc-uid",
                    required_files={"task-images.json": "0" * 64},
                )

    def test_requirements_reject_symlinked_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            (outside / "task-images.json").write_text("outside")
            (source / "swebench").symlink_to(outside, target_is_directory=True)
            digest = migrate.file_sha256(outside / "task-images.json")
            with self.assertRaisesRegex(ValueError, "traverses a symlink"):
                migrate.copy_and_verify(
                    source,
                    root / "destination",
                    "pvc-uid",
                    required_files={"swebench/task-images.json": digest},
                )


if __name__ == "__main__":
    unittest.main()
