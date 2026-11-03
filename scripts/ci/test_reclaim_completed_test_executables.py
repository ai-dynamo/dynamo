# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem regressions for Cargo-record-scoped executable cleanup."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reclaim_completed_test_executables import execute_cleanup, plan_cleanup


class ReclaimTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "workspace"
        self.target = self.workspace / "target"
        self.deps = self.target / "debug" / "deps"
        self.deps.mkdir(parents=True)

    def executable(self, name, content=b"test executable"):
        path = self.deps / (name.replace("-", "_") + "-0123456789abcdef")
        path.write_bytes(content)
        path.chmod(0o700)
        return path

    def artifact(self, name, path, kind="lib", test=True, **target_fields):
        return {
            "reason": "compiler-artifact",
            "target": {"name": name, "kind": [kind], **target_fields},
            "profile": {"test": test},
            "executable": str(path) if path is not None else None,
            "filenames": [str(path)] if path is not None else [],
        }

    def messages(self, records, name="cargo.jsonl", success=True):
        path = self.workspace / name
        lines = [json.dumps(row) for row in records]
        lines.append(json.dumps({"reason": "build-finished", "success": success}))
        path.write_text("\n".join(lines) + "\n")
        return path

    def plan(self, *logs):
        return plan_cleanup(list(logs), self.workspace, self.target)

    def test_unit_bin_and_hyphenated_integration_are_removed_only_when_recorded(self):
        records = []
        expected_bytes = 0
        expected_blocks = 0
        for name, kind in [
            ("library", "lib"),
            ("tool", "bin"),
            ("http-transcriptions", "test"),
        ]:
            path = self.executable(name)
            info = path.stat()
            expected_bytes += info.st_size
            expected_blocks += info.st_blocks * 512
            records.append(self.artifact(name, path, kind))
        unrelated = self.executable("not-recorded")
        sentinels = [
            self.deps / name
            for name in ["libshared.so", "liba.rlib", "liba.rmeta", "test.d"]
        ]
        for path in sentinels:
            path.write_bytes(b"keep")
        result = execute_cleanup(self.plan(self.messages(records)))
        self.assertEqual(result["removed_count"], 3)
        self.assertEqual(result["removed_logical_bytes"], expected_bytes)
        self.assertEqual(result["removed_allocated_bytes"], expected_blocks)
        self.assertTrue(unrelated.exists())
        self.assertTrue(all(path.read_bytes() == b"keep" for path in sentinels))

    def test_production_proc_macro_converter_bench_and_example_are_retained(self):
        records = []
        paths = []
        for name, kind, test in [
            ("production", "bin", False),
            ("macro", "proc-macro", True),
            ("request_trace_to_mooncake", "bin", True),
            ("benchmark", "bench", True),
            ("example", "example", True),
        ]:
            path = self.executable(name)
            paths.append(path)
            records.append(self.artifact(name, path, kind, test))
        result = execute_cleanup(self.plan(self.messages(records)))
        self.assertEqual(result["removed_count"], 0)
        self.assertEqual(result["removed_allocated_bytes"], 0)
        self.assertTrue(all(path.exists() for path in paths))

    def test_noise_is_counted_but_paths_in_text_are_never_selected(self):
        selected = self.executable("selected")
        retained = self.executable("only-mentioned")
        log = self.messages([self.artifact("selected", selected)])
        log.write_text(
            f"tool says remove {retained}\n" + '{"tool": "stdout"}\n' + log.read_text()
        )
        result = execute_cleanup(self.plan(log))
        self.assertEqual(result["ignored_stdout_lines"], 2)
        self.assertFalse(selected.exists())
        self.assertTrue(retained.exists())

    def test_reason_bearing_tool_json_after_terminal_is_noise(self):
        selected = self.executable("selected")
        retained = self.executable("only-mentioned")
        log = self.messages([self.artifact("selected", selected)])
        noise = {"reason": "tool-progress", "executable": str(retained)}
        with log.open("a") as stream:
            stream.write(json.dumps(noise) + "\n")
        result = execute_cleanup(self.plan(log))
        self.assertEqual(result["ignored_stdout_lines"], 1)
        self.assertEqual(result["removed_count"], 1)
        self.assertFalse(selected.exists())
        self.assertTrue(retained.exists())

    def test_failed_incomplete_and_malformed_artifacts_fail_before_any_unlink(self):
        good = self.executable("good")
        bad_artifact = self.artifact("bad", self.executable("bad"))
        del bad_artifact["profile"]
        for records, success in [
            ([self.artifact("good", good)], False),
            ([self.artifact("good", good), bad_artifact], True),
        ]:
            with self.subTest(records=records, success=success):
                with self.assertRaises(ValueError):
                    self.plan(self.messages(records, success=success))
                self.assertTrue(good.exists())
        incomplete = self.workspace / "incomplete.jsonl"
        incomplete.write_text(json.dumps(self.artifact("good", good)) + "\n")
        with self.assertRaises(ValueError):
            self.plan(incomplete)
        self.assertTrue(good.exists())

    def test_outside_nonhashed_symlink_and_build_paths_fail_before_any_unlink(self):
        good = self.executable("good")
        outside = Path(self.temporary.name) / "outside-0123456789abcdef"
        outside.write_bytes(b"outside")
        outside.chmod(0o700)
        symlink = self.deps / "linked-0123456789abcdef"
        symlink.symlink_to(outside)
        plain = self.deps / "plain"
        plain.write_bytes(b"plain")
        plain.chmod(0o700)
        build = self.target / "debug/build/package/deps/build_test-0123456789abcdef"
        build.parent.mkdir(parents=True)
        build.write_bytes(b"keep")
        build.chmod(0o700)
        for name, path in [
            ("outside", outside),
            ("linked", symlink),
            ("plain", plain),
            ("build-test", build),
        ]:
            with self.subTest(path=path):
                log = self.messages(
                    [self.artifact("good", good), self.artifact(name, path)]
                )
                with self.assertRaises(ValueError):
                    self.plan(log)
                self.assertTrue(good.exists())

    def test_duplicates_missing_and_hardlinks_do_not_inflate_removed_bytes(self):
        good = self.executable("good")
        linked = self.executable("shared")
        alias = self.workspace / "shared-alias"
        os.link(linked, alias)
        missing = self.deps / "missing-0123456789abcdef"
        record = self.artifact("good", good)
        plan = self.plan(
            self.messages(
                [
                    record,
                    record,
                    self.artifact("missing", missing),
                    self.artifact("shared", linked),
                ]
            )
        )
        result = execute_cleanup(plan)
        self.assertEqual(result["removed_count"], 1)
        self.assertEqual(result["removed_logical_bytes"], len(b"test executable"))
        self.assertEqual(result["duplicate_records"], 1)
        self.assertEqual(result["skipped_missing"], 1)
        self.assertEqual(result["skipped_hardlinks"], 1)
        self.assertEqual(linked.read_bytes(), alias.read_bytes())

    def test_changed_identity_is_detected_for_all_files_before_first_unlink(self):
        first = self.executable("first")
        second = self.executable("second")
        plan = self.plan(
            self.messages(
                [self.artifact("first", first), self.artifact("second", second)]
            )
        )
        second.unlink()
        second.write_bytes(b"replacement")
        second.chmod(0o700)
        with self.assertRaises(ValueError):
            execute_cleanup(plan)
        self.assertTrue(first.exists())
        self.assertEqual(second.read_bytes(), b"replacement")

    def test_parent_symlink_added_after_preflight_cannot_redirect_unlink(self):
        path = self.executable("selected")
        plan = self.plan(self.messages([self.artifact("selected", path)]))
        moved = Path(self.temporary.name) / "moved-deps"
        self.deps.rename(moved)
        self.deps.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            execute_cleanup(plan)
        self.assertTrue((moved / path.name).exists())

    def test_cli_observes_real_filesystems_without_claiming_free_space(self):
        path = self.executable("selected")
        log = self.messages([self.artifact("selected", path)])
        command = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("reclaim_completed_test_executables.py")),
            "--cargo-messages",
            str(log),
            "--workspace",
            str(self.workspace),
            "--target-dir",
            str(self.target),
        ]
        result = subprocess.run(
            command,
            env=dict(
                os.environ, HOME=str(self.workspace), RUNNER_TEMP=str(self.workspace)
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        report = json.loads(result.stdout)
        self.assertFalse(path.exists())
        self.assertEqual(report["removed_count"], 1)
        self.assertTrue(
            report["allocated_bytes_are_file_accounting_not_a_free_space_claim"]
        )
        for observations in [report["filesystem_before"], report["filesystem_after"]]:
            self.assertEqual(set(observations), {"target", "HOME", "RUNNER_TEMP"})
            for row in observations.values():
                self.assertIsInstance(row["device"], int)
                self.assertGreaterEqual(row["available_bytes"], 0)
                self.assertGreaterEqual(row["available_inodes"], 0)


if __name__ == "__main__":
    unittest.main()
