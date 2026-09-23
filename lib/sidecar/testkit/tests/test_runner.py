# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTKIT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sidecar_runner", TESTKIT / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LaneSelectionTests(unittest.TestCase):
    def test_inventory_requires_a_lane_before_filtering(self):
        for suffix in (
            "missing",
            "__sidecar_lane_typo",
            "__sidecar_lane_pre_merge::extra",
        ):
            with self.subTest(suffix=suffix), self.assertRaises(RuntimeError):
                runner.inventory(f"unit_requests::case::{suffix}: test\n", "vllm")
        marked = "unit_requests::shared::case::__sidecar_lane_pre_merge: test\n"
        with self.assertRaises(RuntimeError):
            runner.inventory(marked * 2, "vllm")
        with self.assertRaises(RuntimeError):
            runner.inventory("legacy::case: test\n", "vllm")
        cases = runner.inventory(marked + "legacy::case: test\n", "vllm")
        self.assertEqual(cases[0]["category"], "shared")
        self.assertEqual(runner.select_cases(cases, "pre-merge"), cases)
        self.assertEqual(
            runner.select_cases([{**cases[0], "lane": "nightly"}], "pre-merge"), []
        )

    def test_export_requires_the_complete_supported_suite(self):
        specs = runner.specifications()
        entries = [
            {
                "name": spec["name"],
                "path": spec["name"],
                "suite": "unit",
                "framework": spec["framework"],
                "cases": [],
            }
            for spec in specs
        ]
        manifest = {"version": runner.MANIFEST_VERSION, "entries": entries}
        self.assertEqual(runner.select_entries(manifest, specs), entries)
        for invalid_entries in (
            entries[:1],
            entries * 2,
            [{**entries[0], "name": "unknown"}],
        ):
            with self.subTest(entries=invalid_entries), self.assertRaises(RuntimeError):
                runner.select_entries({**manifest, "entries": invalid_entries}, specs)
        invalid = copy.deepcopy(manifest)
        invalid["entries"][0]["framework"] = "vllm"
        with self.assertRaises(RuntimeError):
            runner.select_entries(invalid, specs)

    def test_cli_keeps_legacy_aliases_and_rejects_unimplemented_backends(self):
        for level, lane in (
            ("unit", "all"),
            ("all", "all"),
            ("pre-merge", "pre-merge"),
        ):
            self.assertEqual(runner.arguments(["--level", level]).lane, lane)
        for args in (
            ["--framework", "sglang"],
            ["--level", "unit", "--lane", "nightly"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.arguments(args)

    def test_compiled_lane_names_and_real_libtest_selection(self):
        with tempfile.TemporaryDirectory(prefix="sidecar-lanes-") as directory:
            source = Path(directory) / "lanes.rs"
            binary = Path(directory) / "lanes"
            macro = TESTKIT / "tests/unit/lane.rs"
            source.write_text(
                f'#[macro_use] #[path = "{macro}"] mod lane;\n'
                "const VALUE: u8 = 1;\n"
                "mod unit_fixture {\n"
                "use super::VALUE;\n"
                "sidecar_test! { lane: pre_merge; #[test] fn pre() { assert_eq!(VALUE, 1); } }\n"
                "sidecar_test! { lane: post_merge; #[test] fn post() -> Result<(), ()> { assert_eq!(VALUE, 1); Ok(()) } }\n"
                "sidecar_test! { lane: nightly; #[test] fn night() { assert_eq!(VALUE, 1); } }\n"
                "}\n"
                "#[test] fn retained() { assert_eq!(VALUE, 1); }\n"
            )
            subprocess.run(
                ["rustc", "--edition=2024", "--test", str(source), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            cases = runner.collect(binary, "common")
            for lane, expected in (
                ("pre-merge", 1),
                ("post-merge", 2),
                ("nightly", 3),
                ("all", 3),
            ):
                with self.subTest(lane=lane):
                    selected = runner.select_cases(cases, lane)
                    self.assertEqual(len(selected), expected)
                    runner.execute(binary, selected)
                    result = subprocess.run(
                        [str(binary), *runner.libtest_args(lane)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertIn(
                        f"{expected + 1} passed; 0 failed; 0 ignored", result.stdout
                    )
            original = source.read_text()
            source.write_text(
                original.replace(
                    "lane: nightly; #[test]", "lane: nightly; #[ignore] #[test]"
                )
            )
            subprocess.run(
                ["rustc", "--edition=2024", "--test", str(source), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            with self.assertRaisesRegex(RuntimeError, "must not be ignored"):
                runner.collect(binary, "common")
            source.write_text(original.replace("lane: nightly", "lane: typo"))
            rejected = subprocess.run(
                ["rustc", "--edition=2024", "--test", str(source), "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("no rules expected", rejected.stderr)

    def test_workspace_lane_filters_preserve_custom_and_legacy_targets(self):
        with tempfile.TemporaryDirectory(prefix="sidecar-workspace-") as directory:
            root = Path(directory)
            executed = root / "executed"
            executed.mkdir()
            owners = [spec["package"] for spec in runner.specifications()]
            members = ", ".join(f'"{name}"' for name in [*owners, "unrelated"])
            (root / "Cargo.toml").write_text(
                f'[workspace]\nmembers = [{members}]\nresolver = "3"\n'
            )
            (root / "rust-toolchain.toml").write_bytes(
                (TESTKIT.parents[2] / "rust-toolchain.toml").read_bytes()
            )
            record = (
                "fn record(name: &str) {\n"
                'let directory = std::env::var("SIDECAR_TEST_RUN_DIR").unwrap();\n'
                'std::fs::write(std::path::Path::new(&directory).join(name), "ran").unwrap();\n'
                "}\n"
            )
            macro = TESTKIT / "tests/unit/lane.rs"
            for name in [*owners, "unrelated"]:
                package = root / name
                (package / "src").mkdir(parents=True)
                manifest = (
                    f'[package]\nname = "{name}"\nversion = "0.0.0"\nedition = "2024"\n'
                    "[features]\ntonic-v14 = []\nextra = []\n"
                )
                source = (
                    "#[cfg(test)]\n"
                    + record
                    + (
                        '#[test] fn legacy() { record(&format!("legacy-{}", env!("CARGO_PKG_NAME"))); }\n'
                    )
                )
                if name in owners:
                    source += (
                        f'#[cfg(test)] #[macro_use] #[path = "{macro}"] mod lane;\n'
                        "#[cfg(test)] mod unit_fixture {\nuse super::record;\n"
                        'sidecar_test! { lane: pre_merge; #[test] fn pre() { record(env!("CARGO_PKG_NAME")); } }\n'
                        'sidecar_test! { lane: post_merge; #[test] fn post() { record("post-merge"); panic!("post-merge leaked"); } }\n'
                        'sidecar_test! { lane: nightly; #[test] fn night() { record("nightly"); panic!("nightly leaked"); } }\n'
                        "}\n"
                    )
                else:
                    manifest += (
                        '[[bench]]\nname = "custom"\nharness = false\n'
                        '[[bench]]\nname = "optional"\nharness = false\nrequired-features = ["extra"]\n'
                    )
                    (package / "benches").mkdir()
                    (package / "benches/custom.rs").write_text(
                        record
                        + 'fn main() { assert!(!std::env::args().any(|arg| arg == "--skip")); record("custom-bench"); }\n'
                    )
                    (package / "benches/optional.rs").write_text(
                        'compile_error!("disabled benchmark was selected");\n'
                    )
                (package / "Cargo.toml").write_text(manifest)
                (package / "src/lib.rs").write_text(source)
            with mock.patch.dict(
                os.environ,
                {
                    "CARGO_TARGET_DIR": str(root / "target"),
                    "CARGO_NET_OFFLINE": "true",
                    "SIDECAR_TEST_RUN_DIR": str(executed),
                },
            ):
                subprocess.run(
                    ["cargo", "generate-lockfile", "--offline"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                runner.workspace_tests(root, "pre-merge")
            self.assertEqual(
                {path.name for path in executed.iterdir()},
                {
                    *owners,
                    *(f"legacy-{name}" for name in owners),
                    "legacy-unrelated",
                    "custom-bench",
                },
            )


if __name__ == "__main__":
    unittest.main()
