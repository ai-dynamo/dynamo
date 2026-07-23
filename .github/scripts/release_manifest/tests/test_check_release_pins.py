# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for check_release_pins.py.

Run: python3 -m unittest discover -s .github/scripts/release_manifest/tests -v
"""

import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_release_pins as crp  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_context.yaml"


def manifest_yaml(pins: str, version: str = "1.4.0") -> str:
    header = textwrap.dedent(
        f"""\
        schema_version: 1
        release:
          version: "{version}"
        pins:
        """
    )
    return header + pins


PIN_BLOCK = textwrap.dedent(
    """\
      - name: vllm-cuda-runtime-image
        file: mini_context.yaml
        path: vllm.cuda13.0.runtime_image_tag
        required: {vllm_tag}
        auto_upgrade: {vllm_auto}
      - name: sglang-nixl
        file: mini_context.yaml
        path: sglang.nixl_ref
        required: {sglang_nixl}
        auto_upgrade: true
      - name: modelexpress
        file: mini_context.yaml
        path: sglang.modelexpress_version
        required: "0.4.0"
        auto_upgrade: false
    """
)


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        shutil.copy(FIXTURE, self.root / "mini_context.yaml")
        for var in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY"):
            old = os.environ.pop(var, None)
            if old is not None:
                self.addCleanup(os.environ.__setitem__, var, old)

    def write_manifest(
        self, vllm_tag="v0.24.0-ubuntu2404", vllm_auto="true", sglang_nixl="v1.0.1"
    ):
        text = manifest_yaml(
            PIN_BLOCK.format(
                vllm_tag=vllm_tag, vllm_auto=vllm_auto, sglang_nixl=sglang_nixl
            )
        )
        path = self.root / "release-manifest.yaml"
        path.write_text(text)
        return path

    def run_main(self, *args, manifest=None):
        argv = list(args) + [
            "--manifest",
            str(manifest or self.root / "release-manifest.yaml"),
            "--repo-root",
            str(self.root),
        ]
        return crp.main(argv)


class TestCheck(BaseCase):
    def test_all_pass(self):
        self.write_manifest()
        self.assertEqual(self.run_main("--check", "--version", "1.4.0"), 0)

    def test_github_outputs(self):
        self.write_manifest()
        out = self.root / "gh_output"
        os.environ["GITHUB_OUTPUT"] = str(out)
        self.addCleanup(os.environ.pop, "GITHUB_OUTPUT", None)
        self.assertEqual(self.run_main("--check"), 0)
        content = out.read_text()
        self.assertIn("blocked=false", content)
        self.assertIn("auto_upgrades=[]", content)
        self.assertIn("pin_table<<", content)

    def test_stale_auto_upgradable_passes_check(self):
        self.write_manifest(vllm_tag="v0.25.0-ubuntu2404")
        self.assertEqual(self.run_main("--check"), 0)

    def test_stale_auto_upgradable_blocks_strict(self):
        self.write_manifest(vllm_tag="v0.25.0-ubuntu2404")
        self.assertEqual(self.run_main("--check", "--strict"), 1)

    def test_stale_not_auto_upgradable_blocks(self):
        self.write_manifest(vllm_tag="v0.25.0-ubuntu2404", vllm_auto="false")
        self.assertEqual(self.run_main("--check"), 1)

    def test_version_mismatch_blocks(self):
        self.write_manifest()
        self.assertEqual(self.run_main("--check", "--version", "1.5.0"), 1)

    def test_malformed_manifest_exits_2(self):
        path = self.root / "release-manifest.yaml"
        path.write_text("schema_version: 2\n")
        self.assertEqual(self.run_main("--check"), 2)

    def test_missing_path_exits_2(self):
        path = self.root / "release-manifest.yaml"
        path.write_text(
            manifest_yaml(
                textwrap.dedent(
                    """\
                      - name: ghost
                        file: mini_context.yaml
                        path: vllm.cuda13.0.no_such_key
                        required: x
                        auto_upgrade: true
                    """
                )
            )
        )
        self.assertEqual(self.run_main("--check"), 2)


class TestApply(BaseCase):
    def test_apply_rewrites_only_target_token(self):
        self.write_manifest(vllm_tag="v0.25.0-ubuntu2404")
        before = (self.root / "mini_context.yaml").read_text()
        self.assertEqual(self.run_main("--apply"), 0)
        after = (self.root / "mini_context.yaml").read_text()
        self.assertEqual(
            after,
            before.replace(
                "runtime_image_tag: v0.24.0-ubuntu2404  # keep in sync with upstream",
                "runtime_image_tag: v0.25.0-ubuntu2404  # keep in sync with upstream",
            ),
        )
        # Idempotent: second apply changes nothing.
        self.assertEqual(self.run_main("--apply"), 0)
        self.assertEqual((self.root / "mini_context.yaml").read_text(), after)

    def test_apply_disambiguates_nixl_depths(self):
        # Upgrade the top-level sglang.nixl_ref; the xpu one must not move.
        self.write_manifest(sglang_nixl="v1.2.0")
        self.assertEqual(self.run_main("--apply"), 0)
        after = (self.root / "mini_context.yaml").read_text()
        self.assertIn("  nixl_ref: v1.2.0", after)
        self.assertIn("    nixl_ref: v1.1.0", after)

    def test_apply_preserves_quote_style(self):
        path = self.root / "release-manifest.yaml"
        path.write_text(
            manifest_yaml(
                textwrap.dedent(
                    """\
                      - name: modelexpress
                        file: mini_context.yaml
                        path: sglang.modelexpress_version
                        required: "0.5.0"
                        auto_upgrade: true
                    """
                )
            )
        )
        self.assertEqual(self.run_main("--apply"), 0)
        after = (self.root / "mini_context.yaml").read_text()
        self.assertIn('modelexpress_version: "0.5.0"', after)

    def test_apply_refuses_when_blocked(self):
        self.write_manifest(vllm_tag="v0.25.0-ubuntu2404", vllm_auto="false")
        before = (self.root / "mini_context.yaml").read_text()
        self.assertEqual(self.run_main("--apply"), 1)
        self.assertEqual((self.root / "mini_context.yaml").read_text(), before)


class TestLint(BaseCase):
    def test_lint_passes_and_ignores_values(self):
        # Values disagree with the tree, but lint only checks resolution.
        self.write_manifest(vllm_tag="v9.9.9", sglang_nixl="v9.9.9")
        self.assertEqual(self.run_main("--lint"), 0)

    def test_lint_fails_on_renamed_key(self):
        path = self.root / "release-manifest.yaml"
        path.write_text(
            manifest_yaml(
                textwrap.dedent(
                    """\
                      - name: ghost
                        file: mini_context.yaml
                        path: dynamo.renamed_key
                        required: x
                        auto_upgrade: true
                    """
                )
            )
        )
        self.assertEqual(self.run_main("--lint"), 1)


class TestInternals(BaseCase):
    def test_resolve_dotted_key_with_dots(self):
        data = {"vllm": {"cuda13.0": {"runtime_image_tag": "t"}}}
        keys, value = crp.resolve_dotted(data, "vllm.cuda13.0.runtime_image_tag", "x")
        self.assertEqual(keys, ["vllm", "cuda13.0", "runtime_image_tag"])
        self.assertEqual(value, "t")

    def test_resolve_dotted_ambiguity(self):
        data = {"a.b": 1, "a": {"b": 2}}
        with self.assertRaises(crp.ManifestError):
            crp.resolve_dotted(data, "a.b", "x")

    def test_set_yaml_scalar_zero_matches_raises(self):
        with self.assertRaises(crp.ManifestError):
            crp.set_yaml_scalar("a: 1\n", ["missing"], "v", "x")

    def test_set_yaml_scalar_url_value(self):
        text = "dynamo:\n  repo_url: https://github.com/example/repo.git\n"
        out = crp.set_yaml_scalar(
            text, ["dynamo", "repo_url"], "https://example.com/x.git", "x"
        )
        self.assertEqual(out, "dynamo:\n  repo_url: https://example.com/x.git\n")


if __name__ == "__main__":
    unittest.main()
