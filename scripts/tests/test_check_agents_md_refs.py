# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/check_agents_md_refs.py.

Run: python3 scripts/tests/test_check_agents_md_refs.py
"""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_agents_md_refs.py"
spec = importlib.util.spec_from_file_location("check_agents_md_refs", SCRIPT)
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)


class ClassifyTest(unittest.TestCase):
    def test_paths(self):
        cases = {
            "lib/router/kv-router/src/lib.rs": "lib/router/kv-router/src/lib.rs",
            "indexer/": "indexer/",
            "args.rs": "args.rs",
            "kv_router.rs:120-140": "kv_router.rs",
            "docs/guide.md#setup": "docs/guide.md",
            "tests/router/test_x.py::test_y": "tests/router/test_x.py",
            ".github/workflows/pre-merge.yml": ".github/workflows/pre-merge.yml",
            "../../lib/llm/src/lib.rs": "../../lib/llm/src/lib.rs",
        }
        for span, path in cases.items():
            with self.subTest(span=span):
                self.assertEqual(refs.classify(span), ("path", path))

    def test_identifiers(self):
        cases = {
            "KvIndexer::find_matches": ["KvIndexer", "find_matches"],
            "find_matches()": ["find_matches"],
            "tokio::select!": ["select"],
            "Arc<dyn AsyncEngineContext>": ["AsyncEngineContext"],
            "reconcileWorkload": ["reconcileWorkload"],
            "fn score(&self, overlap: OverlapScores) -> f64": ["OverlapScores"],
            "KVIndexer": ["KVIndexer"],
        }
        for span, words in cases.items():
            with self.subTest(span=span):
                self.assertEqual(refs.classify(span), ("identifiers", words))

    def test_skipped(self):
        for span in [
            "cargo test -p dynamo-kv-router",
            "python3 -m pytest tests/router",
            "--block-size",
            "DYN_ROUTER_MODE",
            "block_size=16",
            '"model_name"',
            "https://github.com/ai-dynamo/dynamo",
            "sigs.k8s.io/cluster-api",
            "nvidia.com/v1beta1",
            "kvbm_kernels_launch_*",
            "files/upgrade-preflight/<check-id>/",
            "run(<YourEngine>)",
            "/v1/chat/completions",
            "~/.cargo/config.toml",
            ".rs",
            "state",
            "Worker",
            "Option<u64>",
        ]:
            with self.subTest(span=span):
                self.assertIsNone(refs.classify(span))


class CodeSpansTest(unittest.TestCase):
    def test_skips_fences_and_marked_lines(self):
        text = "\n".join(
            [
                "Use `alpha_one` here.",
                "```rust",
                "let x = `not_a_span`;",
                "```",
                "~~~",
                "`also_fenced`",
                "~~~",
                "Upstream `kubernetes-sigs/x.go` <!-- agents-md-refs: ignore -->",
                "Both `beta_two` and `GammaThree`.",
            ]
        )
        self.assertEqual(
            list(refs.code_spans(text)),
            [(1, "alpha_one"), (9, "beta_two"), (9, "GammaThree")],
        )


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.git("init", "-q")
        self.write("lib/foo/Cargo.toml", "[package]\nname = 'foo'\n")
        self.write(
            "lib/foo/src/engine/scorer.rs",
            "pub struct WorkerScorer;\nimpl WorkerScorer { fn score_worker() {} }\n",
        )
        self.write("lib/foo/src/lib.rs", "macro_rules! define_score { () => {} }\n")
        self.write("lib/foo/README.md", "`only_in_docs` lives here.\n")

    def tearDown(self):
        self._tmp.cleanup()

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        self.git("add", path)

    def check(self, agents_text, scopes=()):
        self.write("lib/foo/AGENTS.md", agents_text)
        return refs.Checker(self.root).check(scopes)[2]

    def test_existing_references_pass(self):
        errors = self.check(
            "- `src/engine/scorer.rs`, `engine/`, `scorer.rs`, `lib/foo/src/lib.rs`\n"
            "- `WorkerScorer::score_worker`, `define_score!`\n"
            "- `cargo test -p foo`, `FOO_ENV`, `RUST_LOG=debug`\n"
        )
        self.assertEqual(errors, [])

    def test_missing_path_and_identifier_fail(self):
        errors = self.check("- `src/engine/gone.rs` and `WorkerScorer::removed_fn`\n")
        self.assertEqual(
            errors,
            [
                "lib/foo/AGENTS.md:1: identifier `removed_fn` in "
                "`WorkerScorer::removed_fn` not found",
                "lib/foo/AGENTS.md:1: path `src/engine/gone.rs` does not exist",
            ],
        )

    def test_markdown_mentions_do_not_count(self):
        errors = self.check("- `only_in_docs`\n")
        self.assertEqual(
            errors, ["lib/foo/AGENTS.md:1: identifier `only_in_docs` not found"]
        )

    def test_code_rename_fails_without_touching_agents_md(self):
        self.assertEqual(self.check("- `WorkerScorer::score_worker`\n"), [])
        self.write(
            "lib/foo/src/engine/scorer.rs",
            "pub struct WorkerScorer;\nimpl WorkerScorer { fn rank_worker() {} }\n",
        )
        errors = refs.Checker(self.root).check()[2]
        self.assertEqual(
            errors,
            [
                "lib/foo/AGENTS.md:1: identifier `score_worker` in "
                "`WorkerScorer::score_worker` not found"
            ],
        )

    def test_reads_the_index_not_the_worktree(self):
        self.check("- `scorer.rs`\n")
        (self.root / "lib/foo/src/engine/scorer.rs").unlink()
        self.assertEqual(refs.Checker(self.root).check()[2], [])
        self.git("rm", "-q", "--cached", "lib/foo/src/engine/scorer.rs")
        self.assertEqual(
            refs.Checker(self.root).check()[2],
            ["lib/foo/AGENTS.md:1: path `scorer.rs` does not exist"],
        )

    def test_scope_limits_files_and_rejects_empty_scope(self):
        self.write("lib/bar/AGENTS.md", "- `bar_missing_fn`\n")
        self.assertEqual(self.check("- `scorer.rs`\n", scopes=["lib/foo"]), [])
        self.assertEqual(
            refs.Checker(self.root).check(["lib/foo", "lib/typo"])[2],
            ["lib/typo: no tracked AGENTS.md under this scope"],
        )


if __name__ == "__main__":
    unittest.main()
