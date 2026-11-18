# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/check_agents_md_refs.py.

Run: python3 scripts/tests/test_check_agents_md_refs.py
"""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

# Git exports GIT_DIR and GIT_INDEX_FILE to commit hooks. Left set, the fixture
# repository's `git init` and `git add` would act on the repository that is
# being committed to.
for name in [name for name in os.environ if name.startswith("GIT_")]:
    del os.environ[name]

SCRIPT = Path(__file__).resolve().parents[1] / "check_agents_md_refs.py"
spec = importlib.util.spec_from_file_location("check_agents_md_refs", SCRIPT)
refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refs)


class SpanTest(unittest.TestCase):
    def test_classify(self):
        cases = {
            "lib/foo/src/lib.rs": ("lib/foo/src/lib.rs", []),
            "../engine/": ("../engine/", []),
            "scorer.rs": ("scorer.rs", []),
            "WidgetScorer::score_widget()": (None, ["WidgetScorer", "score_widget"]),
            "fn score(&self, input: WidgetInput) -> f64": (None, ["WidgetInput"]),
            "reconcileWidget": (None, ["reconcileWidget"]),
            "HTTPWidget": (None, ["HTTPWidget"]),
            "MAX_WIDGETS": (None, ["MAX_WIDGETS"]),
            "cargo bench --bench widget_bench": (None, ["widget_bench"]),
            "WIDGET_*": (None, []),
            "Worker": (None, []),
            "TTFT": (None, []),
            "Option<u64>": (None, []),
            "--block-size": (None, []),
        }
        for span, expected in cases.items():
            with self.subTest(span=span):
                self.assertEqual(refs.classify(span), expected)

    def test_code_spans_skip_fences_and_marked_lines(self):
        text = "\n".join(
            [
                "Use `alpha_one` and ``beta_two``.",
                "```rust",
                "let x = `fenced_one`;",
                "```",
                "~~~",
                "~~~ not a closing fence",
                "`fenced_two`",
                "~~~",
                "Upstream `kubernetes-sigs/x.go` <!-- agents-md-refs: ignore -->",
                "Then `GammaThree`.",
            ]
        )
        self.assertEqual(
            list(refs.code_spans(text)),
            [(1, "alpha_one"), (1, "beta_two"), (10, "GammaThree")],
        )


class CheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.git("init", "-q")
        self.write(
            "lib/foo/src/engine/scorer.rs",
            "pub struct WidgetScorer;\n"
            "impl WidgetScorer { fn score_widget() {} }\n"
            "const MAX_WIDGETS: usize = 4;\n",
        )
        self.write("lib/foo/README.md", "`only_in_docs` lives here.\n")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        self.git("add", path)

    def check(self, agents_text=None, scopes=(".",)):
        if agents_text is not None:
            self.write("lib/foo/AGENTS.md", agents_text)
        return refs.check(self.root, scopes)[2]

    def test_existing_references_pass_and_missing_ones_fail(self):
        errors = self.check(
            "- `src/engine/scorer.rs`, `engine/`, `scorer.rs`, `lib/foo/README.md`\n"
            "- `WidgetScorer::score_widget`, `MAX_WIDGETS`\n"
        )
        self.assertEqual(errors, [])
        errors = self.check(
            "- `src/engine/gone.rs` and `WidgetScorer::removed_fn`\n"
            "- `only_in_docs`\n"
        )
        self.assertEqual(
            errors,
            [
                "lib/foo/AGENTS.md:1: path `src/engine/gone.rs` does not exist",
                "lib/foo/AGENTS.md:1: identifier `removed_fn` in "
                "`WidgetScorer::removed_fn` not found",
                "lib/foo/AGENTS.md:2: identifier `only_in_docs` not found",
            ],
        )

    def test_staged_code_rename_fails_without_an_agents_md_edit(self):
        self.assertEqual(self.check("- `WidgetScorer::score_widget`\n"), [])
        scorer = self.root / "lib/foo/src/engine/scorer.rs"
        scorer.write_text(scorer.read_text().replace("score_widget", "rank_widget"))
        self.assertEqual(self.check(), [], "an unstaged rename is not in the index")
        self.git("add", "-A")
        self.assertEqual(
            self.check(),
            [
                "lib/foo/AGENTS.md:1: identifier `score_widget` in "
                "`WidgetScorer::score_widget` not found"
            ],
        )

    def test_scopes(self):
        self.write("lib/foo/AGENTS.md", "- `scorer.rs`\n")
        self.write("lib/bar/AGENTS.md", "- `bar_missing_fn`\n")
        self.assertEqual(self.check(scopes=["./lib/foo/"]), [])
        self.assertEqual(
            self.check(scopes=["."]),
            ["lib/bar/AGENTS.md:1: identifier `bar_missing_fn` not found"],
        )
        self.assertEqual(
            self.check(scopes=["lib/foo", "lib/typo"]),
            ["lib/typo: no tracked AGENTS.md under this directory"],
        )


if __name__ == "__main__":
    unittest.main()
