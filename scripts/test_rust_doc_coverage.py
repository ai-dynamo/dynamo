# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and real-index regression tests for the Rust documentation hook."""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from check_rust_doc_coverage import changed_functions, functions

CHECKER = pathlib.Path(__file__).with_name("check_rust_doc_coverage.py").resolve()


class SyntaxTests(unittest.TestCase):
    """Exercise Rust syntax rather than matching function-looking text."""

    def test_doc_forms_and_attributes(self):
        """Doc comments remain attached across attributes and ordinary comments."""
        source = b"""/// Line docs.
#[test]
// A normal comment between docs and function.
async fn a() {}
/** Block docs. */
pub(crate) unsafe fn b() {}
#[doc = "Attribute docs."]
fn c() {}
fn d() { //! Inner docs.
}
"""
        self.assertTrue(all(f.documented for f in functions(source)))
        self.assertEqual(len(functions(source)), 4)

    def test_ordinary_module_and_empty_comments_do_not_count(self):
        source = b"""//! Module documentation.
fn a() {}
// Ordinary comment.
fn b() {}
///
fn c() {}
#[doc = ""]
fn d() {}
"""
        self.assertFalse(any(f.documented for f in functions(source)))

    def test_methods_trait_declarations_and_nested_functions(self):
        """Private and trait methods count, including declarations without bodies."""
        source = b"""trait T { fn a(); }
impl T for S { fn a() {} }
fn outer() { fn nested() {} }
"""
        self.assertEqual(
            [f.name for f in functions(source)], ["a", "a", "outer", "nested"]
        )

    def test_strings_and_macros_are_not_functions(self):
        source = b'const S: &str = "fn fake() {}"; macro_rules! m { () => { fn generated() {} }; }'
        self.assertEqual(functions(source), [])

    def test_only_body_edits_count_not_line_shifts_or_deletions(self):
        """Compare function contents rather than unstable line numbers."""
        before = b"fn removed() {}\nfn edited() { work(); }\nfn same() {}"
        after = b"\n\nfn edited() {}\nfn same() {}"
        self.assertEqual([f.name for f in changed_functions(before, after)], ["edited"])

    def test_removing_doc_comment_is_an_edit(self):
        changed = changed_functions(b"/// Docs.\nfn f() {}", b"fn f() {}")
        self.assertEqual(len(changed), 1)
        self.assertFalse(changed[0].documented)

    def test_invalid_rust_fails(self):
        with self.assertRaises(ValueError):
            functions(b"fn broken(")


class IndexTests(unittest.TestCase):
    """Run the actual CLI against temporary Git indexes without making commits."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        # Commit hooks export Git paths pointing at the caller's repository.
        # Never let temporary-repository tests inherit those paths.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.git("init", "-q")
        self.path = self.root / "sample.rs"
        self.path.write_text("fn unchanged() {}\n")
        self.git("add", ".")
        self.base = self.git("write-tree").strip()

    def git(self, *args):
        """Run Git only inside the temporary test repository."""
        return subprocess.check_output(
            ["git", *args], cwd=self.root, text=True, env=self.env
        )

    def run_check(self, *args):
        """Capture checker output and exit status without shell interpolation."""
        return subprocess.run(
            [sys.executable, str(CHECKER), *args],
            cwd=self.root,
            env=self.env,
            text=True,
            errors="surrogateescape",
            capture_output=True,
            check=False,
        )

    def commit(self, message):
        """Create a commit without invoking the caller's hooks."""
        self.git("commit", "-q", "--no-verify", "-m", message)

    def start_conflicted_merge(self):
        """Create an index where an imported undocumented function is staged."""
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test User")
        self.git("config", "commit.gpgsign", "false")
        self.commit("initial")
        self.git("checkout", "-q", "-b", "merge-feature")
        self.path.write_text("fn feature_side() {}\n")
        self.git("add", ".")
        self.commit("feature")
        self.git("checkout", "-q", "-b", "merge-source", "HEAD~1")
        self.path.write_text("fn main_side() {}\n")
        self.git("add", ".")
        self.commit("main")
        (self.root / "imported.rs").write_text("fn imported() {}\n")
        self.git("add", ".")
        self.commit("imported")
        self.git("checkout", "-q", "merge-feature")
        with self.assertRaises(subprocess.CalledProcessError):
            self.git("merge", "merge-source")
        self.path.write_text("fn feature_side() {}\n")
        self.git("add", "sample.rs")

    def test_threshold_boundary(self):
        """Four documented functions out of five pass; three fail."""
        for documented, status in [(4, 0), (3, 1)]:
            self.path.write_text(
                "".join(
                    ("/// Docs.\n" if i < documented else "") + f"fn f{i}() {{}}\n"
                    for i in range(5)
                )
            )
            self.git("add", ".")
            result = self.run_check("--base", self.base)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertIn(f"{documented}/5", result.stdout)
            self.assertIn("missing doc comment on f4", result.stdout)

    def test_unstaged_docs_cannot_hide_staged_failure(self):
        self.path.write_text("fn added() {}\n")
        self.git("add", ".")
        self.path.write_text("/// Unstaged docs.\nfn added() {}\n")
        self.assertEqual(self.run_check("--base", self.base).returncode, 1)

    def test_unstaged_removal_cannot_break_staged_success(self):
        self.path.write_text("/// Staged docs.\nfn added() {}\n")
        self.git("add", ".")
        self.path.write_text("fn added() {}\n")
        self.assertEqual(self.run_check("--base", self.base).returncode, 0)

    def test_rename_only_does_not_count_unchanged_functions(self):
        self.git("mv", "sample.rs", "file with spaces.rs")
        result = self.run_check("--base", self.base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no added or edited", result.stdout)

    def test_deleted_file_is_ignored(self):
        self.git("rm", "-f", "sample.rs")
        self.assertEqual(self.run_check("--base", self.base).returncode, 0)

    def test_first_commit_and_explicit_full_audit(self):
        self.assertEqual(self.run_check().returncode, 1)
        self.assertEqual(self.run_check("--all-files").returncode, 1)

    def test_default_check_skips_active_merge(self):
        self.start_conflicted_merge()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("merge in progress", result.stdout)

    def test_explicit_audits_still_run_during_merge(self):
        self.start_conflicted_merge()
        for args in [("--base", self.base), ("--all-files",)]:
            result = self.run_check(*args)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("missing doc comment on", result.stdout)

    def test_normal_index_is_checked_without_merge(self):
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test User")
        self.git("config", "commit.gpgsign", "false")
        self.commit("initial")
        self.path.write_text("fn added() {}\n")
        self.git("add", ".")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("missing doc comment on added", result.stdout)

    def test_non_utf8_path_is_checked(self):
        (self.root / "blob").write_text("fn added() {}\n")
        blob = self.git("hash-object", "-w", "blob").strip()
        path = os.fsdecode(b"nonutf8-\xff.rs")
        self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        result = self.run_check("--base", self.base)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("missing doc comment on added", result.stdout)

    def test_unparsed_committed_copy_does_not_block(self):
        self.path.write_text("fn f(E { a, #[cfg(test)] b }: E) {}\n")
        self.git("add", ".")
        base = self.git("write-tree").strip()
        self.path.write_text("fn f(E { a, #[cfg(test)] b }: E) { work(); }\n")
        self.git("add", ".")
        result = self.run_check("--base", base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipped sample.rs", result.stderr)

    def test_parse_error_is_actionable(self):
        """Report the path instead of treating invalid Rust as an empty file."""
        self.path.write_text("fn invalid(")
        self.git("add", ".")
        result = self.run_check("--base", self.base)
        self.assertEqual(result.returncode, 2)
        self.assertIn("sample.rs", result.stderr)


if __name__ == "__main__":
    unittest.main()
