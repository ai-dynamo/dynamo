# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require doc comments on 80% of added or edited Rust functions in the index.

Reads staged blobs, not working files. Compare against HEAD by default, or use
--base REF to cover a whole branch. Unchanged and deleted functions do not count.
Includes private functions, tests, impl methods, and trait declarations. Recognizes
Rust doc comments and #[doc = ...] attributes; ordinary comments do not count.
Macro-generated functions are not expanded. This checks presence, not prose quality,
and does not reproduce CodeRabbit's proprietary coverage calculation.

Run through pre-commit's rust-doc-coverage hook for isolated parser dependencies.
--all-files explicitly audits all tracked Rust files in the index.
"""

import argparse
import collections
import dataclasses
import subprocess
import sys

import tree_sitter_rust
from tree_sitter import Language, Parser


@dataclasses.dataclass(frozen=True)
class Function:
    """A function's location, documentation flag, and source used for comparison."""

    name: str
    line: int
    documented: bool
    source: bytes


def is_doc(node):
    """Recognize nonempty Rust doc comments and explicit doc attributes."""
    comment = node.child_by_field_name("doc")
    if comment is not None:
        return bool(comment.text.strip(b" \t\r\n*"))
    if node.type in {"attribute_item", "inner_attribute_item"}:
        attribute = next(c for c in node.named_children if c.type == "attribute")
        value = attribute.child_by_field_name("value")
        return (
            attribute.named_children[0].text == b"doc"
            and value is not None
            and value.text.strip(b'" \t\r\n') != b""
        )
    return False


def functions(source):
    """Parse function declarations with their attached attributes and comments."""
    root = Parser(Language(tree_sitter_rust.language())).parse(source).root_node
    if root.has_error:
        raise ValueError("Rust syntax could not be parsed; coverage was not calculated")
    found = []
    pending = [root]
    while pending:
        node = pending.pop()
        pending.extend(reversed(node.named_children))
        if node.type not in {"function_item", "function_signature_item"}:
            continue
        start = node.start_byte
        documented = False
        previous = node.prev_named_sibling
        while previous and previous.type in {
            "attribute_item",
            "line_comment",
            "block_comment",
        }:
            # Inner comments describe the containing module, not this function.
            if previous.child_by_field_name("inner") is not None:
                break
            start = previous.start_byte
            documented |= is_doc(previous)
            previous = previous.prev_named_sibling
        body = node.child_by_field_name("body")
        if body:
            for child in body.named_children:
                if child.type not in {
                    "inner_attribute_item",
                    "line_comment",
                    "block_comment",
                }:
                    break
                if child.type == "inner_attribute_item" or child.child_by_field_name(
                    "inner"
                ):
                    documented |= is_doc(child)
        found.append(
            Function(
                node.child_by_field_name("name").text.decode(),
                node.start_point.row + 1,
                documented,
                source[start : node.end_byte],
            )
        )
    return found


def changed_functions(before, after):
    """Return added or edited functions, ignoring line shifts and deletions."""
    old = collections.Counter(f.source for f in functions(before))
    changed = []
    for function in functions(after):
        if old[function.source]:
            old[function.source] -= 1
        else:
            changed.append(function)
    return changed


def git(*args):
    """Read Git output without shell interpolation."""
    return subprocess.check_output(["git", *args])


def staged_files(base, all_files=False):
    """Yield current and previous Rust paths, including Git-detected renames."""
    if all_files:
        for path in git("ls-files", "-z", "--", "*.rs").split(b"\0"):
            if path:
                yield path.decode(), None
        return
    args = ["diff", "--cached", "--name-status", "-z", "--find-renames"]
    if base:
        args.append(base)
    entries = iter(git(*args, "--", "*.rs").split(b"\0")[:-1])
    for status in entries:
        path = next(entries).decode()
        if status.startswith(b"R"):
            yield next(entries).decode(), path
        elif status == b"A":
            yield path, None
        elif status == b"M":
            yield path, path
        elif status != b"D":
            raise ValueError(f"unsupported index status {status!r} for {path}")


def check(base=None, all_files=False):
    """Print changed-function coverage and return a failing status below 80%."""
    checked = []
    for path, old_path in staged_files(base, all_files):
        before = git("show", f"{base or 'HEAD'}:{old_path}") if old_path else b""
        after = git("show", f":{path}")
        try:
            checked.extend((path, f) for f in changed_functions(before, after))
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
    if not checked:
        print("Rust doc coverage: no added or edited functions in the index.")
        return 0
    documented = sum(f.documented for _, f in checked)
    total = len(checked)
    print(
        f"Rust doc coverage: {documented}/{total} ({100 * documented / total:.2f}%), required 80%"
    )
    for path, function in checked:
        if not function.documented:
            print(f"{path}:{function.line}: missing doc comment on {function.name}")
    return int(documented * 100 < total * 80)


def main():
    """Parse the comparison scope and report configuration or syntax errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--base", help="compare the index against this Git revision")
    scope.add_argument(
        "--all-files",
        action="store_true",
        help="audit all tracked Rust files in the index",
    )
    args = parser.parse_args()
    try:
        return check(args.base, args.all_files)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"Rust doc coverage: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
