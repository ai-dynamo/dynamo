#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove that a router crate move is a move.

Collects every line of the --from paths at <base-rev> and every line of the
--to paths at <head-rev> (or the working tree), normalizes both sides, and
compares them as multisets. A pure move reports zero new and zero dropped
lines; anything else is listed, one line per entry, so the reviewer reads the
semantic diff instead of a relocated file.

Normalization, per line:
  * leading and trailing whitespace is stripped
  * blank lines are dropped
  * `//`, `///`, `//!` comment lines are dropped
  * `use ...;` / `pub use ...;` / `pub(...) use ...;` are dropped, including
    multi-line imports up to the closing `;`
  * `mod x;` / `pub mod x;` / `pub(...) mod x;` are dropped
  * attribute-only lines `#[cfg(...)]`, `#![cfg(...)]`, `#[cfg_attr(...)]`,
    `#[doc(hidden)]` are dropped, including multi-line attributes

Directories expand to the `.rs` files under them (--all-files lifts the
filter). Paths are relative to the repository root; a path that exists
relative to the current directory is translated.

Usage:
  python3 -I scripts/router_move_diff.py <base-rev> [<head-rev>|WORKTREE] \\
      --from <path>... --to <path>...

Exit status: 0 the move is exact, 1 text differs, 2 usage or git error.

Output (stable, greppable):
  router_move_diff base=<rev> head=<rev|WORKTREE>
  from <path> raw=<n> normalized=<n> found=<n> fraction=<0.000>
  to <path> raw=<n> normalized=<n>
  summary from_normalized=<n> to_normalized=<n> common=<n> new=<n> dropped=<n>
  new <count>x <line>
  dropped <count>x <line>
  result MOVE | result DIFFERS
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter

WORKTREE = "WORKTREE"

USE_RE = re.compile(r"^(pub(\([^)]*\))?\s+)?use\b")
MOD_RE = re.compile(r"^(pub(\([^)]*\))?\s+)?mod\s+\w+\s*;$")
ATTR_RE = re.compile(r"^#!?\[(cfg\b|cfg_attr\b|doc\(hidden\))")
COMMENT_RE = re.compile(r"^//")


def fail(code, message):
    print(f"router_move_diff: error: {message}", file=sys.stderr)
    sys.exit(code)


def git(root, *args):
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        fail(
            2, f"git {' '.join(args)}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def repo_root():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        fail(2, f"not inside a git repository: {exc}")
    return out.decode("utf-8").strip()


def root_relative(root, path):
    path = path.rstrip("/") or "."
    if os.path.exists(path):
        rel = os.path.relpath(os.path.abspath(path), root)
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return path.replace(os.sep, "/")


def normalize(lines):
    """Return the semantic lines of a Rust source, per the module docstring."""
    out = []
    in_use = False
    attr_depth = 0
    for raw in lines:
        line = raw.strip()
        if in_use:
            if line.endswith(";"):
                in_use = False
            continue
        if attr_depth > 0:
            attr_depth += line.count("[") - line.count("]")
            if attr_depth < 0:
                attr_depth = 0
            continue
        if not line or COMMENT_RE.match(line):
            continue
        if USE_RE.match(line):
            if not line.endswith(";"):
                in_use = True
            continue
        if MOD_RE.match(line):
            continue
        if ATTR_RE.match(line):
            depth = line.count("[") - line.count("]")
            if depth > 0:
                attr_depth = depth
                continue
            if line.endswith("]"):
                continue
        out.append(line)
    return out


def files_at_rev(root, rev, path, all_files):
    out = git(root, "ls-tree", "-r", "--name-only", "-z", rev, "--", path)
    names = [n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n]
    if not names:
        fail(2, f"{path}: not found at {rev}")
    if names == [path]:
        return names
    return [n for n in names if all_files or n.endswith(".rs")]


def files_in_worktree(root, path, all_files):
    full = os.path.join(root, path)
    if os.path.isfile(full):
        return [path]
    if not os.path.isdir(full):
        fail(2, f"{path}: not found in the worktree")
    names = []
    for dirpath, dirnames, filenames in os.walk(full):
        dirnames.sort()
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(
                os.sep, "/"
            )
            if all_files or rel.endswith(".rs"):
                names.append(rel)
    return names


def read_at_rev(root, rev, path):
    return git(root, "show", f"{rev}:{path}").decode("utf-8", "replace").splitlines()


def read_in_worktree(root, path):
    with open(os.path.join(root, path), encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()


def collect(root, rev, paths, all_files):
    """Return [(file, raw_count, normalized_lines)] for paths at rev or WORKTREE."""
    result = []
    for path in paths:
        if rev == WORKTREE:
            files = files_in_worktree(root, path, all_files)
        else:
            files = files_at_rev(root, rev, path, all_files)
        if not files:
            fail(2, f"{path}: no .rs files (use --all-files to include other files)")
        for name in files:
            raw = (
                read_in_worktree(root, name)
                if rev == WORKTREE
                else read_at_rev(root, rev, name)
            )
            result.append((name, len(raw), normalize(raw)))
    return result


def emit(prefix, counter, limit):
    shown = 0
    for line in sorted(counter):
        if limit and shown >= limit:
            print(
                f"{prefix} ... {len(counter) - shown} more distinct lines (raise --limit)"
            )
            break
        print(f"{prefix} {counter[line]}x {line}")
        shown += 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="router_move_diff.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("base", help="revision holding the --from paths")
    parser.add_argument(
        "head",
        nargs="?",
        default=WORKTREE,
        help=f"revision holding the --to paths, or {WORKTREE} (default) for the working tree",
    )
    parser.add_argument(
        "--from",
        dest="from_paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="files or directories at <base>",
    )
    parser.add_argument(
        "--to",
        dest="to_paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="files or directories at <head>",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="expand directories to every file, not only *.rs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="print at most N distinct new and N distinct dropped lines (0 = all)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    from_paths = [root_relative(root, p) for p in args.from_paths]
    to_paths = [root_relative(root, p) for p in args.to_paths]

    from_files = collect(root, args.base, from_paths, args.all_files)
    to_files = collect(root, args.head, to_paths, args.all_files)

    from_counter = Counter()
    for _, _, lines in from_files:
        from_counter.update(lines)
    to_counter = Counter()
    for _, _, lines in to_files:
        to_counter.update(lines)

    print(f"router_move_diff base={args.base} head={args.head}")

    # Per --from file: how many of its lines are present in --to, consuming
    # multiplicity in file order so the per-file sums equal `common`.
    remaining = Counter(to_counter)
    for name, raw_count, lines in from_files:
        found = 0
        for line in lines:
            if remaining[line] > 0:
                remaining[line] -= 1
                found += 1
        fraction = found / len(lines) if lines else 1.0
        print(
            f"from {name} raw={raw_count} normalized={len(lines)} found={found} fraction={fraction:.3f}"
        )
    for name, raw_count, lines in to_files:
        print(f"to {name} raw={raw_count} normalized={len(lines)}")

    new = to_counter - from_counter
    dropped = from_counter - to_counter
    common = sum((from_counter & to_counter).values())
    print(
        f"summary from_normalized={sum(from_counter.values())} "
        f"to_normalized={sum(to_counter.values())} common={common} "
        f"new={sum(new.values())} dropped={sum(dropped.values())}"
    )
    emit("new", new, args.limit)
    emit("dropped", dropped, args.limit)

    if new or dropped:
        print("result DIFFERS")
        return 1
    print("result MOVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
