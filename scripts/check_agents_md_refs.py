#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that file paths and identifiers named in AGENTS.md files still exist.

Prose that names files and symbols drifts as code moves. For each tracked
AGENTS.md under the given directories (the whole repository if none are
given), every inline code span outside a fenced code block is checked:

  - A span that is one path, meaning it contains "/" or ends in a source-file
    extension, must resolve from the AGENTS.md's directory or be the tail of a
    tracked file or directory path.
  - In any other span, each snake_case, CamelCase or SCREAMING_SNAKE word must
    occur as a whole word in a tracked file that is not Markdown. Single plain
    words and bare acronyms are too generic to check.

A span containing "*" is skipped. To exempt a name that is deliberately not in
this repository, such as an upstream path or a file an agent writes, put
"<!-- agents-md-refs: ignore -->" on the same line. It exempts every span on
that line.

Everything is read from the Git index, so the check sees what would be
committed.

Usage: check_agents_md_refs.py [DIR ...]
"""

import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

IGNORE_MARKER = "<!-- agents-md-refs: ignore -->"

FENCE = re.compile(r"\s{0,3}(`{3,}|~{3,})(.*)")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
PATH = re.compile(
    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]*)+"
    r"|[A-Za-z0-9_.-]+\.(?:rs|md|mdx|toml|py|go|sh|yaml|yml|json|proto)"
)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# snake_case and SCREAMING_SNAKE, lowerCamel and UpperCamel, KVIndexer.
COMPOUND = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]|[a-z0-9][A-Z]|[A-Z]{2}[a-z]")

GREP = "grep --cached --no-color --no-line-number --no-column -I -h -o -w -E".split()
# A name mentioned only in documentation, including the AGENTS.md itself, does
# not count as existing.
NOT_MARKDOWN = (":(exclude)*.md", ":(exclude)*.mdx")


def git(root, *args, ok=(0,)):
    """Run Git in root and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode not in ok:
        sys.exit(f"error: git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def code_spans(text):
    """Yield (line_number, span) for inline code spans outside fenced blocks."""
    fence = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if fence:
            # Only the same character, at least as long, with nothing after it.
            if match and match[1].startswith(fence) and not match[2].strip():
                fence = None
        elif match:
            fence = match[1]
        elif IGNORE_MARKER not in line:
            for span in CODE_SPAN.finditer(line):
                yield line_number, span[2].strip()


def classify(span):
    """Return (path, identifiers) named by a span. Either may be empty."""
    if "*" in span:
        return None, []
    if PATH.fullmatch(span):
        return span, []
    return None, [word for word in WORD.findall(span) if COMPOUND.search(word)]


def path_tails(files):
    """Return every tail of every tracked file and directory path.

    `a/b/c.rs` gives `a/b/c.rs`, `b/c.rs`, `c.rs`, `a/b`, `b` and `a`.
    """
    tails = set()
    for path in files:
        while path:
            tail = path
            while tail and tail not in tails:
                tails.add(tail)
                tail = tail.partition("/")[2]
            path = posixpath.dirname(path)
    return tails


def missing_identifiers(root, words):
    """Return the words that occur in no tracked non-Markdown file."""
    if not words:
        return set()
    # One alternation, not one pattern per word: git grep tries every pattern
    # on every line, so 500 patterns take minutes. Longest first so that the
    # whole word wins whichever alternative the regex engine prefers.
    pattern = "|".join(sorted(words, key=len, reverse=True))
    found = git(root, *GREP, "-e", f"({pattern})", "--", *NOT_MARKDOWN, ok=(0, 1))
    return words - set(found.split())


def check(root, scopes):
    """Return (AGENTS.md files checked, spans checked, error lines)."""
    files = set(git(root, "ls-files", "-z").split("\0")) - {""}
    tails = path_tails(files)
    agents = sorted(p for p in files if posixpath.basename(p) == "AGENTS.md")
    by_scope = [
        [p for p in agents if PurePosixPath(p).is_relative_to(scope)]
        for scope in scopes
    ]
    errors = [
        f"{scope}: no tracked AGENTS.md under this directory"
        for scope, found in zip(scopes, by_scope)
        if not found
    ]

    agents = sorted(set().union(*by_scope))

    refs = []
    for agents_path in agents:
        for line_number, span in code_spans(git(root, "show", f":{agents_path}")):
            path, words = classify(span)
            if path or words:
                refs.append((agents_path, line_number, span, path, words))

    missing = missing_identifiers(root, {word for *_, words in refs for word in words})
    for agents_path, line_number, span, path, words in refs:
        where = f"{agents_path}:{line_number}"
        # A path is relative to the AGENTS.md, or the tail of any tracked path.
        if path and not (
            posixpath.normpath(posixpath.join(posixpath.dirname(agents_path), path))
            in tails
            or posixpath.normpath(path) in tails
        ):
            errors.append(f"{where}: path `{span}` does not exist")
        for word in words:
            if word in missing:
                context = "" if word == span else f" in `{span}`"
                errors.append(f"{where}: identifier `{word}`{context} not found")
    return len(agents), len(refs), list(dict.fromkeys(errors))


def main():
    root = Path(__file__).resolve().parents[1]
    files, spans, errors = check(root, sys.argv[1:] or ["."])
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(
            f"\n{len(errors)} stale AGENTS.md reference(s). Update the AGENTS.md "
            f"to the current name, or add {IGNORE_MARKER} to a line whose spans "
            f"do not refer to this repository.",
            file=sys.stderr,
        )
        return 1
    print(f"checked {spans} code spans in {files} AGENTS.md files: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
