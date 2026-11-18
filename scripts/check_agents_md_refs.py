#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that file paths and identifiers named in AGENTS.md files still exist.

Prose that names files and symbols drifts as code moves. For each tracked
AGENTS.md under the given directories (every tracked AGENTS.md if none are
given), this reads the inline code spans outside fenced code blocks and checks
two kinds of span:

  - Paths: a span containing "/" or ending in a source-file extension must
    name a tracked file or directory relative to the AGENTS.md's directory or
    any directory above it, or to the `src/` of its crate (the nearest
    directory above it with a Cargo.toml). A bare file name such as `args.rs`
    must name a tracked file somewhere in the repository.
  - Identifiers: snake_case words containing "_", CamelCase words with at
    least two capitals, lowerCamelCase words, and `macro!` names, including
    those inside larger spans such as `Type::method()` or a signature. Each
    must occur as a whole word in a tracked file that is not Markdown. All
    identifiers are found with a single `git grep`.

Skipped spans: commands (`cargo test ...`), flags, ALL_CAPS words
(environment variables and constants), config values (anything with "="),
URLs, globs, `<placeholders>`, and domain-qualified paths such as
`sigs.k8s.io/...`. Single lowercase words and single-hump capitalised words
are too generic to check.

To exempt a span that is deliberately not a reference to this repository,
such as an upstream path or an output file an agent writes, put the marker
comment "<!-- agents-md-refs: ignore -->" on the same line. It exempts every
span on that line.

The checker reads the Git index, so staged additions, removals and renames
are checked as they would be committed. The pre-commit hook runs on every
commit, not only on AGENTS.md edits, because a code change that removes or
renames a named file or symbol must fail it too.

Usage: check_agents_md_refs.py [--root REPO_ROOT] [DIR ...]
"""

import argparse
import posixpath
import re
import subprocess
import sys
from pathlib import Path

AGENTS_FILE = "AGENTS.md"
IGNORE_MARKER = "<!-- agents-md-refs: ignore -->"

# Markdown is excluded from the identifier search so that a name mentioned
# only in documentation, including the AGENTS.md itself, does not count.
IDENTIFIER_SEARCH_PATHSPEC = (":(exclude)*.md", ":(exclude)*.mdx")

PATH_EXTENSIONS = (
    ".rs",
    ".md",
    ".mdx",
    ".toml",
    ".py",
    ".pyi",
    ".go",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".proto",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cc",
    ".cpp",
)
COMMANDS = frozenset(
    {
        "bash",
        "cargo",
        "cd",
        "curl",
        "docker",
        "export",
        "gh",
        "git",
        "go",
        "helm",
        "just",
        "kubectl",
        "make",
        "maturin",
        "npm",
        "npx",
        "perf",
        "pip",
        "pre-commit",
        "pytest",
        "python",
        "python3",
        "ruff",
        "rustup",
        "sh",
        "sudo",
        "uv",
        "uvx",
    }
)

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# `path.rs:120`, `path.rs:120-140`, `page.md#anchor`
LOCATION_SUFFIX = re.compile(r"(?::\d+(?:-\d+)?|#[\w-]+)$")
NON_PATH = re.compile(r"[\s*?{}<>\[\]$|,;=()'\"`]|://|\.\.\.")
DOMAIN = re.compile(r"^[\w-]+(?:\.[\w-]+)+$")
# `<YourType>` stands for a name; `Vec<T>` and `::<T>` are real generics.
PLACEHOLDER = re.compile(r"(?<![\w:])<[^<>]*>")
ALL_CAPS = re.compile(r"^[A-Z0-9_]+$")
SNAKE_CASE = re.compile(r"^_*[a-z][a-z0-9]*(?:_+[a-z0-9]+)+_*$")
UPPER_CAMEL_CASE = re.compile(r"^_*[A-Z](?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])\w*$")
LOWER_CAMEL_CASE = re.compile(r"^_*[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*$")
MACRO = re.compile(r"^(?:\w+::)*(\w+)!$")


def git(root, *args, input=None, ok_codes=(0,)):
    """Run Git in root and return stdout as bytes."""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in ok_codes:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.stdout


def tracked_paths(root):
    """Return the set of stage-0 paths in the Git index."""
    output = git(root, "ls-files", "--stage", "-z")
    paths = set()
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        if metadata.split()[2] != b"0":
            continue
        paths.add(raw_path.decode("utf-8", errors="surrogateescape"))
    return paths


def read_index_blobs(root, paths):
    """Return {path: text} for paths, read from the index in one process."""
    request = "".join(f":{path}\n" for path in paths).encode()
    output = git(root, "cat-file", "--batch", input=request)
    blobs = {}
    offset = 0
    for path in paths:
        header_end = output.index(b"\n", offset)
        size = int(output[offset:header_end].split()[2])
        start = header_end + 1
        blobs[path] = output[start : start + size].decode("utf-8", errors="replace")
        offset = start + size + 1
    return blobs


def code_spans(text):
    """Yield (line_number, span) for inline code spans outside fences."""
    fence = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * len(marker)
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is not None or IGNORE_MARKER in line:
            continue
        for span in CODE_SPAN.finditer(line):
            yield line_number, span.group(1).strip()


def classify(span):
    """Return ("path", path), ("identifiers", [word, ...]) or None."""
    if not span or "*" in span or "://" in span:
        return None
    first = span.split()[0]
    if first in COMMANDS or first.startswith(("-", "$", "~")):
        return None

    path = LOCATION_SUFFIX.sub("", span.split("::", 1)[0])
    if "/" in path:
        segments = path.split("/")
        if (
            NON_PATH.search(path)
            or path.startswith("/")
            or (DOMAIN.match(segments[0]) and segments[0] not in (".", ".."))
        ):
            return None
        return "path", path
    if not NON_PATH.search(path) and any(
        path.endswith(ext) and len(path) > len(ext) for ext in PATH_EXTENSIONS
    ):
        return "path", path

    if "=" in span or span.startswith(("'", '"')):
        return None
    macro = MACRO.match(span)
    if macro:
        return "identifiers", [macro.group(1)]
    words = [
        word
        for word in WORD.findall(PLACEHOLDER.sub(" ", span))
        if not ALL_CAPS.match(word)
        and (
            SNAKE_CASE.match(word)
            or UPPER_CAMEL_CASE.match(word)
            or LOWER_CAMEL_CASE.match(word)
        )
    ]
    return ("identifiers", words) if words else None


class Checker:
    def __init__(self, root):
        self.root = root
        self.paths = tracked_paths(root)
        self.dirs = set()
        self.basenames = set()
        for path in self.paths:
            self.basenames.add(posixpath.basename(path))
            parent = posixpath.dirname(path)
            while parent and parent not in self.dirs:
                self.dirs.add(parent)
                parent = posixpath.dirname(parent)
        self.crate_roots = {
            posixpath.dirname(p)
            for p in self.paths
            if posixpath.basename(p) == "Cargo.toml"
        }

    def bases(self, directory):
        """Directories a relative path in directory/AGENTS.md may start from."""
        bases = []
        crate_src = None
        while True:
            bases.append(directory)
            if crate_src is None and directory in self.crate_roots:
                crate_src = posixpath.join(directory, "src")
            if not directory:
                break
            directory = posixpath.dirname(directory)
        if crate_src is not None:
            bases.append(crate_src)
        return bases

    def path_exists(self, agents_dir, path):
        path = path.rstrip("/")
        if "/" not in path and path in self.basenames:
            return True
        for base in self.bases(agents_dir):
            candidate = posixpath.normpath(posixpath.join(base, path))
            if candidate in self.paths or candidate in self.dirs:
                return True
        return False

    def missing_identifiers(self, words):
        """Return the subset of words that occur nowhere outside Markdown."""
        if not words:
            return set()
        # One alternation, not one -F pattern per word: git grep tries every
        # pattern on every line, so 500 -F patterns take minutes. Longest
        # first so that a leftmost-first engine also prefers the whole word.
        alternation = "|".join(sorted(words, key=lambda w: (-len(w), w)))
        output = git(
            self.root,
            "grep",
            "--cached",
            "--no-color",
            "--no-line-number",
            "--no-column",
            "-I",
            "-h",
            "-o",
            "-w",
            "-E",
            "-f",
            "-",
            "--",
            *IDENTIFIER_SEARCH_PATHSPEC,
            input=f"({alternation})\n".encode(),
            ok_codes=(0, 1),
        )
        found = set(output.decode("utf-8", errors="replace").splitlines())
        return set(words) - found

    def check(self, scopes=()):
        """Return (files checked, spans checked, error lines)."""
        errors = []
        all_agents = sorted(
            p for p in self.paths if posixpath.basename(p) == AGENTS_FILE
        )
        if scopes:
            prefixes = [f"{scope.strip('/')}/" for scope in scopes]
            for scope, prefix in zip(scopes, prefixes):
                if not any(p.startswith(prefix) for p in all_agents):
                    errors.append(f"{scope}: no tracked {AGENTS_FILE} under this scope")
            agents_files = [p for p in all_agents if p.startswith(tuple(prefixes))]
        else:
            agents_files = all_agents

        blobs = read_index_blobs(self.root, agents_files)
        identifier_uses = []
        checked = 0
        for agents_path in agents_files:
            agents_dir = posixpath.dirname(agents_path)
            for line_number, span in code_spans(blobs[agents_path]):
                kind = classify(span)
                if kind is None:
                    continue
                checked += 1
                where = f"{agents_path}:{line_number}"
                if kind[0] == "path":
                    if not self.path_exists(agents_dir, kind[1]):
                        errors.append(f"{where}: path `{span}` does not exist")
                    continue
                identifier_uses.extend((where, span, word) for word in kind[1])

        missing = self.missing_identifiers({word for _, _, word in identifier_uses})
        for where, span, word in identifier_uses:
            if word in missing:
                context = "" if word == span else f" in `{span}`"
                errors.append(f"{where}: identifier `{word}`{context} not found")
        return len(agents_files), checked, sorted(set(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the checkout containing this script)",
    )
    parser.add_argument(
        "scopes",
        nargs="*",
        metavar="DIR",
        help="check only AGENTS.md files under these repo-relative directories",
    )
    args = parser.parse_args(argv)

    try:
        files, checked, errors = Checker(args.root).check(args.scopes)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        print(f"error: Git command failed: {message}", file=sys.stderr)
        return 1

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

    print(f"checked {checked} code spans in {files} AGENTS.md files: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
