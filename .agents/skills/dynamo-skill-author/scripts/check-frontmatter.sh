#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Strict frontmatter validator for a Dynamo skill's SKILL.md.
#
# Checks:
#   - YAML frontmatter parses.
#   - Required fields present: name, description, version, author, tags, tools.
#   - `name` matches the parent directory name and starts with `dynamo-`.
#   - `description` is 50-500 chars, contains no angle-bracket placeholders.
#   - `version` matches MAJOR.MINOR.PATCH.
#   - `author` is `NVIDIA`.
#   - `tags` includes `dynamo` and is 3-6 entries.
#   - `tools` is a non-empty list.
#
# Implemented as a shell wrapper around an inline python3 program. This
# keeps the validator out of the repository's Python formatter scope while
# still using PyYAML for the actual parse.
#
# Exits 0 on all-pass, non-zero on any failure. Output uses PASS / FAIL
# lines on stdout for the agent to parse.
#
# Usage:
#   bash check-frontmatter.sh <path-to-SKILL.md>

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-SKILL.md>" >&2
    exit 2
fi

exec python3 - "$1" <<'PYTHON'
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FAIL|deps|pyyaml not installed. Install via: pip install pyyaml", flush=True)
    sys.exit(2)

REQUIRED_FIELDS = {"name", "description", "version", "author", "tags", "tools"}
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
XML_TAG_RE = re.compile(r"<[a-zA-Z][^>]*>")


def split_frontmatter(text):
    parts = text.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("file does not start with --- frontmatter delimiters")
    return parts[1]


def check(path):
    failures = 0
    passes = 0

    def pf(ok, name, detail):
        nonlocal failures, passes
        if ok:
            passes += 1
            print("PASS|" + name + "|" + detail, flush=True)
        else:
            failures += 1
            print("FAIL|" + name + "|" + detail, flush=True)

    if not path.is_file():
        print("FAIL|exists|" + str(path) + " not found", flush=True)
        return 1

    raw = path.read_text(encoding="utf-8")

    try:
        fm_text = split_frontmatter(raw)
    except ValueError as e:
        print("FAIL|frontmatter|" + str(e), flush=True)
        return 1

    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        print("FAIL|yaml|parse error: " + str(e), flush=True)
        return 1

    if not isinstance(fm, dict):
        print("FAIL|yaml|frontmatter is not a mapping", flush=True)
        return 1

    missing = REQUIRED_FIELDS - set(fm.keys())
    pf(not missing, "required-fields", "missing: " + (str(sorted(missing)) if missing else "none"))
    if missing:
        return 1

    name = fm["name"]
    pf(isinstance(name, str) and name.startswith("dynamo-"),
       "name-prefix", "name='" + str(name) + "' (must start with 'dynamo-')")

    parent = path.parent.name
    pf(name == parent,
       "name-matches-dir", "name='" + str(name) + "' vs directory='" + parent + "'")

    desc = fm["description"]
    pf(isinstance(desc, str), "description-type", "type=" + type(desc).__name__)
    if isinstance(desc, str):
        length = len(desc)
        pf(50 <= length <= 500,
           "description-length", str(length) + " chars (target 50-500)")
        xml_hits = XML_TAG_RE.findall(desc)
        pf(not xml_hits,
           "description-xml",
           "angle-bracket placeholders detected: " + (str(xml_hits[:3]) if xml_hits else "none"))

    version = fm["version"]
    pf(isinstance(version, str) and bool(VERSION_RE.match(version)),
       "version-format", "version='" + str(version) + "' (must match MAJOR.MINOR.PATCH)")

    author = fm["author"]
    pf(author == "NVIDIA",
       "author", "author='" + str(author) + "' (must be 'NVIDIA')")

    tags = fm["tags"]
    pf(isinstance(tags, list) and "dynamo" in tags,
       "tags-dynamo", "tags=" + str(tags))
    if isinstance(tags, list):
        pf(3 <= len(tags) <= 6,
           "tags-count", str(len(tags)) + " tags (target 3-6)")

    tools = fm["tools"]
    pf(isinstance(tools, list) and len(tools) > 0,
       "tools-nonempty", "tools=" + str(tools))

    print("", flush=True)
    print("Frontmatter check: " + str(passes) + " passed, " + str(failures) + " failed", flush=True)
    return 0 if failures == 0 else 1


sys.exit(check(Path(sys.argv[1])))
PYTHON
