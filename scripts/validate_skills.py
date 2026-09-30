#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate agent skills under .agents/skills/ against the conventions
documented in the Skills section of AGENTS.md.

For every .agents/skills/<dir>/SKILL.md this checks that:
  - the file starts with a parseable YAML frontmatter block
  - frontmatter `name` equals the directory name and is kebab-case
  - `description` is a non-empty string of at most 1024 characters
  - `license` is Apache-2.0
  - a `metadata` block is present with a non-empty `author` and `tags` list

It also checks that the Skills section of AGENTS.md lists every skill
directory (and lists nothing that does not exist), so the index cannot
drift from the actual skill set.

Usage: validate_skills.py [repo_root]
"""

import re
import sys
from pathlib import Path

import yaml

MAX_DESCRIPTION_LENGTH = 1024
REQUIRED_LICENSE = "Apache-2.0"
KEBAB_CASE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SKILLS_HEADING = re.compile(r"^(#{1,6})\s+Skills\s*$", re.IGNORECASE)
# Index entries look like "- `skill-name` — one-line summary." and may list
# several sibling skills before the em-dash separator, e.g.
# "- `dep-create` / `dep-status` / `dep-update` — manage DEPs". The em-dash
# distinguishes them from convention bullets like "- `name` must ...".
INDEX_ENTRY = re.compile(r"^-\s+(`[^`]+`(?:\s*/\s*`[^`]+`)*)\s+—")
INDEX_NAME = re.compile(r"`([a-z0-9][a-z0-9-]*)`")


def parse_frontmatter(text):
    """Return (frontmatter dict, error message); exactly one is None."""
    if not text.startswith("---\n"):
        return None, "missing YAML frontmatter (file must start with '---')"
    end = text.find("\n---", 4)
    if end == -1:
        return None, "unterminated YAML frontmatter (no closing '---')"
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        return None, f"invalid YAML frontmatter: {exc}"
    if not isinstance(data, dict):
        return None, "YAML frontmatter is not a mapping"
    return data, None


def check_skill(skill_dir, errors):
    skill_md = skill_dir / "SKILL.md"
    rel = skill_md.relative_to(skill_dir.parent.parent.parent)
    if not skill_md.is_file():
        errors.append(f"{rel}: missing SKILL.md in skill directory")
        return

    data, err = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    if err:
        errors.append(f"{rel}: {err}")
        return

    name = data.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{rel}: frontmatter 'name' is missing or empty")
    else:
        if name != skill_dir.name:
            errors.append(
                f"{rel}: frontmatter name '{name}' does not match "
                f"directory name '{skill_dir.name}'"
            )
        if not KEBAB_CASE.match(name):
            errors.append(f"{rel}: name '{name}' is not kebab-case")

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{rel}: frontmatter 'description' is missing or empty")
    elif len(description.strip()) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"{rel}: description is {len(description.strip())} characters "
            f"(max {MAX_DESCRIPTION_LENGTH})"
        )

    if data.get("license") != REQUIRED_LICENSE:
        errors.append(
            f"{rel}: frontmatter 'license' must be '{REQUIRED_LICENSE}' "
            f"(got {data.get('license')!r})"
        )

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"{rel}: frontmatter 'metadata' block is missing")
        return
    author = metadata.get("author")
    if not isinstance(author, str) or not author.strip():
        errors.append(f"{rel}: metadata 'author' is missing or empty")
    tags = metadata.get("tags")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(t, str) and t.strip() for t in tags)
    ):
        errors.append(f"{rel}: metadata 'tags' must be a non-empty list of strings")


def skills_section(agents_md_text):
    """Return the body of the Skills section of AGENTS.md, or None."""
    lines = agents_md_text.splitlines()
    level = None
    body = []
    for line in lines:
        heading = re.match(r"^(#{1,6})\s", line)
        if level is None:
            match = SKILLS_HEADING.match(line)
            if match:
                level = len(match.group(1))
        elif heading and len(heading.group(1)) <= level:
            break
        else:
            body.append(line)
    return "\n".join(body) if level is not None else None


def check_index(agents_md, skill_names, errors):
    if not agents_md.is_file():
        errors.append(f"{agents_md.name}: file not found")
        return
    section = skills_section(agents_md.read_text(encoding="utf-8"))
    if section is None:
        errors.append(f"{agents_md.name}: no 'Skills' section heading found")
        return
    listed = set()
    for line in section.splitlines():
        match = INDEX_ENTRY.match(line)
        if match:
            listed.update(INDEX_NAME.findall(match.group(1)))
    for name in sorted(skill_names - listed):
        errors.append(
            f"{agents_md.name}: skill '{name}' is missing from the Skills index"
        )
    for name in sorted(listed - skill_names):
        errors.append(
            f"{agents_md.name}: Skills index lists '{name}' but "
            f".agents/skills/{name}/ does not exist"
        )


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
    skills_dir = root / ".agents" / "skills"
    if not skills_dir.is_dir():
        print(f"error: {skills_dir} is not a directory", file=sys.stderr)
        return 1

    errors = []
    skill_dirs = sorted(
        d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    for skill_dir in skill_dirs:
        check_skill(skill_dir, errors)
    check_index(root / "AGENTS.md", {d.name for d in skill_dirs}, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"\n{len(errors)} skill validation error(s)", file=sys.stderr)
        return 1
    print(f"validated {len(skill_dirs)} skills: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
