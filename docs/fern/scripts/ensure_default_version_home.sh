#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ensure the default published version starts with the shared Home page.
#
# Fern's versioned navigation resolves the bare site URL through the first page
# of the default version; the top-level landing-page is not used for that
# redirect. Older release snapshots predate the Home tab, so composing them as
# the default version sends /dynamo to their Quickstart page. Normalize the
# default version against dev.yml so Home is canonical, unique, and first.

set -euo pipefail

fern_dir=${1:?usage: ensure_default_version_home.sh <fern-dir>}
docs_file="$fern_dir/docs.yml"
dev_nav="$fern_dir/versions/dev.yml"

if [ ! -f "$docs_file" ] || [ ! -f "$dev_nav" ]; then
  echo "ERROR: expected $docs_file and $dev_nav" >&2
  exit 1
fi

default_path=$(yq -r '.versions[0].path // ""' "$docs_file")
if [ -z "$default_path" ]; then
  echo "ERROR: docs.yml has no default version path" >&2
  exit 1
fi
default_nav="$fern_dir/${default_path#./}"
if [ ! -f "$default_nav" ]; then
  echo "ERROR: default version navigation does not exist: $default_nav" >&2
  exit 1
fi

home_tab=$(yq -r '.tabs.home // ""' "$dev_nav")
home_navigation=$(yq -r '.navigation[]? | select(.tab? == "home")' "$dev_nav")
if [ -z "$home_tab" ] || [ -z "$home_navigation" ] ||
   [ "$(yq '[.navigation[]? | select(.tab? == "home")] | length' "$dev_nav")" != "1" ]; then
  echo "ERROR: dev navigation must define exactly one shared Home tab" >&2
  exit 1
fi

if [ "$(yq '[.navigation[]? | select(.tab? == "home")] | length' "$default_nav")" = "1" ] &&
   [ "$(yq -r '.navigation[0].tab // ""' "$default_nav")" = "home" ] &&
   [ "$(yq -o=json '.tabs.home' "$default_nav")" = "$(yq -o=json '.tabs.home' "$dev_nav")" ] &&
   [ "$(yq -o=json '.navigation[0]' "$default_nav")" = "$(yq -o=json '.navigation[] | select(.tab == "home")' "$dev_nav")" ]; then
  echo "Default version already has canonical Home navigation: $default_nav"
  exit 0
fi

HOME_TAB="$home_tab" HOME_NAVIGATION="$home_navigation" \
  python3 - "$default_nav" <<'PY'
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)


def root_index(name: str) -> int:
    target = f"{name}:\n"
    try:
        return lines.index(target)
    except ValueError as exc:
        raise SystemExit(f"ERROR: cannot locate {name} root in {path}") from exc


def remove_tab_home() -> None:
    tabs = root_index("tabs")
    navigation = root_index("navigation")
    starts = [
        i
        for i in range(tabs + 1, navigation)
        if re.match(r"^  home:\s*(?:#.*)?$", lines[i].rstrip("\n"))
    ]
    for start in reversed(starts):
        end = start + 1
        while end < navigation and not re.match(r"^  [^ ].*:\s*(?:#.*)?$", lines[end].rstrip("\n")):
            end += 1
        del lines[start:end]


def remove_navigation_home() -> None:
    navigation = root_index("navigation")
    starts = [i for i in range(navigation + 1, len(lines)) if lines[i].startswith("  - ")]
    blocks = [(start, starts[index + 1] if index + 1 < len(starts) else len(lines)) for index, start in enumerate(starts)]
    for start, end in reversed(blocks):
        if re.match(r"^  - tab:\s*home\s*$", lines[start].rstrip("\n")):
            del lines[start:end]


def indented_block(value: str, *, sequence: bool) -> list[str]:
    source = value.rstrip("\n").splitlines()
    if sequence:
        return [f"  - {source[0]}\n", *[f"    {line}\n" for line in source[1:]]]
    return ["  home:\n", *[f"    {line}\n" for line in source]]


remove_navigation_home()
remove_tab_home()
lines[root_index("tabs") + 1 : root_index("tabs") + 1] = indented_block(os.environ["HOME_TAB"], sequence=False)
lines[root_index("navigation") + 1 : root_index("navigation") + 1] = indented_block(
    os.environ["HOME_NAVIGATION"], sequence=True
)
path.write_text("".join(lines), encoding="utf-8")
PY

if [ "$(yq '[.navigation[]? | select(.tab? == "home")] | length' "$default_nav")" != "1" ] || \
   [ "$(yq -r '.navigation[0].tab // ""' "$default_nav")" != "home" ] || \
   [ "$(yq -o=json '.tabs.home' "$default_nav")" != "$(yq -o=json '.tabs.home' "$dev_nav")" ] || \
   [ "$(yq -o=json '.navigation[0]' "$default_nav")" != "$(yq -o=json '.navigation[] | select(.tab == "home")' "$dev_nav")" ]; then
  echo "ERROR: failed to normalize Home navigation in $default_nav" >&2
  exit 1
fi

echo "Normalized shared Home navigation in default version: $default_nav"
