#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ensure the default published version starts with the shared Home page.
#
# Fern's versioned navigation resolves the bare site URL through the first page
# of the default version; the top-level landing-page is not used for that
# redirect. Older release snapshots predate the Home tab, so composing them as
# the default version sends /dynamo to their Quickstart page. Add the shared
# Home tab and navigation entry when they are missing.

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

if [ "$(yq '[.navigation[]? | select(.tab? == "home")] | length' "$default_nav")" != "0" ]; then
  echo "Default version already includes Home: $default_nav"
  exit 0
fi

if [ "$(yq '[.navigation[]? | select(.tab? == "home")] | length' "$dev_nav")" = "0" ]; then
  echo "ERROR: dev navigation does not define the shared Home tab" >&2
  exit 1
fi

python3 - "$default_nav" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

tabs = """tabs:
  home:
    display-name: Home
    icon: house
    skip-slug: true
"""
navigation = """navigation:
  - tab: home
    layout:
      - page: Welcome
        path: ../index.mdx
"""

if "\ntabs:\n" not in text or "\nnavigation:\n" not in text:
    raise SystemExit(f"ERROR: cannot locate tabs/navigation roots in {path}")

text = text.replace("\ntabs:\n", "\n" + tabs, 1)
text = text.replace("\nnavigation:\n", "\n" + navigation, 1)
path.write_text(text, encoding="utf-8")
PY

echo "Added shared Home navigation to default version: $default_nav"
