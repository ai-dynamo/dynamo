# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Site settings for the Slurm CPU lane (stdlib only).

``load()`` reads ``KEY=VALUE`` lines from the file named by ``LR_SITE_ENV``, else ``site.env`` next
to this module, and sets every key that is not already in the environment, so environment variables
win. Values may be quoted and may reference ``$VAR``; ``#`` starts a comment. ``common.sh`` reads
the same file with the same rules for the shell scripts.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def site_env_path() -> Path:
    return Path(os.environ.get("LR_SITE_ENV") or Path(__file__).with_name("site.env"))


def load(path: Path | None = None) -> Path | None:
    path = path or site_env_path()
    if not path.is_file():
        return None
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.isidentifier():
            raise SystemExit(f"{path}:{number}: expected KEY=VALUE, got {raw!r}")
        words = shlex.split(value, comments=True)
        if key not in os.environ:
            os.environ[key] = os.path.expandvars(words[0] if words else "")
    return path


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"error: set {name} in {site_env_path()} (see site.env.example) or the environment"
        )
    return value
