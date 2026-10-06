# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content hashes and deterministic provenance for the executable tooling package."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tool_provenance() -> dict[str, Any]:
    """Hash shipped implementation modules, including shared helpers and CLI.

    Tests, caches and generated reports are excluded. Labels are repository-relative;
    local filesystem paths never appear in public reports. Hashing the package avoids
    silently losing provenance when a helper moves or a new dependency is introduced.
    """
    package = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for path in package.rglob("*.py")
        if "tests" not in path.relative_to(package).parts
    )
    return {
        "python_version": platform.python_version(),
        "parser_versions": {
            name: version(name) for name in ("tree-sitter", "tree-sitter-rust")
        },
        "tools_sha256": {
            "scripts/protocol_compatibility/"
            + path.relative_to(package)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in paths
        },
    }
