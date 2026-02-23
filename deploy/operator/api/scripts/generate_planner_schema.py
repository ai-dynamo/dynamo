#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate planner_config_json_schema.json from PlannerConfig.model_json_schema().

This script is a lightweight alternative to running
``python -m dynamo.planner.utils.planner_config`` that works without the full
dynamo runtime installed (no ``dynamo.runtime`` package required).

It stubs out the two heavy dependencies that are not needed for schema
generation:

* ``dynamo.runtime.logging``  (called at module load time in defaults.py)
* ``dynamo.planner`` __init__ (imports connectors that require dynamo.runtime)

Usage
-----
    python deploy/operator/api/scripts/generate_planner_schema.py

Or via the Makefile::

    make generate-planner-schema
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path


def _resolve_repo_root(start: Path) -> Path:
    """Return the repository root via git, falling back to go.mod traversal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=start,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Fallback: walk up until we find go.mod
    p = start
    while p != p.parent:
        if (p / "go.mod").exists():
            return p
        p = p.parent
    raise RuntimeError(
        f"Could not locate repository root from {start}. "
        "Ensure the script is run inside the dynamo repository."
    )


_REPO_ROOT = _resolve_repo_root(Path(__file__).resolve().parent)
_COMPONENTS_SRC = _REPO_ROOT / "components" / "src"

# ---------------------------------------------------------------------------
# Stub dynamo.runtime.logging BEFORE any dynamo sub-module is imported.
# defaults.py calls configure_dynamo_logging() at module-load time.
# ---------------------------------------------------------------------------
_runtime_mod = types.ModuleType("dynamo.runtime")
_logging_mod = types.ModuleType("dynamo.runtime.logging")
_logging_mod.configure_dynamo_logging = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("dynamo", types.ModuleType("dynamo"))
sys.modules.setdefault("dynamo.runtime", _runtime_mod)
sys.modules["dynamo.runtime.logging"] = _logging_mod

# ---------------------------------------------------------------------------
# Register a bare namespace for dynamo.planner so Python uses the filesystem
# for sub-packages (utils/, etc.) but skips the heavy __init__.py which would
# otherwise attempt to import KubernetesConnector, GlobalPlannerConnector, etc.
# ---------------------------------------------------------------------------
_planner_path = str(_COMPONENTS_SRC / "dynamo" / "planner")
_planner_mod = types.ModuleType("dynamo.planner")
_planner_mod.__path__ = [_planner_path]  # type: ignore[attr-defined]
_planner_mod.__package__ = "dynamo.planner"
sys.modules["dynamo.planner"] = _planner_mod

# ---------------------------------------------------------------------------
# Now add components/src to sys.path and import the target module normally.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_COMPONENTS_SRC))

from dynamo.planner.utils.planner_config import PlannerConfig  # noqa: E402

schema = PlannerConfig.model_json_schema()

output_path = (
    _COMPONENTS_SRC / "dynamo" / "planner" / "utils" / "planner_config_json_schema.json"
)
output_path.write_text(json.dumps(schema, indent=2) + "\n")
print(f"PlannerConfig JSON schema written to {output_path}")
