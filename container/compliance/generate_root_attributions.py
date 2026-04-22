#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate repo-root ATTRIBUTIONS-*.md for direct dependencies.

This is the repo-root counterpart to the in-container SBOM-based attribution
generator (container/compliance/sbom/render_attributions.py). The in-container
one is comprehensive; this one is minimal.

Usage:
    python3 container/compliance/generate_root_attributions.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent / "generate-attributions-md" / "src"
sys.path.insert(0, str(PKG))

from dynamo_attributions.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
