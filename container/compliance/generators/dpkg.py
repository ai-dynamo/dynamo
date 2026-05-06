# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Apt.txt generator.

Enumerates installed dpkg packages via `dpkg-query -W` and resolves licenses
by parsing each package's DEP-5 `/usr/share/doc/<pkg>/copyright` file.
Falls back to syft-extracted license info when DEP-5 isn't parseable, and
to the license_db / license_overrides.yaml chain when neither works.

TODO: implement. Scope:
  - run `dpkg-query -W -f='${Package}\\t${Version}\\n'` to get the install set
  - for each pkg, read /usr/share/doc/<pkg>/copyright; parse DEP-5 header
    blocks for License: lines and merge into a single SPDX expression
  - syft fallback via license_db.lookup chain
  - render NOTICES-Apt.txt with full DEP-5 copyright file embedded inline
    where available (legal-distribution requirement is the whole DEP-5
    file, not just the SPDX ID)
"""

from __future__ import annotations

from pathlib import Path

from .common import Component

ECOSYSTEM = "dpkg"


def collect_components() -> list[Component]:
    raise NotImplementedError("dpkg generator not yet implemented")


def generate(output_dir: Path) -> list[Component]:
    raise NotImplementedError("dpkg generator not yet implemented")
