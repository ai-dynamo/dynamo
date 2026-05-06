# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Python.txt generator.

Walks each search path's site-packages, reads `*.dist-info/METADATA` via
`importlib.metadata`, and normalizes the license string to an SPDX expression.

Resolution preference per package (PEP 639 ordering):
  1. `License-Expression: <SPDX>` header (PEP 639, modern wheels)
  2. `License: <free-form>` header (legacy, normalized via _FREE_FORM_MAP)
  3. `Classifier: License :: ...` headers (normalized via _CLASSIFIER_MAP)
  4. UNKNOWN — surfaces in the policy gate; fix via license_overrides.yaml

Why importlib.metadata vs pip-licenses:
  - Zero extra installs in the runtime image (pip-licenses + transitively
    prettytable would otherwise need ephemeral install + post-install cleanup
    to avoid polluting NOTICES with the licensing tool itself).
  - importlib.metadata is in stdlib (3.8+) and reads the same METADATA
    files pip-licenses parses.

First-party packages (`ai-dynamo`, `ai-dynamo-runtime`, `kvbm`, `nixl_*`,
`nvidia-*`, `dynamo-*`) are KEPT in the output — same principle as Rust.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .common import UNKNOWN, Component, dedupe_by_name_version

logger = logging.getLogger(__name__)

ECOSYSTEM = "python"


# ---- License normalization mappings ----------------------------------------------


# PyPI classifier-string → canonical SPDX ID. Sorted longest-prefix-first
# so more-specific classifiers match before less-specific ones.
# Source: PEP 639 + cross-reference with SPDX license-list-data.
_CLASSIFIER_MAP: list[tuple[str, str]] = [
    ("License :: OSI Approved :: Apache Software License", "Apache-2.0"),
    ("License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)", "AGPL-3.0-or-later"),
    ("License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)", "AGPL-3.0-only"),
    ("License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)", "GPL-2.0-or-later"),
    ("License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)", "GPL-3.0-or-later"),
    ("License :: OSI Approved :: GNU General Public License v2 (GPLv2)", "GPL-2.0-only"),
    ("License :: OSI Approved :: GNU General Public License v3 (GPLv3)", "GPL-3.0-only"),
    ("License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)", "LGPL-2.1-or-later"),
    ("License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)", "LGPL-3.0-or-later"),
    ("License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)", "LGPL-2.1-only"),
    ("License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)", "LGPL-3.0-only"),
    ("License :: OSI Approved :: ISC License (ISCL)", "ISC"),
    ("License :: OSI Approved :: MIT License", "MIT"),
    ("License :: OSI Approved :: MIT No Attribution License (MIT-0)", "MIT-0"),
    ("License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)", "MPL-2.0"),
    ("License :: OSI Approved :: Python Software Foundation License", "Python-2.0"),
    ("License :: OSI Approved :: BSD License", "BSD-3-Clause"),
    ("License :: OSI Approved :: zlib/libpng License", "Zlib"),
    ("License :: OSI Approved :: Universal Permissive License (UPL)", "UPL-1.0"),
    ("License :: OSI Approved :: The Unlicense (Unlicense)", "Unlicense"),
    ("License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)", "CDDL-1.0"),
    ("License :: OSI Approved :: Eclipse Public License 1.0 (EPL-1.0)", "EPL-1.0"),
    ("License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)", "EPL-2.0"),
    ("License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)", "BSL-1.0"),
    ("License :: Public Domain", "CC0-1.0"),
]


# Free-form License field values seen in the wild → SPDX ID.
# Keys are matched case-insensitively, longest first.
_FREE_FORM_MAP: list[tuple[str, str]] = [
    ("apache-2.0 with llvm-exception", "Apache-2.0 WITH LLVM-exception"),
    ("apache 2.0", "Apache-2.0"),
    ("apache license 2.0", "Apache-2.0"),
    ("apache software license", "Apache-2.0"),
    ("apache-2.0", "Apache-2.0"),
    ("apache2", "Apache-2.0"),
    ("agpl-3.0-or-later", "AGPL-3.0-or-later"),
    ("agpl-3.0-only", "AGPL-3.0-only"),
    ("agpl-3.0", "AGPL-3.0-only"),
    ("agplv3+", "AGPL-3.0-or-later"),
    ("agplv3", "AGPL-3.0-only"),
    ("bsd 3-clause license", "BSD-3-Clause"),
    ("bsd-3-clause", "BSD-3-Clause"),
    ("bsd 2-clause license", "BSD-2-Clause"),
    ("bsd-2-clause", "BSD-2-Clause"),
    ("new bsd license", "BSD-3-Clause"),
    ("bsd license", "BSD-3-Clause"),
    ("3-clause bsd", "BSD-3-Clause"),
    ("2-clause bsd", "BSD-2-Clause"),
    ("bsd", "BSD-3-Clause"),
    ("gpl-2.0-or-later", "GPL-2.0-or-later"),
    ("gpl-2.0-only", "GPL-2.0-only"),
    ("gpl-3.0-or-later", "GPL-3.0-or-later"),
    ("gpl-3.0-only", "GPL-3.0-only"),
    ("gplv2+", "GPL-2.0-or-later"),
    ("gplv2", "GPL-2.0-only"),
    ("gplv3+", "GPL-3.0-or-later"),
    ("gplv3", "GPL-3.0-only"),
    ("lgpl-3.0", "LGPL-3.0-only"),
    ("lgpl-2.1", "LGPL-2.1-only"),
    ("lgplv3+", "LGPL-3.0-or-later"),
    ("lgplv3", "LGPL-3.0-only"),
    ("isc license", "ISC"),
    ("isc", "ISC"),
    ("mit-0", "MIT-0"),
    ("mit license", "MIT"),
    ("mit", "MIT"),
    ("mpl-2.0", "MPL-2.0"),
    ("mozilla public license 2.0", "MPL-2.0"),
    ("mozilla public license, v. 2.0", "MPL-2.0"),
    ("psf-2.0", "Python-2.0"),
    ("python software foundation license", "Python-2.0"),
    ("zlib", "Zlib"),
    ("zlib/libpng", "Zlib"),
    ("the unlicense", "Unlicense"),
    ("unlicense", "Unlicense"),
    ("public domain", "CC0-1.0"),
    ("cc0-1.0", "CC0-1.0"),
    ("0bsd", "0BSD"),
    ("bsl-1.0", "BSL-1.0"),
    ("boost software license 1.0", "BSL-1.0"),
    ("cddl-1.0", "CDDL-1.0"),
    ("epl-2.0", "EPL-2.0"),
    ("eupl-1.2", "EUPL-1.2"),
    ("unicode-3.0", "Unicode-3.0"),
    ("unicode-dfs-2016", "Unicode-DFS-2016"),
]


# Already-canonical SPDX IDs we should pass through as-is. Cuts noise on
# packages that already publish a clean SPDX expression.
_PASSTHROUGH_SPDX = frozenset({
    "0BSD", "AGPL-3.0-only", "AGPL-3.0-or-later", "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause", "BSD-3-Clause",
    "BSL-1.0", "CC0-1.0", "CDDL-1.0", "CDLA-Permissive-2.0", "EPL-1.0",
    "EPL-2.0", "EUPL-1.2", "GPL-2.0-only", "GPL-2.0-or-later",
    "GPL-3.0-only", "GPL-3.0-or-later", "ISC", "LGPL-2.1-only",
    "LGPL-2.1-or-later", "LGPL-3.0", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "MIT", "MIT-0", "MPL-2.0", "NCSA", "OpenSSL", "Python-2.0",
    "Unicode-3.0", "Unicode-DFS-2016", "Unlicense", "WTFPL", "Zlib",
})


def _normalize(raw: str | None) -> str | None:
    """Map a free-form / classifier license string to an SPDX ID, or None."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    # Compound-expression passthrough (already SPDX-shaped)
    if any(op in s for op in (" AND ", " OR ", " WITH ")):
        return s

    if s in _PASSTHROUGH_SPDX:
        return s

    if s.startswith("License :: "):
        for prefix, spdx in _CLASSIFIER_MAP:
            if s.startswith(prefix):
                return spdx
        return None

    s_lower = s.lower()
    for pattern, spdx in _FREE_FORM_MAP:
        if pattern == s_lower or pattern in s_lower:
            return spdx

    return None


# ---- Distribution scan ----------------------------------------------------------


def _resolve_site_packages(search_paths: list[Path]) -> list[Path]:
    """Mirror rust.py's strategy: each search path is either a venv root
    (in which case we glob lib/python*/site-packages) or a site-packages
    directory directly. Returns a deduped list of site-packages paths.
    """
    out: set[Path] = set()
    for p in search_paths:
        for site in p.glob("lib/python*/site-packages"):
            if site.is_dir():
                out.add(site.resolve())
        if p.is_dir() and any(p.glob("*.dist-info")):
            out.add(p.resolve())
    return sorted(out)


def _read_license_text(dist_info_dir: Path) -> str | None:
    """Return the upstream LICENSE text for a wheel, if present.

    Modern wheels put it under <dist-info>/licenses/LICENSE (PEP 639). Older
    wheels use <dist-info>/LICENSE or LICENSE.txt at the top of the dir.
    """
    candidates = [
        dist_info_dir / "licenses" / "LICENSE",
        dist_info_dir / "licenses" / "LICENSE.txt",
        dist_info_dir / "LICENSE",
        dist_info_dir / "LICENSE.txt",
        dist_info_dir / "LICENSE.md",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


_NAME_RE = re.compile(r"^Name:\s*(.+)$", re.MULTILINE)
_VERSION_RE = re.compile(r"^Version:\s*(.+)$", re.MULTILINE)
_LICENSE_EXPR_RE = re.compile(r"^License-Expression:\s*(.+)$", re.MULTILINE)
_LICENSE_RE = re.compile(r"^License:\s*(.+?)(?=^[A-Z][a-zA-Z\-]*:|\Z)", re.MULTILINE | re.DOTALL)
_CLASSIFIER_RE = re.compile(r"^Classifier:\s*(.+)$", re.MULTILINE)


def _extract_metadata_field(metadata_text: str) -> tuple[str | None, str | None, str | None, list[str], list[str]]:
    """Return (name, version, license_expression, license_free_form_lines, classifiers).

    `license_free_form_lines` may be multi-line if the License: header has
    continuation lines. Returned as a list of stripped lines for caller flexibility.
    """
    name = m.group(1).strip() if (m := _NAME_RE.search(metadata_text)) else None
    version = m.group(1).strip() if (m := _VERSION_RE.search(metadata_text)) else None
    license_expr = m.group(1).strip() if (m := _LICENSE_EXPR_RE.search(metadata_text)) else None

    license_free = []
    if (m := _LICENSE_RE.search(metadata_text)):
        for line in m.group(1).splitlines():
            stripped = line.strip()
            if stripped:
                license_free.append(stripped)

    classifiers = [m.group(1).strip() for m in _CLASSIFIER_RE.finditer(metadata_text)]
    return name, version, license_expr, license_free, classifiers


def _resolve_spdx(license_expr: str | None, license_free: list[str], classifiers: list[str]) -> str:
    """PEP-639-ordered: License-Expression > License free-form > Classifiers."""
    if license_expr:
        return license_expr  # treat as authoritative SPDX expression
    for line in license_free:
        spdx = _normalize(line)
        if spdx:
            return spdx
    for cls in classifiers:
        if cls.startswith("License :: "):
            spdx = _normalize(cls)
            if spdx:
                return spdx
    return UNKNOWN


def collect_components(search_paths: list[Path]) -> list[Component]:
    """Walk each site-packages dir, parse *.dist-info/METADATA, return Components."""
    sites = _resolve_site_packages(search_paths)
    if not sites:
        logger.warning("No site-packages dirs resolved from %s", search_paths)
        return []

    components: list[Component] = []
    for site in sites:
        dist_infos = sorted(site.glob("*.dist-info"))
        for dist_info in dist_infos:
            metadata_path = dist_info / "METADATA"
            if not metadata_path.is_file():
                continue
            try:
                metadata_text = metadata_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", metadata_path, exc)
                continue

            name, version, license_expr, license_free, classifiers = _extract_metadata_field(metadata_text)
            if not name or not version:
                logger.warning("dist-info %s missing Name/Version", dist_info)
                continue

            spdx = _resolve_spdx(license_expr, license_free, classifiers)
            components.append(
                Component(
                    ecosystem=ECOSYSTEM,
                    name=name,
                    version=version,
                    spdx=spdx,
                    source_url=f"https://pypi.org/project/{name}/{version}/",
                    license_text=_read_license_text(dist_info),
                )
            )

    deduped = dedupe_by_name_version(components)
    logger.info(
        "Collected %d unique python packages (from %d dist-info dirs across %d site-packages roots)",
        len(deduped),
        len(components),
        len(sites),
    )
    return deduped


def generate(search_paths: list[Path], output_dir: Path) -> list[Component]:
    """Read METADATA from each search path, write NOTICES-Python.txt + python-deps.csv."""
    from . import common

    components = collect_components(search_paths)
    common.write_notices(ECOSYSTEM, components, output_dir)
    common.write_deps_csv(ECOSYSTEM, components, output_dir)
    return components
