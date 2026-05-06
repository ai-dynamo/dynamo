# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOTICES-Apt.txt generator.

Enumerates installed dpkg packages via `dpkg-query -W` and resolves licenses
by parsing each package's `/usr/share/doc/<pkg>/copyright` file.

Two parsers:
  - DEP-5 (machine-readable copyright format): structured `License:` fields
    in `Files:` blocks. The most common shape on modern Debian/Ubuntu.
    https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
  - Free-form fallback: scan the file body for SPDX-shaped license tags or
    well-known license-name strings. Used when the copyright file lacks the
    DEP-5 `Format:` header (older / non-conforming packages).

Debian's traditional license short names are mapped to canonical SPDX IDs
via _DEBIAN_TO_SPDX (e.g. "Expat" → "MIT", "GPL-2+" → "GPL-2.0-or-later").

Anything we still can't resolve falls through to UNKNOWN, which the policy
gate rejects by default — fix via license_overrides.yaml.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from .common import UNKNOWN, Component
from ..license_db import lookup as license_db_lookup

logger = logging.getLogger(__name__)

ECOSYSTEM = "dpkg"

_COPYRIGHT_DIR = Path("/usr/share/doc")


# ---- Debian short-name → SPDX ID mapping ----------------------------------------


_DEBIAN_TO_SPDX: dict[str, str] = {
    # Permissive
    "Apache-2.0": "Apache-2.0",
    "Apache-2": "Apache-2.0",
    "Apache": "Apache-2.0",
    "BSD-2-clause": "BSD-2-Clause",
    "BSD-2-Clause": "BSD-2-Clause",
    "BSD-3-clause": "BSD-3-Clause",
    "BSD-3-Clause": "BSD-3-Clause",
    "BSD-4-clause": "BSD-4-Clause",
    "ISC": "ISC",
    "Expat": "MIT",  # Debian's name for MIT
    "MIT": "MIT",
    "MIT-0": "MIT-0",
    "MPL-1.1": "MPL-1.1",
    "MPL-2.0": "MPL-2.0",
    "Public-Domain": "CC0-1.0",
    "public-domain": "CC0-1.0",
    "Zlib": "Zlib",
    "zlib": "Zlib",
    "Boost": "BSL-1.0",
    "BSL-1.0": "BSL-1.0",
    "PSF-2.0": "Python-2.0",
    "Python": "Python-2.0",
    "Unlicense": "Unlicense",
    "0BSD": "0BSD",
    # GPL family — Debian uses both `GPL-N` and `GPL-N+` for or-later.
    "GPL-1": "GPL-1.0-only",
    "GPL-1+": "GPL-1.0-or-later",
    "GPL-2": "GPL-2.0-only",
    "GPL-2+": "GPL-2.0-or-later",
    "GPL-3": "GPL-3.0-only",
    "GPL-3+": "GPL-3.0-or-later",
    # LGPL
    "LGPL-2": "LGPL-2.0-only",
    "LGPL-2+": "LGPL-2.0-or-later",
    "LGPL-2.1": "LGPL-2.1-only",
    "LGPL-2.1+": "LGPL-2.1-or-later",
    "LGPL-3": "LGPL-3.0-only",
    "LGPL-3+": "LGPL-3.0-or-later",
    # AGPL
    "AGPL-3": "AGPL-3.0-only",
    "AGPL-3+": "AGPL-3.0-or-later",
    # Other GNU
    "FDL-1.2": "GFDL-1.2-only",
    "FDL-1.2+": "GFDL-1.2-or-later",
    "FDL-1.3": "GFDL-1.3-only",
    "FDL-1.3+": "GFDL-1.3-or-later",
    # Misc
    "Artistic": "Artistic-2.0",
    "Artistic-1.0": "Artistic-1.0",
    "Artistic-2.0": "Artistic-2.0",
    "OpenSSL": "OpenSSL",
    "WTFPL-2": "WTFPL",
    "WTFPL": "WTFPL",
    "CC-BY-3.0": "CC-BY-3.0",
    "CC-BY-4.0": "CC-BY-4.0",
    "CC-BY-SA-3.0": "CC-BY-SA-3.0",
    "CC-BY-SA-4.0": "CC-BY-SA-4.0",
    "CC0-1.0": "CC0-1.0",
    "Unicode-3.0": "Unicode-3.0",
    "Unicode-DFS-2016": "Unicode-DFS-2016",
}


def _normalize_debian_license(raw: str) -> str | None:
    """Map a Debian copyright `License:` header value to an SPDX ID, or None."""
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    # Direct lookup is fast and covers most cases.
    if s in _DEBIAN_TO_SPDX:
        return _DEBIAN_TO_SPDX[s]

    # Compound expressions (DEP-5 allows `GPL-2+ or LGPL-2.1+`)
    # Normalize each token, join with OR.
    if " or " in s.lower() and " and " not in s.lower():
        parts = re.split(r"\s+or\s+", s, flags=re.IGNORECASE)
        normalized = [_normalize_debian_license(p) for p in parts]
        normalized_clean = [n for n in normalized if n]
        if normalized_clean and len(normalized_clean) == len(parts):
            return " OR ".join(normalized_clean)

    if " and " in s.lower() and " or " not in s.lower():
        parts = re.split(r"\s+and\s+", s, flags=re.IGNORECASE)
        normalized = [_normalize_debian_license(p) for p in parts]
        normalized_clean = [n for n in normalized if n]
        if normalized_clean and len(normalized_clean) == len(parts):
            return " AND ".join(normalized_clean)

    # Strip common decorations
    cleaned = s.split(",")[0].strip()
    cleaned = re.sub(r"\s+\(.*?\)\s*$", "", cleaned).strip()
    if cleaned != s and cleaned in _DEBIAN_TO_SPDX:
        return _DEBIAN_TO_SPDX[cleaned]

    return None


# ---- Copyright file parsing -----------------------------------------------------


_DEP5_FORMAT_RE = re.compile(r"^Format:\s*https?://", re.IGNORECASE)


def _split_paragraphs(text: str) -> list[str]:
    """Split a DEP-5 file on blank lines into paragraphs."""
    return [p for p in re.split(r"\n[ \t]*\n", text) if p.strip()]


def _paragraph_headers(paragraph: str) -> dict[str, str]:
    """Return the set of top-level (non-continuation) `Header: value` pairs.

    DEP-5 uses RFC 822-style continuation lines: a line that begins with whitespace
    is a continuation of the previous header's value, not a new header. We only
    care about top-level header NAMES here, so we just look at lines that start
    in column 0 and contain a colon.
    """
    out: dict[str, str] = {}
    for line in paragraph.splitlines():
        if not line or line[0] in (" ", "\t"):
            continue  # continuation line — body of the previous header's value
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def _parse_dep5(text: str) -> str | None:
    """Extract a unioned SPDX expression from a DEP-5 copyright file.

    Only paragraphs that have a `Files:` header contribute to the package's
    effective license — those are the actual file-set declarations. Paragraphs
    that have only `License:` (no `Files:`) are standalone license-text
    definitions providing the body of a license referenced elsewhere; ignoring
    them avoids counting the same license twice (once as the Files: header,
    once as the standalone definition).

    Multiple `Files:` paragraphs with distinct License: headers are unioned
    with AND (the package contains ALL of them — different files have
    different licenses).
    """
    if not _DEP5_FORMAT_RE.search(text):
        return None  # not DEP-5

    licenses: set[str] = set()
    for paragraph in _split_paragraphs(text):
        headers = _paragraph_headers(paragraph)
        if "Files" not in headers:
            continue  # standalone license-text definition or document header
        license_value = headers.get("License", "").strip()
        if not license_value:
            continue
        normalized = _normalize_debian_license(license_value)
        if normalized:
            licenses.add(normalized)

    if not licenses:
        return None
    if len(licenses) == 1:
        return next(iter(licenses))
    return " AND ".join(sorted(licenses))


# Map Debian common-licenses filenames to SPDX IDs. Most legacy Debian/Ubuntu
# copyright files reference these instead of inlining the full license text:
#   "see /usr/share/common-licenses/GPL-3 for the full license"
# Detecting these references is the highest-leverage lever for cutting UNKNOWNs.
_COMMON_LICENSES_MAP: dict[str, str] = {
    "GPL": "GPL-2.0-or-later",        # default ambiguous; older convention
    "GPL-1": "GPL-1.0-only",
    "GPL-2": "GPL-2.0-only",
    "GPL-3": "GPL-3.0-only",
    "LGPL": "LGPL-2.1-or-later",      # default ambiguous; older convention
    "LGPL-2": "LGPL-2.0-only",
    "LGPL-2.1": "LGPL-2.1-only",
    "LGPL-3": "LGPL-3.0-only",
    "Apache-2.0": "Apache-2.0",
    "Artistic": "Artistic-2.0",
    "BSD": "BSD-3-Clause",
    "MPL-1.1": "MPL-1.1",
    "MPL-2.0": "MPL-2.0",
}

_COMMON_LICENSES_RE = re.compile(
    r"/usr/share/common-licenses/(GPL-[123](?:\.\d+)?|LGPL-[23](?:\.\d+)?|GPL|LGPL|Apache-2\.0|Artistic|BSD|MPL-[12]\.[01])\b"
)


# Free-form scan for SPDX-shaped tags or known license names anywhere in
# the copyright file body. Keep this conservative — we'd rather return None
# (UNKNOWN) than mis-classify.
_FREE_FORM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # SPDX-License-Identifier tag (highest-trust signal)
    (re.compile(r"\bSPDX-License-Identifier:\s*([A-Za-z0-9.+\- ]+(?:WITH [A-Za-z0-9.+\- ]+)?)"), "_match_spdx_tag"),

    # Versioned GPL/LGPL patterns. Specific versions first.
    (re.compile(r"\bGNU GENERAL PUBLIC LICENSE\s+Version 3\b", re.IGNORECASE), "GPL-3.0-only"),
    (re.compile(r"\bGNU GENERAL PUBLIC LICENSE\s+Version 2\b", re.IGNORECASE), "GPL-2.0-only"),
    (re.compile(r"\bGNU GENERAL PUBLIC LICENSE,? VERSION 3\b", re.IGNORECASE), "GPL-3.0-only"),
    (re.compile(r"\bGNU GENERAL PUBLIC LICENSE,? VERSION 2\b", re.IGNORECASE), "GPL-2.0-only"),
    (re.compile(r"\bGNU LESSER GENERAL PUBLIC LICENSE\s+Version 3\b", re.IGNORECASE), "LGPL-3.0-only"),
    (re.compile(r"\bGNU LESSER GENERAL PUBLIC LICENSE\s+Version 2\.1\b", re.IGNORECASE), "LGPL-2.1-only"),
    (re.compile(r"\bGNU LIBRARY GENERAL PUBLIC LICENSE\s+Version 2\b", re.IGNORECASE), "LGPL-2.0-only"),

    # Common phrases that don't include the version number — only match if
    # we couldn't find a versioned variant above.
    (re.compile(r"\bGNU GPL\s+v?(\d)\b", re.IGNORECASE), "_match_gpl_version"),
    (re.compile(r"\bGNU LGPL\s+v?(\d(?:\.\d)?)\b", re.IGNORECASE), "_match_lgpl_version"),
    (re.compile(r"\bGPL\s+(?:version\s+)?v?(\d(?:\.\d)?)\b", re.IGNORECASE), "_match_gpl_version"),
    (re.compile(r"\bLGPL\s+(?:version\s+)?v?(\d(?:\.\d)?)\b", re.IGNORECASE), "_match_lgpl_version"),

    # Apache
    (re.compile(r"\bApache License,?\s+Version 2\.0\b", re.IGNORECASE), "Apache-2.0"),
    (re.compile(r"\bApache License,?\s+Version 2\b", re.IGNORECASE), "Apache-2.0"),

    # MIT, BSD, MPL, ISC family
    (re.compile(r"\bThe MIT License\b", re.IGNORECASE), "MIT"),
    (re.compile(r"\b(?:Expat|MIT) license\b", re.IGNORECASE), "MIT"),
    (re.compile(r"\bBSD 3-Clause\b", re.IGNORECASE), "BSD-3-Clause"),
    (re.compile(r"\bBSD 2-Clause\b", re.IGNORECASE), "BSD-2-Clause"),
    (re.compile(r"\bMozilla Public License,?\s+v\.?\s*2\.0\b", re.IGNORECASE), "MPL-2.0"),
    (re.compile(r"\bISC License\b", re.IGNORECASE), "ISC"),
    (re.compile(r"\bzlib license\b", re.IGNORECASE), "Zlib"),
    (re.compile(r"\bcreative commons attribution[\s\-]+(?:share[\-\s]?alike\s+)?(\d(?:\.\d)?)\b", re.IGNORECASE), "_match_cc_version"),

    # Public domain
    (re.compile(r"\bpublic domain\b", re.IGNORECASE), "CC0-1.0"),
]


def _parse_free_form(text: str) -> str | None:
    """Best-effort scan for license markers in non-DEP-5 copyright files.

    Order of preference:
      1. SPDX-License-Identifier tag (highest trust)
      2. /usr/share/common-licenses/<NAME> reference (Debian-specific, very common)
      3. Versioned GPL/LGPL/Apache phrases
      4. Other well-known license-name strings
    """
    # /usr/share/common-licenses/ references — covers the bulk of Debian's
    # legacy copyright files which use this pattern instead of inline text.
    cl_match = _COMMON_LICENSES_RE.search(text)
    if cl_match:
        cl_name = cl_match.group(1)
        if cl_name in _COMMON_LICENSES_MAP:
            return _COMMON_LICENSES_MAP[cl_name]

    for pattern, marker in _FREE_FORM_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if marker == "_match_spdx_tag":
            spdx = m.group(1).strip()
            return _normalize_debian_license(spdx) or spdx
        if marker == "_match_gpl_version":
            v = m.group(1)
            return f"GPL-{v}.0-only" if "." not in v else f"GPL-{v}-only"
        if marker == "_match_lgpl_version":
            v = m.group(1)
            return f"LGPL-{v}-only" if "." in v else f"LGPL-{v}.0-only"
        if marker == "_match_cc_version":
            v = m.group(1)
            v_full = v if "." in v else f"{v}.0"
            sa = "SA-" if "share" in m.group(0).lower() and "alike" in m.group(0).lower() else ""
            return f"CC-BY-{sa}{v_full}"
        return marker
    return None


def _resolve_license(pkg_name: str, version: str = "") -> str:
    """Return the SPDX expression for a single dpkg package, or UNKNOWN.

    Resolution order:
      1. license_overrides.yaml (via license_db.lookup) — authoritative.
         For dpkg, the override takes priority over the copyright-file
         parser because the parser is heuristic and frequently wrong on
         vendored packages (e.g., NVIDIA's CUDA dpkgs reference
         /usr/share/common-licenses/GPL-3 because they bundle cuda-gdb,
         but the packages themselves are NVIDIA-proprietary). When we've
         explicitly verified a package's license and added it to overrides,
         that's the source of truth.
      2. DEP-5 copyright parse — structured Debian/Ubuntu format.
      3. Free-form copyright parse — pattern match common-licenses refs,
         "GPL Version 3" boilerplate, etc.
      4. UNKNOWN — surfaces in the policy gate.
    """
    overridden = license_db_lookup.lookup(ECOSYSTEM, pkg_name, version)
    if overridden:
        return overridden

    copyright_path = _COPYRIGHT_DIR / pkg_name / "copyright"
    if not copyright_path.is_file():
        return UNKNOWN

    try:
        text = copyright_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("could not read %s: %s", copyright_path, exc)
        return UNKNOWN

    spdx = _parse_dep5(text) or _parse_free_form(text)
    return spdx or UNKNOWN


# ---- Distribution scan ---------------------------------------------------------


def collect_components() -> list[Component]:
    """Run dpkg-query, resolve licenses, return Components.

    Returns an empty list (with a warning) when dpkg-query is unavailable —
    the dpkg generator is a no-op outside Debian/Ubuntu-derived environments
    (e.g. macOS dev shells, Alpine builders, distroless images). Inside the
    runtime images we ship, dpkg-query is always present.
    """
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}\\t${Version}\\n"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.warning(
            "dpkg-query not found; skipping dpkg generator. "
            "(Expected on macOS / Alpine / distroless; should not happen inside "
            "the dynamo runtime images.)"
        )
        return []

    components: list[Component] = []
    unresolved = 0
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        name, version = line.split("\t", 1)
        name = name.strip()
        version = version.strip()
        if not name or not version:
            continue
        spdx = _resolve_license(name, version)
        if spdx == UNKNOWN:
            unresolved += 1
        components.append(
            Component(
                ecosystem=ECOSYSTEM,
                name=name,
                version=version,
                spdx=spdx,
                source_url=None,  # apt-get source URL is the right answer; see source-archival
            )
        )

    logger.info(
        "Collected %d dpkg packages (%d unresolved → UNKNOWN)",
        len(components),
        unresolved,
    )
    return components


def generate(output_dir: Path) -> list[Component]:
    """Read dpkg state, write NOTICES-Apt.txt + dpkg-deps.csv."""
    from . import common

    components = collect_components()
    common.write_notices(ECOSYSTEM, components, output_dir)
    common.write_deps_csv(ECOSYSTEM, components, output_dir)
    return components
