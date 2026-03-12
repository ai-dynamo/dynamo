# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""SPDX license normalization utilities.

Conservative approach: only assign SPDX identifiers for unambiguous matches.
Ambiguous cases return 'UNKNOWN'.
"""

import re

# Exact mapping from common free-text license strings to SPDX identifiers.
# Only includes unambiguous, exact matches.
COMMON_LICENSE_MAP: dict[str, str] = {
    # Exact SPDX identifiers (passthrough)
    "MIT": "MIT",
    "MIT-0": "MIT-0",
    "ISC": "ISC",
    "0BSD": "0BSD",
    "Zlib": "Zlib",
    "Unlicense": "Unlicense",
    "WTFPL": "WTFPL",
    "CC0-1.0": "CC0-1.0",
    "Apache-2.0": "Apache-2.0",
    "MPL-2.0": "MPL-2.0",
    "BSL-1.0": "BSL-1.0",
    "BSD-2-Clause": "BSD-2-Clause",
    "BSD-3-Clause": "BSD-3-Clause",
    "LGPL-2.0-only": "LGPL-2.0-only",
    "LGPL-2.0-or-later": "LGPL-2.0-or-later",
    "LGPL-2.1-only": "LGPL-2.1-only",
    "LGPL-2.1-or-later": "LGPL-2.1-or-later",
    "LGPL-3.0-only": "LGPL-3.0-only",
    "LGPL-3.0-or-later": "LGPL-3.0-or-later",
    "GPL-2.0-only": "GPL-2.0-only",
    "GPL-2.0-or-later": "GPL-2.0-or-later",
    "GPL-3.0-only": "GPL-3.0-only",
    "GPL-3.0-or-later": "GPL-3.0-or-later",
    "AGPL-3.0-only": "AGPL-3.0-only",
    "AGPL-3.0-or-later": "AGPL-3.0-or-later",
    "PSF-2.0": "PSF-2.0",
    "Artistic-2.0": "Artistic-2.0",
    "Unicode-3.0": "Unicode-3.0",
    "Unicode-DFS-2016": "Unicode-DFS-2016",
    # Common free-text variations (exact match only)
    "MIT License": "MIT",
    "The MIT License": "MIT",
    "The MIT License (MIT)": "MIT",
    "Apache License 2.0": "Apache-2.0",
    "Apache License, Version 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache-2": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "BSD": "BSD-3-Clause",
    "3-Clause BSD License": "BSD-3-Clause",
    "2-Clause BSD License": "BSD-2-Clause",
    "Simplified BSD License": "BSD-2-Clause",
    "New BSD License": "BSD-3-Clause",
    "ISC License": "ISC",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0": "MPL-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "Python Software Foundation License": "PSF-2.0",
    "PSF": "PSF-2.0",
    "Public Domain": "CC0-1.0",
    "public-domain": "CC0-1.0",
    "The Unlicense": "Unlicense",
    "Boost Software License 1.0": "BSL-1.0",
    "Boost Software License": "BSL-1.0",
}

# Python trove classifier to SPDX mapping (exact).
CLASSIFIER_TO_SPDX: dict[str, str] = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: OSI Approved :: Artistic License": "Artistic-2.0",
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
    "License :: OSI Approved :: Academic Free License (AFL)": "AFL-3.0",
    "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)": "EUPL-1.2",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
    "License :: Public Domain": "CC0-1.0",
}

# Known SPDX identifiers for DEP-5 License field values.
# DEP-5 spec uses these short names: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
_DEP5_LICENSE_MAP: dict[str, str] = {
    "Apache-2.0": "Apache-2.0",
    "Apache-2": "Apache-2.0",
    "Artistic": "Artistic-2.0",
    "Artistic-2.0": "Artistic-2.0",
    "BSD-2-clause": "BSD-2-Clause",
    "BSD-3-clause": "BSD-3-Clause",
    "BSL-1.0": "BSL-1.0",
    "CC0-1.0": "CC0-1.0",
    "Expat": "MIT",
    "GPL-2": "GPL-2.0-only",
    "GPL-2+": "GPL-2.0-or-later",
    "GPL-2.0": "GPL-2.0-only",
    "GPL-2.0+": "GPL-2.0-or-later",
    "GPL-3": "GPL-3.0-only",
    "GPL-3+": "GPL-3.0-or-later",
    "GPL-3.0": "GPL-3.0-only",
    "GPL-3.0+": "GPL-3.0-or-later",
    "ISC": "ISC",
    "LGPL-2": "LGPL-2.0-only",
    "LGPL-2+": "LGPL-2.0-or-later",
    "LGPL-2.0": "LGPL-2.0-only",
    "LGPL-2.0+": "LGPL-2.0-or-later",
    "LGPL-2.1": "LGPL-2.1-only",
    "LGPL-2.1+": "LGPL-2.1-or-later",
    "LGPL-3": "LGPL-3.0-only",
    "LGPL-3+": "LGPL-3.0-or-later",
    "LGPL-3.0": "LGPL-3.0-only",
    "LGPL-3.0+": "LGPL-3.0-or-later",
    "MIT": "MIT",
    "MPL-2.0": "MPL-2.0",
    "PSF-2": "PSF-2.0",
    "public-domain": "CC0-1.0",
    "Zlib": "Zlib",
    "OpenSSL": "OpenSSL",
    "Unicode": "Unicode-3.0",
    "WTFPL": "WTFPL",
}


def normalize_license(raw: str) -> str:
    """Normalize a free-text license string to an SPDX identifier.

    Conservative: returns 'UNKNOWN' for anything that isn't an exact match.
    """
    if not raw or not raw.strip():
        return "UNKNOWN"

    stripped = raw.strip()

    # Direct lookup
    if stripped in COMMON_LICENSE_MAP:
        return COMMON_LICENSE_MAP[stripped]

    # Case-insensitive lookup
    stripped_lower = stripped.lower()
    for key, val in COMMON_LICENSE_MAP.items():
        if key.lower() == stripped_lower:
            return val

    return "UNKNOWN"


def normalize_dep5_license(license_field: str) -> str:
    """Normalize a DEP-5 License: field value to SPDX.

    DEP-5 License fields can be simple (e.g., 'MIT') or compound
    (e.g., 'GPL-2+ with OpenSSL exception'). We only handle simple cases.
    """
    if not license_field or not license_field.strip():
        return "UNKNOWN"

    stripped = license_field.strip()

    # Direct DEP-5 lookup
    if stripped in _DEP5_LICENSE_MAP:
        return _DEP5_LICENSE_MAP[stripped]

    # Case-insensitive
    stripped_lower = stripped.lower()
    for key, val in _DEP5_LICENSE_MAP.items():
        if key.lower() == stripped_lower:
            return val

    return "UNKNOWN"


def classifier_to_spdx(classifier: str) -> str | None:
    """Map a Python trove classifier to an SPDX identifier.

    Returns None if no mapping exists.
    """
    return CLASSIFIER_TO_SPDX.get(classifier)


def parse_dep5_licenses(content: str) -> list[str]:
    """Extract unique License: field values from a DEP-5 format copyright file.

    Returns a list of raw license strings found in License: fields.
    """
    licenses = []
    for line in content.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("License:"):
            value = line_stripped[len("License:"):].strip()
            if value and value not in licenses:
                licenses.append(value)
    return licenses


def is_dep5_format(content: str) -> bool:
    """Check if a copyright file is in DEP-5 machine-readable format."""
    # DEP-5 files start with a Format: field
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return bool(re.match(r"^Format:\s+", stripped))
    return False
