# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ecosystem-scoped inventory identities; Python uses PEP 503 normalization."""

import re

PURL_ECOSYSTEMS = {
    "pypi": "python",
    "cargo": "rust",
    "golang": "go",
    "deb": "dpkg",
    "generic": "native",
}


ARTIFACT_ECOSYSTEMS = {
    "rust-crate": "rust",
    "go-module": "go",
    "python": "python",
    "deb": "dpkg",
}


def identity(ecosystem: str, name: str) -> tuple[str, str]:
    if ecosystem == "python":
        name = re.sub(r"[-_.]+", "-", name).lower()
    return ecosystem, name


def component_identity(component: dict) -> tuple[str, str]:
    purl = component.get("purl") or ""
    ptype = purl[4:].split("/", 1)[0] if purl.startswith("pkg:") else ""
    ecosystem = PURL_ECOSYSTEMS.get(ptype)
    if ecosystem is None:
        properties = {
            entry["name"]: entry["value"] for entry in component.get("properties", [])
        }
        ecosystem = ARTIFACT_ECOSYSTEMS.get(
            properties.get("syft:package:type"), "native"
        )
    return identity(ecosystem, component["name"])
