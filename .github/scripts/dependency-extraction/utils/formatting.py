# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Formatting utilities for dependency names and notes.

This module contains functions for formatting dependency names, stripping suffixes,
and creating human-readable notes.
"""

import re

from ..constants import NORMALIZATIONS, PYTORCH_EXCEPTIONS


def format_package_name(name: str, category: str) -> str:
    """Format a package/module name to be human-readable."""
    # Handle special cases and well-known packages
    special_cases = {
        "fastapi": "FastAPI",
        "numpy": "NumPy",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "kubernetes": "Kubernetes",
        "pydantic": "Pydantic",
        "openai": "OpenAI",
        "httpx": "HTTPX",
        "uvicorn": "Uvicorn",
        "pytest": "pytest",
        "mypy": "mypy",
        "pyright": "Pyright",
        "golang": "Go",
        "grpc": "gRPC",
        "protobuf": "Protocol Buffers",
        "yaml": "YAML",
        "toml": "TOML",
        "json": "JSON",
        "jwt": "JWT",
        "oauth": "OAuth",
        "redis": "Redis",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "etcd": "etcd",
        "nats": "NATS",
        "cuda": "CUDA",
        "nvidia": "NVIDIA",
        "asyncio": "asyncio",
        "aiohttp": "aiohttp",
        "sqlalchemy": "SQLAlchemy",
        "alembic": "Alembic",
        "celery": "Celery",
        "flask": "Flask",
        "django": "Django",
        "jinja2": "Jinja2",
    }

    name_lower = name.lower()
    if name_lower in special_cases:
        return special_cases[name_lower]

    # Check for partial matches in the name
    for key, value in special_cases.items():
        if key in name_lower:
            return (
                name.replace(key, value)
                .replace(key.upper(), value)
                .replace(key.capitalize(), value)
            )

    # Handle hyphen-separated or underscore-separated names
    if "-" in name or "_" in name:
        words = re.split(r"[-_]", name)
        formatted_words = []
        for word in words:
            # Keep acronyms uppercase (short all-caps words)
            if word.isupper() and len(word) <= 4:
                formatted_words.append(word)
            # Make 1-2 letter words uppercase (likely acronyms like "io", "db")
            elif len(word) <= 2:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.capitalize())
        return " ".join(formatted_words)

    # Handle camelCase by inserting spaces
    if any(c.isupper() for c in name[1:]) and not name.isupper():
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        return spaced

    # Default: capitalize first letter
    return name.capitalize() if name else name


def strip_version_suffixes(name: str) -> str:
    """Remove common version-related suffixes from dependency names."""
    # Common suffixes that don't add value (version info is in separate column)
    suffixes = [" Ver", " Version", " Ref", " Tag"]

    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()

    return name


def format_dependency_name(name: str, category: str, version: str) -> str:
    """Format dependency name to be human-readable and well-formatted."""
    # Handle URLs and Git repositories
    if "git+" in name or name.startswith("http://") or name.startswith("https://"):
        # Extract repository name from URL
        parts = name.rstrip("/").split("/")
        if len(parts) >= 2:
            repo_name = parts[-1].replace(".git", "")
            # Convert kebab-case or snake_case to Title Case
            formatted = " ".join(
                word.capitalize() for word in re.split(r"[-_]", repo_name)
            )
            return strip_version_suffixes(formatted)
        return name

    # Handle package names with extras (e.g., "package[extra]")
    if "[" in name and "]" in name:
        base_name = name.split("[")[0]
        extras = name[name.find("[") : name.find("]") + 1]
        formatted_base = format_package_name(base_name, category)
        return f"{strip_version_suffixes(formatted_base)} {extras}"

    # Handle Go modules - keep full path for uniqueness
    if category == "Go Module":
        # For Go modules, we want to keep the full import path to avoid ambiguity
        # Different packages may have the same last component but different domains
        # e.g., "emperror.dev/errors" vs "github.com/pkg/errors"
        return name  # Return as-is, no formatting needed

    # Handle Docker base images
    if category == "Base Image":
        # Format: "nvcr.io/nvidia/pytorch" -> "NVIDIA PyTorch"
        if "/" in name and "nvidia" in name.lower():
            parts = name.split("/")
            image_name = parts[-1]
            return f"NVIDIA {strip_version_suffixes(format_package_name(image_name, category))}"
        elif "/" in name:
            # Generic format: use last part
            parts = name.split("/")
            return strip_version_suffixes(format_package_name(parts[-1], category))

    # Handle ARG/ENV variable names that are already formatted (e.g., "Base Image Tag")
    if " " in name and name[0].isupper():
        return strip_version_suffixes(name)

    # Default: format as a package name
    return strip_version_suffixes(format_package_name(name, category))


def format_notes(notes: str, category: str, source_file: str) -> str:
    """Format notes to be more user-friendly and concise."""
    if not notes:
        return ""

    # Handle "ARG: VARIABLE_NAME" format
    if notes.startswith("ARG: "):
        return "Dockerfile build argument"

    # Handle "From install script: VARIABLE_NAME" format
    if notes.startswith("From install script:"):
        return "From installation script"

    # Handle "ENV: VARIABLE_NAME" format
    if notes.startswith("ENV: "):
        return "Dockerfile environment variable"

    # Handle Git dependency notes
    if notes.startswith("Git dependency:"):
        return "Git repository dependency"

    # Handle "Git-based pip install from ..."
    if notes.startswith("Git-based pip install from"):
        org_repo = notes.replace("Git-based pip install from ", "")
        return f"Installed from Git ({org_repo})"

    # Helm dependencies
    if "Helm dependency from" in notes:
        # Extract just the source type
        if "oci://" in notes:
            return "Helm chart from OCI registry"
        elif "file://" in notes:
            return "Local Helm chart"
        else:
            return "Helm chart dependency"

    # Binary download notes
    if "Binary download from" in notes:
        return "Binary installed from remote URL"

    # Python optional dependencies
    if "Python optional dependency" in notes:
        group = notes.split("(")[-1].replace(")", "").strip()
        return f"Optional dependency ({group})"

    # Default: return as-is
    return notes


def normalize_dependency_name(name: str, category: str = "") -> str:
    """
    Normalize dependency names to detect the same dependency referred to differently.

    Examples:
        - torch, pytorch, PyTorch -> pytorch
        - tensorflow, TensorFlow -> tensorflow
        - numpy, NumPy -> numpy

    Note: This is intentionally conservative to avoid false positives.
    Only normalizes well-known dependencies with common naming variations.

    For Go modules, we don't normalize at all since the full import path
    is significant (e.g., github.com/pkg/errors vs k8s.io/errors are different).
    """
    # For Go dependencies, use the full name without normalization
    # Go module paths are unique identifiers and should not be normalized
    if category == "Go Dependency" or category == "Go Module":
        return name.strip()

    # Convert to lowercase for comparison
    name_lower = name.lower()

    # Special handling for PyTorch-related packages that should NOT be normalized to pytorch
    # e.g., "pytorch triton" is the Triton compiler, not PyTorch itself
    if any(exc in name_lower for exc in PYTORCH_EXCEPTIONS):
        return name_lower  # Don't normalize these

    # Check if name matches any normalization rules (exact or starts with)
    for key, normalized in NORMALIZATIONS.items():
        if name_lower == key or name_lower.startswith(key + " "):
            return normalized

    # Default: return the lowercase name unchanged
    # This avoids false positives from overly broad matching
    return name_lower.strip()


def normalize_version_for_comparison(version: str) -> str:
    """
    Normalize version string for comparison by removing pinning operators.

    This allows us to detect true version differences while ignoring
    differences in how versions are pinned.

    Examples:
        - "==0.115.12" -> "0.115.12"
        - ">=0.115.0" -> "0.115.0"
        - ">=32.0.1,<33.0.0" -> "32.0.1"
        - "<=0.6.0" -> "0.6.0"
        - "2.7.1+cu128" -> "2.7.1+cu128" (unchanged)
    """
    # Remove common Python version operators
    # This regex captures: ==, >=, <=, ~=, !=, <, >, and extracts the version
    version = version.strip()

    # Handle compound version specs like ">=32.0.1,<33.0.0" - take the first version
    if "," in version:
        version = version.split(",")[0].strip()

    # Remove operators
    version = re.sub(r"^(==|>=|<=|~=|!=|<|>)\s*", "", version)

    return version.strip()