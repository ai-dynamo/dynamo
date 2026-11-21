#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect examples with no website owner and stale ``examples/`` links.

The examples index page
(``docs/fern/pages/recipes/examples/overview.mdx``) is the curated website
surface for the user-facing tree under ``examples/``. This checker keeps that
surface honest without a separate machine-readable schema:

  ORPHAN  every user-facing example (a directory that ships a README or a
          runnable entrypoint, plus the files in EXTRA_USER_FACING) must be
          referenced from the Recipes examples index, directly or via one of
          its children, unless it is explicitly dispositioned in
          NON_USER_FACING.
  STALE   every ``examples/...`` path referenced from the docs site must exist
          on disk. Broken links on the examples index page and the reference
          examples page are errors; elsewhere they are reported as warnings so
          this check does not gate on pre-existing debt it did not introduce.

Usage:
  python3 docs/fern/scripts/check_examples_index.py
  python3 docs/fern/scripts/check_examples_index.py --strict   # warnings fail too

Exit code: 1 if any error-severity finding, else 0.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

EXAMPLES_DIR = "examples"
DOCS_FERN_DIR = os.path.join("docs", "fern")

# Pages this check owns: a broken link here is an error, not pre-existing debt.
OWNED_FILES = (
    os.path.join("docs", "fern", "pages", "recipes", "examples", "overview.mdx"),
    os.path.join("docs", "fern", "pages", "reference", "general", "examples.md"),
)

# A directory is a user-facing candidate when it directly contains one of these.
# Detection is content-based rather than README-only so that a new example added
# without a README is still caught: a missing example must be dispositioned in
# NON_USER_FACING, not silently ignored.
USER_FACING_FILENAMES = ("README.md",)
USER_FACING_SUFFIXES = (".sh", ".py")

# User-facing example files that live inside a shared directory rather than in a
# dedicated example directory, so per-directory content detection cannot see
# them. Directories themselves are detected automatically and need no entry here.
EXTRA_USER_FACING = (
    "examples/common/gpu_utils.md",
    "examples/common/lora.md",
    "examples/router/policy-class-queues.yaml",
)

# Paths that intentionally do not get a curated website entry, with the reason.
# They are still checked for existence so the disposition does not go stale.
NON_USER_FACING = {
    "examples/backends/sample": "test-only CPU smoke backends",
    "examples/backends/sample/launch": "test-only CPU smoke launch scripts",
    "examples/backends/mocker": "GPU-free mocker workers used by CI",
    "examples/backends/sglang": "test helpers for the SGLang backend",
    "examples/backends/vllm/omni": "internal dev/qualification overlay",
    "examples/backends/vllm/deploy/v1beta1": "obsolete legacy manifest path",
    "examples/backends/vllm/deploy/gaie": "supporting Gateway API Inference Extension manifest",
    "examples/backends/vllm/launch/stage_configs": "supporting stage configuration",
    "examples/backends/trtllm/templates": "supporting chat template asset",
    "examples/backends/tokenspeed/tests": "test-only",
    "examples/nemotron_speech_cascaded_pipeline/container": "supporting adapter image build",
    "examples/nemotron_speech_cascaded_pipeline/nemotron_speech": "internal adapter code",
    "examples/nemotron_speech_cascaded_pipeline/nemotron_speech/asr": "internal adapter code",
    "examples/nemotron_speech_cascaded_pipeline/nemotron_speech/tts": "internal adapter code",
    "examples/nemotron_speech_cascaded_pipeline/tests": "test-only",
    "examples/custom_encoder/launch": "supporting launch scripts for the encoder example",
    "examples/deployments/EKS/manifests": "supporting EKS manifests",
    "examples/deployments/EKS/templates": "supporting eksctl template",
    "examples/deployments/GKE/vllm": "supporting GKE manifests",
    "examples/deployments/GKE/sglang": "supporting GKE manifests",
    "examples/deployments/dgdr/generated-dgd-override.yaml": "generated override",
    "examples/router/custom-policy-example/catalog": "supporting Rust policy crate",
    "examples/router/custom-policy-example/epp": "supporting EPP binary crate",
    "examples/router/custom-policy-example/simple-filter-score-pick": "supporting sample crate",
    "examples/router/custom-policy-example/disagg-filter-score-pick": "supporting sample crate",
    "examples/router/custom-policy-example/simple-stacked-score-pick": "supporting sample crate",
    "examples/router/custom-policy-example/AGENTS.md": "internal agent instructions",
    "examples/router/custom-policy-example/CLAUDE.md": "internal agent instructions",
    "examples/__init__.py": "Python package marker",
}

# Match the top-level `examples/` tree only: a bare path segment, or the
# `.../main/examples/...` (or master) form used by GitHub tree/blob URLs.
# This excludes docs paths such as `recipes/examples/overview.mdx`.
REF_RE = re.compile(r"(?:(?<![\w./-])|/(?:main|master)/)examples/[A-Za-z0-9_./-]+")
TRAILING = ").,;:\"'`"


def normalize_ref(raw: str) -> str:
    """Trim a GitHub-URL prefix, trailing punctuation, and trailing slashes."""
    index = raw.find("examples/")
    ref = raw[index:] if index >= 0 else raw
    while ref and ref[-1] in TRAILING:
        ref = ref[:-1]
    while ref.endswith("/"):
        ref = ref[:-1]
    return ref


def has_user_facing_content(filenames: list[str]) -> bool:
    """True when a directory directly holds a README or a runnable entrypoint."""
    if any(name in USER_FACING_FILENAMES for name in filenames):
        return True
    return any(name.endswith(USER_FACING_SUFFIXES) for name in filenames)


def detected_user_facing() -> dict[str, str]:
    """Path -> why it is user-facing, for every path the index must cover.

    Directories are detected by content, not by README presence alone, so a new
    example without a README is still surfaced as an orphan unless it is
    dispositioned in NON_USER_FACING.
    """
    found: dict[str, str] = {}
    examples_root = os.path.join(REPO_ROOT, EXAMPLES_DIR)
    for dirpath, dirnames, filenames in os.walk(examples_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, REPO_ROOT).replace(os.sep, "/")
        if rel == EXAMPLES_DIR:
            continue
        if "README.md" in filenames:
            found[rel] = "README"
        elif has_user_facing_content(filenames):
            found[rel] = "runnable example entrypoint"
    for rel in EXTRA_USER_FACING:
        found[rel] = "user-facing file"
    return found


def collect_doc_refs() -> dict[str, set[str]]:
    """Path -> set of docs files that reference it."""
    refs: dict[str, set[str]] = {}
    docs_root = os.path.join(REPO_ROOT, DOCS_FERN_DIR)
    for dirpath, dirnames, filenames in os.walk(docs_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if not name.endswith((".md", ".mdx")):
                continue
            abs_path = os.path.join(dirpath, name)
            rel_file = os.path.relpath(abs_path, REPO_ROOT).replace(os.sep, "/")
            with open(abs_path, encoding="utf-8") as handle:
                text = handle.read()
            for raw in REF_RE.findall(text):
                ref = normalize_ref(raw)
                if ref and ref != EXAMPLES_DIR and "/" in ref[len(EXAMPLES_DIR) :]:
                    refs.setdefault(ref, set()).add(rel_file)
    return refs


def main() -> int:
    """Run the orphan and stale-link checks; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings (stale links outside owned files) as errors",
    )
    args = parser.parse_args()

    all_detected = detected_user_facing()
    dispositioned = {p for p in all_detected if p in NON_USER_FACING}
    required = {p: why for p, why in all_detected.items() if p not in dispositioned}
    refs = collect_doc_refs()

    errors: list[str] = []
    warnings: list[str] = []

    # Coverage is judged against the curated Recipes index only: a link from an
    # unrelated page must not satisfy the orphan check. The site-wide ``refs``
    # scan is still used below to detect stale links everywhere.
    index_file = OWNED_FILES[0]
    index_refs = {ref for ref, where in refs.items() if index_file in where}
    for path, why in sorted(required.items()):
        covered = any(ref == path or ref.startswith(path + "/") for ref in index_refs)
        if not covered:
            errors.append(
                f"ORPHAN  {path} ({why}) is not referenced from the Recipes "
                "examples index"
            )

    for path, where in sorted(refs.items()):
        if os.path.exists(os.path.join(REPO_ROOT, path)):
            continue
        message = (
            f"STALE   {path} does not exist (referenced by {', '.join(sorted(where))})"
        )
        if any(w in OWNED_FILES for w in where):
            errors.append(message)
        else:
            warnings.append(message)

    for path in sorted(NON_USER_FACING):
        if not os.path.exists(os.path.join(REPO_ROOT, path)):
            warnings.append(f"STALE   dispositioned path {path} does not exist")

    print(
        f"examples index: {len(required)} required, "
        f"{len(dispositioned)} dispositioned, {len(refs)} referenced paths"
    )
    for warning in warnings:
        print(f"warn:   {warning}")
    for error in errors:
        print(f"error:  {error}")

    if errors or (args.strict and warnings):
        print(f"FAILED ({len(errors)} errors, {len(warnings)} warnings)")
        return 1
    print(f"OK ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
