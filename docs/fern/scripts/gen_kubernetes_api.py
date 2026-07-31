#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the Dynamo Kubernetes API reference source-of-truth outputs.

Thin CLI/orchestrator that composes :mod:`kubernetes_api_discovery` (the
deterministic Markdown parser) with :mod:`kubernetes_api_rendering` (the MDX
serializer). Every run parses
``docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md``
once and emits ``docs/fern/pages/reference/kubernetes-api/full-api-reference.mdx``:
one MDX page built from Fern's own ``<CardGroup>``, ``<Accordion>``,
``<ParamField>``, and ``<Badge>`` components, followed by the twelve
operator-default subsections as demoted Markdown headings.

Usage (from any cwd; paths resolve relative to this file)::

    python3 gen_kubernetes_api.py            # write / refresh outputs
    python3 gen_kubernetes_api.py --check    # exit 1 if any output stale

Isolated invocation (bypasses the repo's Python resolution, which is
unrelated to this generator)::

    uv run --no-project --python 3.13 python3 \\
        docs/fern/scripts/gen_kubernetes_api.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kubernetes_api_discovery import KubernetesReference, parse_reference
from kubernetes_api_rendering import render_mdx

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FERN_ROOT = SCRIPT_DIR.parent


def _source_path(fern_root: Path) -> Path:
    return (
        fern_root
        / "pages"
        / "reference"
        / "kubernetes-api"
        / "additional-resources"
        / "api-reference-k8s.md"
    )


def _mdx_shell_path(fern_root: Path) -> Path:
    return (
        fern_root / "pages" / "reference" / "kubernetes-api" / "full-api-reference.mdx"
    )


def _load_reference(fern_root: Path) -> KubernetesReference:
    """Read the upstream Markdown source and parse it into the model."""
    return parse_reference(_source_path(fern_root).read_text(encoding="utf-8"))


def _rendered_outputs(
    fern_root: Path, reference: KubernetesReference
) -> dict[Path, str]:
    """Compute every output path -> new text mapping in one pass."""
    return {_mdx_shell_path(fern_root): render_mdx(reference)}


def _apply_outputs(outputs: dict[Path, str], *, check: bool) -> int:
    """Write outputs (or diff them in ``--check`` mode) and report drift."""
    stale: list[str] = []
    for path, new_text in outputs.items():
        old_text = path.read_text(encoding="utf-8") if path.is_file() else None
        if new_text == old_text:
            print(f"{path.name}: unchanged")
            continue
        stale.append(path.name)
        if check:
            print(f"{path.name}: STALE (regeneration would change it)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
        print(f"{path.name}: wrote {len(new_text.encode('utf-8'))} bytes")
    if check and stale:
        print(
            f"check failed: {len(stale)} output(s) stale -- run gen_kubernetes_api.py",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; see the module docstring for the two modes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if regeneration would change any output; write nothing",
    )
    parser.add_argument(
        "--fern-root",
        type=Path,
        default=DEFAULT_FERN_ROOT,
        help="docs/fern root (override for hermetic tests; defaults to sibling)",
    )
    args = parser.parse_args(argv)
    reference = _load_reference(args.fern_root)
    outputs = _rendered_outputs(args.fern_root, reference)
    return _apply_outputs(outputs, check=args.check)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
