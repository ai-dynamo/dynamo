# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render checked-in recipe manifests from Kustomize overlays."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

SPDX_HEADER = "\n".join(
    [
        "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
    ]
)
REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_GENERATOR = REPO_ROOT / "scripts/generate_kustomize_openapi.py"
PathPart: TypeAlias = str | int
YamlPath: TypeAlias = tuple[PathPart, ...]
FIRST_TOP_LEVEL_KEYS = ("apiVersion", "kind", "metadata")


@dataclass(frozen=True)
class DocumentId:
    api_version: str
    kind: str
    name: str


@dataclass
class CommentBlock:
    doc_id: DocumentId
    path: YamlPath
    lines: list[str]


@dataclass
class DocumentScan:
    doc_id: DocumentId | None
    comments: list[CommentBlock]
    targets: dict[YamlPath, int]


@dataclass
class TopLevelBlock:
    key: str | None
    lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when rendered outputs do not match the checked-in files",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="optional changed paths; matching recipe roots are selected automatically",
    )
    return parser.parse_args()


def all_recipe_roots() -> list[Path]:
    base_kustomizations = list(
        REPO_ROOT.glob("recipes/**/kustomize/base/kustomization.yaml")
    )
    return sorted({path.parent.parent.parent for path in base_kustomizations})


def select_recipe_roots(paths: list[str]) -> list[Path]:
    recipe_roots = all_recipe_roots()
    if not paths:
        return recipe_roots

    selected: set[Path] = set()
    changed_paths = [Path(path) for path in paths]
    for changed_path in changed_paths:
        abs_changed = (REPO_ROOT / changed_path).resolve()
        try:
            rel_changed = abs_changed.relative_to(REPO_ROOT)
        except ValueError:
            continue

        if rel_changed.parts[:1] != ("recipes",):
            return recipe_roots

        for recipe_root in recipe_roots:
            try:
                abs_changed.relative_to(recipe_root.resolve())
            except ValueError:
                continue
            selected.add(recipe_root)

    return sorted(selected)


def rendered_overlay_paths(recipe_root: Path) -> dict[Path, Path]:
    overlays_dir = recipe_root / "kustomize" / "overlays"
    if not overlays_dir.exists():
        return {}

    outputs = {}
    for overlay_dir in sorted(overlays_dir.iterdir()):
        if not overlay_dir.is_dir() or overlay_dir.name.startswith("_"):
            continue
        kustomization = overlay_dir / "kustomization.yaml"
        if kustomization.exists():
            outputs[overlay_dir] = recipe_root / f"deploy-{overlay_dir.name}.yaml"
    return outputs


def kustomize_command() -> list[str]:
    if shutil.which("kustomize"):
        return ["kustomize", "build"]
    if shutil.which("kubectl"):
        return ["kubectl", "kustomize"]
    raise RuntimeError("install kustomize or kubectl to render recipe overlays")


def generate_kustomize_openapi(*, check: bool) -> None:
    command = [sys.executable, str(OPENAPI_GENERATOR)]
    if check:
        command.append("--check")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def starts_block_scalar(stripped_line: str) -> bool:
    if not stripped_line or stripped_line.startswith("#"):
        return False
    value = stripped_line.split("#", 1)[0].rstrip().rsplit(maxsplit=1)[-1]
    return value.startswith("|") or value.startswith(">")


def line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_key_value(content: str) -> tuple[str, str] | None:
    if ":" not in content:
        return None

    key, value = content.split(":", 1)
    key = key.strip().strip("\"'")
    if not key:
        return None
    return key, value.strip()


def strip_inline_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


def scalar_value(value: str) -> str:
    value = strip_inline_comment(value)
    return value.strip("\"'")


def record_scalar(values: dict[YamlPath, str], path: YamlPath, value: str) -> None:
    value = scalar_value(value)
    if value:
        values[path] = value


def top_level_key(line: str) -> str | None:
    if line_indent(line) != 0:
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("- "):
        return None
    parsed = parse_key_value(stripped)
    return parsed[0] if parsed else None


def split_rendered_documents(rendered: str) -> list[list[str]]:
    documents: list[list[str]] = [[]]
    for line in rendered.splitlines():
        if line.strip() == "---":
            documents.append([])
            continue
        documents[-1].append(line)
    return documents


def reorder_top_level_blocks(document: list[str]) -> list[str]:
    blocks: list[TopLevelBlock] = []
    current = TopLevelBlock(key=None, lines=[])

    for line in document:
        key = top_level_key(line)
        if key is not None:
            if current.lines:
                blocks.append(current)
            current = TopLevelBlock(key=key, lines=[line])
            continue
        current.lines.append(line)

    if current.lines:
        blocks.append(current)

    ordered: list[TopLevelBlock] = []
    for key in FIRST_TOP_LEVEL_KEYS:
        ordered.extend(block for block in blocks if block.key == key)
    ordered.extend(block for block in blocks if block.key not in FIRST_TOP_LEVEL_KEYS)
    return [line for block in ordered for line in block.lines]


def reorder_rendered_top_level_fields(rendered: str) -> str:
    """Keep Kubernetes object identity fields first without changing nested YAML."""

    documents = split_rendered_documents(rendered)
    rendered_documents = [
        "\n".join(reorder_top_level_blocks(document)).rstrip() for document in documents
    ]
    return "\n---\n".join(rendered_documents).rstrip() + "\n"


def scan_yaml(text: str) -> list[DocumentScan]:
    """Map YAML nodes to line numbers and full-line comments to following nodes.

    Kustomize renders the object graph correctly but drops comments. This scanner
    does not interpret Kubernetes patch semantics; it only records enough path
    information to put source comments back next to the same rendered fields.
    """

    docs: list[DocumentScan] = []
    comments: list[tuple[YamlPath, list[str]]] = []
    targets: dict[YamlPath, int] = {}
    values: dict[YamlPath, str] = {}
    stack: list[tuple[int, YamlPath]] = []
    sequence_indexes: dict[tuple[YamlPath, int], int] = {}
    pending_comments: list[str] = []
    block_scalar_indent: int | None = None

    def finish_document() -> None:
        nonlocal comments, targets, values, stack, sequence_indexes, pending_comments
        api_version = values.get(("apiVersion",))
        kind = values.get(("kind",))
        name = values.get(("metadata", "name"))
        doc_id = (
            DocumentId(api_version, kind, name)
            if api_version and kind and name
            else None
        )
        doc_comments = (
            [
                CommentBlock(doc_id=doc_id, path=path, lines=lines)
                for path, lines in comments
            ]
            if doc_id
            else []
        )
        if comments or targets or values:
            docs.append(
                DocumentScan(doc_id=doc_id, comments=doc_comments, targets=targets)
            )
        comments = []
        targets = {}
        values = {}
        stack = []
        sequence_indexes = {}
        pending_comments = []

    for line_number, line in enumerate(text.splitlines()):
        stripped = line.strip()
        indent = line_indent(line)

        if stripped == "---":
            finish_document()
            block_scalar_indent = None
            continue

        if block_scalar_indent is not None:
            if not stripped:
                continue
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None

        if starts_block_scalar(stripped):
            block_scalar_indent = indent
        if not stripped:
            continue

        if stripped.startswith("#"):
            if not stripped.startswith("# SPDX-"):
                pending_comments.append(stripped)
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent_path = stack[-1][1] if stack else ()
        line_paths: list[YamlPath] = []

        if stripped.startswith("- "):
            sequence_key = (parent_path, indent)
            item_index = sequence_indexes.get(sequence_key, 0)
            sequence_indexes[sequence_key] = item_index + 1
            item_path = (*parent_path, item_index)
            line_paths.append(item_path)
            content = stripped[2:].strip()
            parsed = parse_key_value(content)
            if parsed:
                key, value = parsed
                key_path = (*item_path, key)
                line_paths.append(key_path)
                record_scalar(values, key_path, value)
                if not strip_inline_comment(value) or starts_block_scalar(content):
                    stack.append((indent, key_path))
                else:
                    stack.append((indent, item_path))
            else:
                stack.append((indent, item_path))
        else:
            parsed = parse_key_value(stripped)
            if not parsed:
                continue
            key, value = parsed
            key_path = (*parent_path, key)
            line_paths.append(key_path)
            record_scalar(values, key_path, value)
            if not strip_inline_comment(value) or starts_block_scalar(stripped):
                stack.append((indent, key_path))

        for path in line_paths:
            targets.setdefault(path, line_number)

        if pending_comments and line_paths:
            comments.append((line_paths[0], pending_comments))
            pending_comments = []

    finish_document()
    return docs


def comment_blocks_from_sources(source_paths: list[Path]) -> list[CommentBlock]:
    blocks: list[CommentBlock] = []
    for path in source_paths:
        for doc in scan_yaml(path.read_text(encoding="utf-8")):
            blocks.extend(doc.comments)
    return blocks


def source_comment_paths(overlay_dir: Path) -> list[Path]:
    recipe_root = overlay_dir.parent.parent.parent
    base_dir = recipe_root / "kustomize" / "base"
    base_paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in base_dir.rglob(pattern)
        if path.name != "kustomization.yaml"
    )
    overlay_paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in overlay_dir.rglob(pattern)
        if path.name != "kustomization.yaml"
    )
    return base_paths + overlay_paths


def rendered_target_lines(rendered: str) -> dict[tuple[DocumentId, YamlPath], int]:
    targets = {}
    for doc in scan_yaml(rendered):
        if doc.doc_id is None:
            continue
        for path, line_number in doc.targets.items():
            targets[(doc.doc_id, path)] = line_number
    return targets


def restore_source_comments(rendered: str, overlay_dir: Path) -> str:
    """Re-insert source comments before matching rendered object fields."""

    lines = rendered.splitlines()
    insertions: dict[int, list[str]] = {}
    targets = rendered_target_lines(rendered)
    for block in comment_blocks_from_sources(source_comment_paths(overlay_dir)):
        line_number = targets.get((block.doc_id, block.path))
        if line_number is None:
            continue
        indent = " " * line_indent(lines[line_number])
        insertions.setdefault(line_number, []).extend(
            f"{indent}{line}" for line in block.lines
        )

    restored = []
    for line_number, line in enumerate(lines):
        restored.extend(insertions.get(line_number, []))
        restored.append(line)
    return "\n".join(restored) + "\n"


def generated_header(overlay_dir: Path) -> str:
    recipe_root = overlay_dir.parent.parent.parent
    output_path = recipe_root / f"deploy-{overlay_dir.name}.yaml"
    command = (
        "python3 scripts/render_recipe_kustomize.py "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    header_lines = [
        f"# Generated by: {command}",
        "# You may edit this manifest before applying it.",
        "# To change the checked-in recipe, edit the Kustomize source files and re-render it.",
    ]
    return "\n".join(header_lines)


def render_overlay(overlay_dir: Path) -> str:
    result = subprocess.run(
        [*kustomize_command(), str(overlay_dir)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    rendered = result.stdout
    if not rendered.endswith("\n"):
        rendered += "\n"
    rendered = reorder_rendered_top_level_fields(rendered)
    rendered = restore_source_comments(rendered, overlay_dir)
    return f"{SPDX_HEADER}\n\n{generated_header(overlay_dir)}\n\n{rendered}"


def write_or_check(output_path: Path, rendered: str, *, check: bool) -> list[Path]:
    if check:
        current = (
            output_path.read_text(encoding="utf-8") if output_path.exists() else None
        )
        return [output_path] if current != rendered else []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return []


def render_recipe_root(recipe_root: Path, *, check: bool) -> list[Path]:
    stale_outputs: list[Path] = []
    overlay_outputs = rendered_overlay_paths(recipe_root)
    expected_outputs = set(overlay_outputs.values())

    for overlay_dir, output_path in overlay_outputs.items():
        stale_outputs.extend(
            write_or_check(output_path, render_overlay(overlay_dir), check=check)
        )

    base_output = recipe_root / "deploy.yaml"
    if base_output.exists():
        stale_outputs.append(base_output)

    stale_outputs.extend(
        output_path
        for output_path in sorted(recipe_root.glob("deploy-*.yaml"))
        if output_path not in expected_outputs
    )

    return stale_outputs


def main() -> int:
    args = parse_args()
    recipe_roots = select_recipe_roots(args.paths)
    if not recipe_roots:
        return 0

    stale_outputs: list[Path] = []
    try:
        generate_kustomize_openapi(check=args.check)
        for recipe_root in recipe_roots:
            stale_outputs.extend(render_recipe_root(recipe_root, check=args.check))
    except (OSError, RuntimeError) as exc:
        print(f"render_recipe_kustomize.py: {exc}", file=sys.stderr)
        return 1

    if stale_outputs:
        print("Rendered recipe manifests are stale:", file=sys.stderr)
        for output in stale_outputs:
            print(f"  {output.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "Run: python3 scripts/render_recipe_kustomize.py, then remove any files that are no longer generated.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
