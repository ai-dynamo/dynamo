# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-consistency test for the DeepSeek-V4 recipe families.

Each family ships one shared ``model-cache/model-download.yaml`` Job and several
``deploy.yaml`` variants. The workers mount the resulting PVC read-only with
``HF_HUB_OFFLINE=1``, so a checkpoint the Job never downloaded is a hard startup
failure rather than a slow first launch. This test asserts the cross-file
invariant that makes that impossible to introduce silently: every checkpoint
repo id a ``deploy.yaml`` asks for must be obtainable from the family's download
Job -- as its default ``MODEL_NAME``, as a repo id in an active ``hf download``
line, or as a documented override value named in a comment.

Scope is the two DeepSeek-V4 families deliberately. Other recipe families use
different download mechanisms (inline ``snapshot_download`` calls, for example)
and some carry pre-existing mismatches of their own; widening this test is a
separate change.
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import pytest
import yaml

# Reads files only, so it is cheap enough to gate every merge.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

REPO_ROOT = Path(__file__).resolve().parents[2]

RECIPE_FAMILIES = [
    "recipes/deepseek-v4/deepseek-v4-pro",
    "recipes/deepseek-v4/deepseek-v4-flash",
]

# A HuggingFace repo id: exactly one "/" separating an org from a model name.
# Anchored, so shell fragments and absolute paths do not match.
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")

# Repo ids written inside a comment, e.g.: Override with "nvidia/Foo-NVFP4".
# Quoting is required so a commented-out shell command does not read as a
# documented override.
QUOTED_REPO_ID_RE = re.compile(r"[\"']([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)[\"']")

# Flags whose argument names the checkpoint, in either vLLM or SGLang spelling.
MODEL_FLAGS = ("--model", "--model-path", "--served-model-name")

# Env vars the deploy.yaml files interpolate into those flags.
MODEL_ENV_VARS = ("MODEL_PATH", "SERVED_MODEL_NAME")


def _iter_nodes(node: Any) -> Iterable[Any]:
    """Yield every mapping and sequence in a loaded YAML document."""
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nodes(item)


def _load_documents(path: Path) -> List[Any]:
    return [doc for doc in yaml.safe_load_all(path.read_text()) if doc is not None]


def _repo_ids_from_flags(text: str) -> Set[str]:
    """Extract checkpoint ids from ``--model <id>`` style flags in a script."""
    found: Set[str] = set()
    for line in text.splitlines():
        tokens = line.replace("\\", " ").split()
        for flag, value in zip(tokens, tokens[1:]):
            if flag.split("=")[0] in MODEL_FLAGS:
                candidate = value.strip("\"'")
                if REPO_ID_RE.match(candidate):
                    found.add(candidate)
    return found


def _repo_ids_from_arg_list(items: List[Any]) -> Set[str]:
    """Extract checkpoint ids from an args list of alternating flag/value items."""
    found: Set[str] = set()
    strings = [item for item in items if isinstance(item, str)]
    for flag, value in zip(strings, strings[1:]):
        if flag in MODEL_FLAGS and REPO_ID_RE.match(value.strip("\"'")):
            found.add(value.strip("\"'"))
    return found


def _repo_ids_from_env(node: Dict[str, Any], names: Iterable[str]) -> Set[str]:
    """Extract checkpoint ids from ``env:`` entries with the given names."""
    found: Set[str] = set()
    env = node.get("env")
    if not isinstance(env, list):
        return found
    wanted = set(names)
    for entry in env:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") in wanted and isinstance(entry.get("value"), str):
            value = entry["value"].strip()
            if REPO_ID_RE.match(value):
                found.add(value)
    return found


def _referenced_checkpoints(deploy_path: Path) -> Set[str]:
    """Every checkpoint repo id a deploy manifest asks a worker to serve."""
    found: Set[str] = set()
    for document in _load_documents(deploy_path):
        for node in _iter_nodes(document):
            if isinstance(node, str):
                if any(flag in node for flag in MODEL_FLAGS):
                    found |= _repo_ids_from_flags(node)
            elif isinstance(node, list):
                found |= _repo_ids_from_arg_list(node)
            elif isinstance(node, dict):
                found |= _repo_ids_from_env(node, MODEL_ENV_VARS)
    return found


def _obtainable_checkpoints(download_path: Path) -> Set[str]:
    """Every checkpoint the download Job can be made to fetch.

    The default ``MODEL_NAME`` and any repo id passed to an active ``hf
    download`` come from the parsed document; documented overrides come from the
    raw text, since YAML parsing discards comments.
    """
    found: Set[str] = set()

    for document in _load_documents(download_path):
        for node in _iter_nodes(document):
            if isinstance(node, dict):
                found |= _repo_ids_from_env(node, ["MODEL_NAME"])

    for raw_line in download_path.read_text().splitlines():
        code, _, comment = raw_line.partition("#")
        if "hf download" in code:
            for token in code.split():
                candidate = token.strip("\"'")
                if REPO_ID_RE.match(candidate):
                    found.add(candidate)
        if comment:
            found |= set(QUOTED_REPO_ID_RE.findall(comment))

    return found


@pytest.mark.parametrize("family", RECIPE_FAMILIES)
def test_every_deployed_checkpoint_is_downloadable(family: str) -> None:
    family_dir = REPO_ROOT / family
    download_path = family_dir / "model-cache" / "model-download.yaml"
    assert download_path.is_file(), f"missing download Job: {download_path}"

    deploy_paths = sorted(family_dir.rglob("deploy.yaml"))
    assert deploy_paths, f"no deploy.yaml found under {family_dir}"

    obtainable = _obtainable_checkpoints(download_path)
    assert obtainable, f"no checkpoint repo id found in {download_path}"

    problems = []
    for deploy_path in deploy_paths:
        referenced = _referenced_checkpoints(deploy_path)
        assert referenced, f"no checkpoint repo id found in {deploy_path}"
        for checkpoint in sorted(referenced - obtainable):
            problems.append(f"{deploy_path.relative_to(REPO_ROOT)} serves {checkpoint}")

    assert not problems, (
        f"{download_path.relative_to(REPO_ROOT)} can only supply "
        f"{sorted(obtainable)}, but these variants ask for a checkpoint it "
        f"neither downloads by default nor documents as an override:\n  "
        + "\n  ".join(problems)
    )
