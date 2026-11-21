# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for linting shipped Kubernetes/DGD manifests without a cluster.

Only the standard library and ``yaml`` are imported, so tests built on these
helpers run on any CPU runner. Containers are found by shape rather than by a
fixed DGD path, so the same code reads v1alpha1
(``spec.services.*.extraPodSpec.mainContainer``), v1beta1
(``spec.components[].podTemplate.spec.containers[]``) and plain pod specs.
"""

import shlex
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


def iter_yaml_documents(
    root: Path, globs: Iterable[str] = ("*.yaml", "*.yml")
) -> Iterator[tuple[Path, Any]]:
    """Yield ``(path, document)`` for every non-empty YAML document under ``root``."""
    paths = sorted({p for pattern in globs for p in Path(root).rglob(pattern)})
    for path in paths:
        with path.open() as f:
            for doc in yaml.safe_load_all(f):
                if doc is not None:
                    yield path, doc


def iter_containers(node: Any, location: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(location, mapping)`` for every mapping with a list or string
    ``command`` or ``args``.

    ``location`` is the dotted key path to the mapping, with list items named
    by their ``name`` field when they have one, so a failure can point at the
    service and container.
    """
    if isinstance(node, Mapping):
        if any(isinstance(node.get(key), (list, str)) for key in ("command", "args")):
            yield location, node
        for key, value in node.items():
            yield from iter_containers(
                value, f"{location}.{key}" if location else str(key)
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            name = item.get("name") if isinstance(item, Mapping) else None
            yield from iter_containers(item, f"{location}[{name or index}]")


def split_shell(text: str) -> list[str]:
    """Split a shell command into tokens, falling back to whitespace on bad quoting."""
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def container_argv(container: Mapping) -> list[str]:
    """Return the combined ``command`` + ``args`` tokens of a container.

    A string ``command``/``args``, and any list item that follows a short
    option group containing ``c`` (the ``sh -c``, ``bash -lc`` and ``sh -ec``
    ``"<script>"`` forms, whether the script sits in ``command`` or ``args``),
    is shell-split so its flags are visible as separate tokens.
    """
    argv: list[str] = []
    for key in ("command", "args"):
        value = container.get(key)
        if isinstance(value, str):
            argv.extend(split_shell(value))
        elif isinstance(value, list):
            for item in value:
                item = str(item)
                prev = argv[-1] if argv else ""
                if prev.startswith("-") and not prev.startswith("--") and "c" in prev:
                    argv.extend(split_shell(item))
                else:
                    argv.append(item)
    return argv


def has_flag(argv: list[str], flag: str, value: str | None = None) -> bool:
    """Return whether ``argv`` passes ``flag`` (with ``value`` when given), as
    either ``--flag value`` or ``--flag=value``."""
    for index, token in enumerate(argv):
        if token == flag:
            if value is None or argv[index + 1 : index + 2] == [value]:
                return True
        elif token.startswith(f"{flag}="):
            if value is None or token[len(flag) + 1 :] == value:
                return True
    return False
