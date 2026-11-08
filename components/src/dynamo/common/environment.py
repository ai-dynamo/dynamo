# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural environment composition matching the operator's semanticenv package.

No runtime values are expanded here. Kubernetes replays the resulting ordered list.
"""

import heapq
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

_ORIGIN = "nvidia.com/dynamo-operator-origin-version"
_VERSION = re.compile(
    r"v?([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def ordered_environment_enabled(annotations: Mapping[str, Any]) -> bool:
    """Use the operator's >=1.8.0 origin gate, including semver prereleases."""

    match = _VERSION.fullmatch(str(annotations.get(_ORIGIN, "")))
    if match is None:
        return False
    parts = match.groups()
    # Go's semver parser stores each core number in a uint64.
    if any(len(part.lstrip("0")) > 20 for part in parts[:3] if part):
        return False
    core = tuple(int((part or "").lstrip("0") or "0") for part in parts[:3])
    if any(part > 2**64 - 1 for part in core):
        return False
    prerelease = parts[3]
    if prerelease and any(
        len(part) > 1 and part.isdigit() and part.startswith("0")
        for part in prerelease.split(".")
    ):
        return False
    return core > (1, 8, 0) or (core == (1, 8, 0) and prerelease is None)


def _references(value: str) -> Iterator[str]:
    index = 0
    while index + 1 < len(value):
        if value[index] != "$":
            index += 1
        elif value[index + 1] == "$":
            index += 2
        elif value[index + 1] == "(":
            end = value.find(")", index + 2)
            if end < 0:
                return
            yield value[index + 2 : end]
            index = end + 1
        else:
            index += 1


def compose_environment(
    *layers: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Interleave general-to-specific lists without changing within-list order.

    Later sources replace whole entries, but duplicates within a source remain.
    Only cross-source references impose additional order. Missing and same-source
    references keep Kubernetes semantics, including forward and self references.
    """

    winners = {
        entry["name"]: source for source, layer in enumerate(layers) for entry in layer
    }
    entries: list[tuple[int, int, Mapping[str, Any]]] = []
    definitions: dict[str, int] = {}
    successors: list[list[int]] = []
    indegrees: list[int] = []

    def edge(start: int, end: int) -> None:
        successors[start].append(end)
        indegrees[end] += 1

    for source, layer in enumerate(layers):
        previous = None
        for position, entry in enumerate(layer):
            if winners[entry["name"]] != source:
                continue
            current = len(entries)
            entries.append((source, position, entry))
            definitions[entry["name"]] = current
            successors.append([])
            indegrees.append(0)
            if previous is not None:
                edge(previous, current)
            previous = current
    for index, (source, _, entry) in enumerate(entries):
        for name in _references(str(entry.get("value") or "")):
            dependency = definitions.get(name)
            if dependency is not None and entries[dependency][0] != source:
                edge(dependency, index)

    ready = [index for index, degree in enumerate(indegrees) if degree == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        index = heapq.heappop(ready)
        result.append(entries[index][2])
        for successor in successors[index]:
            indegrees[successor] -= 1
            if indegrees[successor] == 0:
                heapq.heappush(ready, successor)
    if len(result) != len(entries):
        blocked = ", ".join(
            f"layer[{source}].env[{position}] ({entry['name']})"
            for index, (source, position, entry) in enumerate(entries)
            if indegrees[index] > 0
        )
        raise ValueError(f"environment dependency/order conflict involving {blocked}")
    return result
