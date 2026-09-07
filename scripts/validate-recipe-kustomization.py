#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Validate a v1beta1 recipe Kustomization by replaying its JSON patches.

This validator intentionally supports the small, fail-closed Kustomize surface
used by the recipe scaffold: one multi-document base, ordered
Components, and JSON 6902 ``patches``.  It uses only the Python standard
library and PyYAML, and verifies its replay against Kustomize v5.8.1.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import yaml

REQUIRED_KUSTOMIZE_VERSION = "v5.8.1"
BETA_DGD_API_VERSION = "nvidia.com/v1beta1"
BETA_DGD_KIND = "DynamoGraphDeployment"
FORBIDDEN_BASE_ENV_NAMES = frozenset(
    {
        "GLOO_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
        "UCX_NET_DEVICES",
    }
)
KUSTOMIZATION_FILENAMES = ("kustomization.yaml", "kustomization.yml", "Kustomization")
SUPPORTED_OPERATIONS = frozenset({"test", "add", "replace", "remove"})
TARGET_KEYS = frozenset({"group", "version", "kind"})
ROOT_KUSTOMIZATION_KEYS = frozenset(
    {"apiVersion", "kind", "resources", "components", "patches", "sortOptions"}
)
COMPONENT_KUSTOMIZATION_KEYS = frozenset(
    {"apiVersion", "kind", "components", "patches"}
)
CANONICAL_COMPONENT_ORDER = (
    "cache-binding",
    "registry-credentials",
    "probes",
    "scheduling",
    "network-interface",
    "placement",
)
CONTAINER_COLLECTIONS = frozenset(
    {"containers", "initContainers", "ephemeralContainers"}
)
_MISSING = object()


class ValidationError(Exception):
    """A stable, user-facing recipe validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        layer: Optional[str] = None,
        op_index: Optional[int] = None,
        path: Optional[str] = None,
        expected: Any = _MISSING,
        actual: Any = _MISSING,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.layer = layer
        self.op_index = op_index
        self.path = path
        self.expected = expected
        self.actual = actual

    def diagnostic(self) -> str:
        context: List[str] = []
        if self.layer is not None:
            context.append("layer %s" % self.layer)
        if self.op_index is not None:
            context.append("op %d" % self.op_index)
        if self.path is not None:
            context.append("path %s" % self.path)
        prefix = "ERROR [%s]" % self.code
        if context:
            prefix += " " + ", ".join(context)
        details = self.message
        if self.expected is not _MISSING:
            details += "; expected=%s" % _display(self.expected)
        if self.actual is not _MISSING:
            details += "; actual=%s" % _display(self.actual)
        return prefix + ": " + details

    def __str__(self) -> str:
        return self.diagnostic()


@dataclass(frozen=True)
class _Target:
    group: str
    version: str
    kind: str


@dataclass(frozen=True)
class _RootComponent:
    index: int
    reference: str
    resolved: Path
    concern: str
    topology: str


@dataclass(frozen=True)
class _PatchLayer:
    label: str
    source: Path
    target: _Target
    operations: Tuple[Mapping[str, Any], ...]
    is_component: bool
    root_component: Optional[_RootComponent]


@dataclass(frozen=True)
class _BuildResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class _Difference:
    path: str
    expected: Any
    actual: Any


def _display(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    text = repr(value)
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _load_yaml_documents(path: Path, *, code: str = "yaml-parse") -> List[Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except OSError as error:
        raise ValidationError(code, "%s: %s" % (path, error)) from error
    except yaml.YAMLError as error:
        raise ValidationError(code, "%s: %s" % (path, error)) from error
    return [document for document in documents if document is not None]


def _load_one_mapping(path: Path, expected_kind: str) -> Dict[str, Any]:
    documents = _load_yaml_documents(path, code="unsupported-manifest")
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValidationError(
            "unsupported-manifest",
            "%s must contain exactly one mapping document" % path,
        )
    document = documents[0]
    if document.get("kind") != expected_kind:
        raise ValidationError(
            "unsupported-manifest",
            "%s must have kind %s" % (path, expected_kind),
            actual=document.get("kind"),
        )
    return document


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _find_kustomization(component_path: Path) -> Path:
    if not component_path.is_dir():
        raise ValidationError(
            "unsupported-manifest",
            "Component reference is not a directory: %s" % component_path,
        )
    matches = [
        component_path / name
        for name in KUSTOMIZATION_FILENAMES
        if (component_path / name).is_file()
    ]
    if len(matches) != 1:
        raise ValidationError(
            "unsupported-manifest",
            "Component directory %s must contain exactly one Kustomization file; found %d"
            % (component_path, len(matches)),
        )
    return matches[0].resolve()


def _reject_unsupported_fields(
    document: Mapping[str, Any], path: Path, *, component: bool
) -> None:
    allowed = COMPONENT_KUSTOMIZATION_KEYS if component else ROOT_KUSTOMIZATION_KEYS
    unsupported = sorted(set(document).difference(allowed))
    if unsupported:
        raise ValidationError(
            "unsupported-manifest",
            "%s uses unsupported Kustomize fields: %s" % (path, ", ".join(unsupported)),
        )


def _parse_target(raw: Any, *, label: str) -> _Target:
    if not isinstance(raw, dict):
        raise ValidationError(
            "unsupported-manifest", "patch target must be a mapping", layer=label
        )
    extra = sorted(set(raw).difference(TARGET_KEYS))
    if extra:
        raise ValidationError(
            "unsupported-manifest",
            "unsupported target selector fields: %s" % ", ".join(extra),
            layer=label,
        )
    required = ("group", "version", "kind")
    if any(not isinstance(raw.get(key), str) or not raw.get(key) for key in required):
        raise ValidationError(
            "unsupported-manifest",
            "target requires non-empty group, version, and kind strings",
            layer=label,
        )
    if (
        raw["group"] != "nvidia.com"
        or raw["version"] != "v1beta1"
        or raw["kind"] != BETA_DGD_KIND
    ):
        raise ValidationError(
            "unsupported-manifest",
            "only exact nvidia.com/v1beta1 DynamoGraphDeployment targets are supported",
            layer=label,
            actual={key: raw.get(key) for key in required},
        )
    return _Target(raw["group"], raw["version"], raw["kind"])


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _parse_operations(raw: Any, *, label: str) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValidationError(
            "unsupported-manifest",
            "JSON 6902 patch must be a non-empty operation list",
            layer=label,
        )
    operations: List[Mapping[str, Any]] = []
    for position, operation in enumerate(raw, start=1):
        if not isinstance(operation, dict):
            raise ValidationError(
                "unsupported-manifest",
                "operation must be a mapping",
                layer=label,
                op_index=position,
            )
        op_name = operation.get("op")
        path = operation.get("path")
        if not isinstance(op_name, str) or op_name not in SUPPORTED_OPERATIONS:
            raise ValidationError(
                "unsupported-manifest",
                "unsupported JSON Patch operation",
                layer=label,
                op_index=position,
                actual=op_name,
            )
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValidationError(
                "unsupported-manifest",
                "operation path must be a non-root JSON Pointer",
                layer=label,
                op_index=position,
                actual=path,
            )
        allowed_keys = {"op", "path"}
        if op_name != "remove":
            allowed_keys.add("value")
            if "value" not in operation or not _is_json_value(operation["value"]):
                raise ValidationError(
                    "unsupported-manifest",
                    "operation requires a JSON-compatible value",
                    layer=label,
                    op_index=position,
                    path=path,
                )
        elif "value" in operation:
            raise ValidationError(
                "unsupported-manifest",
                "remove operation must not contain value",
                layer=label,
                op_index=position,
                path=path,
            )
        extra = sorted(set(operation).difference(allowed_keys))
        if extra:
            raise ValidationError(
                "unsupported-manifest",
                "unsupported operation fields: %s" % ", ".join(extra),
                layer=label,
                op_index=position,
                path=path,
            )
        operations.append(operation)
    return tuple(operations)


def _load_patch(
    entry: Any,
    owner: Path,
    root: Path,
    *,
    is_component: bool,
    root_component: Optional[_RootComponent],
    index: int,
) -> _PatchLayer:
    owner_label = _relative_label(owner, root)
    provisional_label = "%s#patches[%d]" % (owner_label, index - 1)
    if not isinstance(entry, dict):
        raise ValidationError(
            "unsupported-manifest",
            "patch entry must be a mapping",
            layer=provisional_label,
        )
    extra = sorted(set(entry).difference({"target", "path", "patch"}))
    if extra:
        raise ValidationError(
            "unsupported-manifest",
            "unsupported patch entry fields: %s" % ", ".join(extra),
            layer=provisional_label,
        )
    has_path = "path" in entry
    has_inline = "patch" in entry
    if has_path == has_inline:
        raise ValidationError(
            "unsupported-manifest",
            "patch entry requires exactly one of path or patch",
            layer=provisional_label,
        )
    target = _parse_target(entry.get("target"), label=provisional_label)
    if has_path:
        relative = entry["path"]
        if not isinstance(relative, str) or not relative:
            raise ValidationError(
                "unsupported-manifest",
                "patch path must be a non-empty string",
                layer=provisional_label,
            )
        source = (owner.parent / relative).resolve()
        documents = _load_yaml_documents(source, code="unsupported-manifest")
        if len(documents) != 1:
            raise ValidationError(
                "unsupported-manifest",
                "%s must contain exactly one JSON 6902 document" % source,
                layer=provisional_label,
            )
        raw_operations = documents[0]
        label = _relative_label(source, root)
    else:
        inline = entry["patch"]
        if not isinstance(inline, str):
            raise ValidationError(
                "unsupported-manifest",
                "inline patch must be a YAML string",
                layer=provisional_label,
            )
        try:
            raw_operations = yaml.safe_load(inline)
        except yaml.YAMLError as error:
            raise ValidationError(
                "unsupported-manifest",
                "invalid inline patch: %s" % error,
                layer=provisional_label,
            ) from error
        source = owner
        label = provisional_label
    return _PatchLayer(
        label=label,
        source=source,
        target=target,
        operations=_parse_operations(raw_operations, label=label),
        is_component=is_component,
        root_component=root_component,
    )


def _component_layers(
    component_reference: str,
    owner: Path,
    root: Path,
    stack: Tuple[Path, ...],
    root_component: _RootComponent,
) -> List[_PatchLayer]:
    component_dir = (owner.parent / component_reference).resolve()
    component_kustomization = _find_kustomization(component_dir)
    if component_kustomization in stack:
        cycle = " -> ".join(
            _relative_label(item, root) for item in stack + (component_kustomization,)
        )
        raise ValidationError("unsupported-manifest", "Component cycle: %s" % cycle)
    document = _load_one_mapping(component_kustomization, "Component")
    if document.get("apiVersion") != "kustomize.config.k8s.io/v1alpha1":
        raise ValidationError(
            "unsupported-manifest",
            "%s must use Component apiVersion kustomize.config.k8s.io/v1alpha1"
            % component_kustomization,
        )
    _reject_unsupported_fields(document, component_kustomization, component=True)
    layers: List[_PatchLayer] = []
    nested = document.get("components", [])
    if not isinstance(nested, list) or not all(
        isinstance(item, str) and item for item in nested
    ):
        raise ValidationError(
            "unsupported-manifest",
            "%s components must be a list of paths" % component_kustomization,
        )
    for reference in nested:
        layers.extend(
            _component_layers(
                reference,
                component_kustomization,
                root,
                stack + (component_kustomization,),
                root_component,
            )
        )
    patches = document.get("patches", [])
    if not isinstance(patches, list):
        raise ValidationError(
            "unsupported-manifest",
            "%s patches must be a list" % component_kustomization,
        )
    for patch_index, entry in enumerate(patches, start=1):
        layers.append(
            _load_patch(
                entry,
                component_kustomization,
                root,
                is_component=True,
                root_component=root_component,
                index=patch_index,
            )
        )
    return layers


def _validate_root_contract(
    document: Mapping[str, Any], kustomization: Path
) -> Tuple[_RootComponent, ...]:
    expected_sort_options = {"order": "fifo"}
    actual_sort_options = document.get("sortOptions", _MISSING)
    if actual_sort_options != expected_sort_options:
        raise ValidationError(
            "component-order",
            "root Kustomization requires exact FIFO sort options",
            expected=expected_sort_options,
            actual=actual_sort_options,
        )

    rank_by_concern = {
        concern: rank for rank, concern in enumerate(CANONICAL_COMPONENT_ORDER)
    }
    root_components: List[_RootComponent] = []
    for index, reference in enumerate(document.get("components", [])):
        segments = reference.split("/")
        if (
            len(segments) != 3
            or segments[0] != "components"
            or segments[1] not in rank_by_concern
            or segments[2] not in ("agg", "disagg")
        ):
            raise ValidationError(
                "unsupported-manifest",
                "root Component path must be exactly "
                "components/<canonical-concern>/<agg|disagg>",
                actual=reference,
            )
        root_components.append(
            _RootComponent(
                index=index,
                reference=reference,
                resolved=(kustomization.parent / reference).resolve(),
                concern=segments[1],
                topology=segments[2],
            )
        )

    for placement in root_components:
        if placement.concern != "placement":
            continue
        scheduling = next(
            (
                component
                for component in root_components
                if component.concern == "scheduling"
                and component.topology == placement.topology
            ),
            None,
        )
        if scheduling is not None and placement.index < scheduling.index:
            raise ValidationError(
                "component-order",
                "placement Component %s at index %d must follow scheduling "
                "Component %s at index %d in canonical concern order"
                % (
                    placement.reference,
                    placement.index,
                    scheduling.reference,
                    scheduling.index,
                ),
            )

    previous: Optional[_RootComponent] = None
    for current in root_components:
        if previous is not None:
            previous_rank = rank_by_concern[previous.concern]
            current_rank = rank_by_concern[current.concern]
            if current_rank <= previous_rank:
                raise ValidationError(
                    "component-order",
                    "root Component %s at index %d must follow %s at index %d "
                    "in canonical concern order"
                    % (
                        current.reference,
                        current.index,
                        previous.reference,
                        previous.index,
                    ),
                    expected="rank greater than %d" % previous_rank,
                    actual=current_rank,
                )
        previous = current

    scheduling_topologies = {
        component.topology
        for component in root_components
        if component.concern == "scheduling"
    }
    for component in root_components:
        if (
            component.concern == "placement"
            and component.topology not in scheduling_topologies
        ):
            scheduling_reference = "components/scheduling/%s" % component.topology
            raise ValidationError(
                "component-dependency",
                "placement Component %s requires preceding scheduling Component %s"
                % (component.reference, scheduling_reference),
            )
    return tuple(root_components)


def _collect_layers(
    base: Path, kustomization: Path
) -> Tuple[Tuple[_PatchLayer, ...], Tuple[_RootComponent, ...]]:
    if kustomization.name not in KUSTOMIZATION_FILENAMES:
        raise ValidationError(
            "unsupported-manifest",
            "KUSTOMIZATION_YAML must use a standard Kustomize filename",
            actual=kustomization.name,
        )
    document = _load_one_mapping(kustomization, "Kustomization")
    if document.get("apiVersion") != "kustomize.config.k8s.io/v1beta1":
        raise ValidationError(
            "unsupported-manifest",
            "%s must use apiVersion kustomize.config.k8s.io/v1beta1" % kustomization,
        )
    _reject_unsupported_fields(document, kustomization, component=False)
    resources = document.get("resources")
    if (
        not isinstance(resources, list)
        or len(resources) != 1
        or not isinstance(resources[0], str)
    ):
        raise ValidationError(
            "unsupported-manifest",
            "%s resources must contain only BASE_YAML" % kustomization,
        )
    resolved_resource = (kustomization.parent / resources[0]).resolve()
    if resolved_resource != base.resolve():
        raise ValidationError(
            "unsupported-manifest",
            "root resource does not resolve to BASE_YAML",
            expected=str(base.resolve()),
            actual=str(resolved_resource),
        )
    components = document.get("components", [])
    if not isinstance(components, list) or not all(
        isinstance(item, str) and item for item in components
    ):
        raise ValidationError(
            "unsupported-manifest",
            "%s components must be a list of paths" % kustomization,
        )
    root_components = _validate_root_contract(document, kustomization)
    root = kustomization.parent.resolve()
    layers: List[_PatchLayer] = []
    for root_component in root_components:
        layers.extend(
            _component_layers(
                root_component.reference,
                kustomization,
                root,
                (),
                root_component,
            )
        )
    patches = document.get("patches", [])
    if not isinstance(patches, list):
        raise ValidationError(
            "unsupported-manifest", "%s patches must be a list" % kustomization
        )
    for patch_index, entry in enumerate(patches, start=1):
        layers.append(
            _load_patch(
                entry,
                kustomization,
                root,
                is_component=False,
                root_component=None,
                index=patch_index,
            )
        )
    return tuple(layers), root_components


def _api_parts(api_version: Any) -> Tuple[str, str]:
    if not isinstance(api_version, str):
        return "", ""
    if "/" in api_version:
        return tuple(api_version.split("/", 1))  # type: ignore[return-value]
    return "", api_version


def _matches_target(document: Any, target: _Target) -> bool:
    if not isinstance(document, dict):
        return False
    group, version = _api_parts(document.get("apiVersion"))
    if (
        group != target.group
        or version != target.version
        or document.get("kind") != target.kind
    ):
        return False
    return True


def _require_one_beta_dgd(documents: Sequence[Any], label: str) -> int:
    matches = [
        index
        for index, document in enumerate(documents)
        if isinstance(document, dict)
        and document.get("apiVersion") == BETA_DGD_API_VERSION
        and document.get("kind") == BETA_DGD_KIND
    ]
    if len(matches) != 1:
        raise ValidationError(
            "beta-dgd-count",
            "%s must contain exactly one %s %s; found %d"
            % (label, BETA_DGD_API_VERSION, BETA_DGD_KIND, len(matches)),
        )
    return matches[0]


def _validate_canonical_components(dgd: Mapping[str, Any]) -> str:
    spec = dgd.get("spec")
    components = spec.get("components") if isinstance(spec, dict) else None
    if not isinstance(components, list):
        raise ValidationError(
            "canonical-components",
            "DGD spec.components must be a list",
            path="/spec/components",
        )
    pairs: List[Tuple[Any, Any]] = []
    names: List[Any] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValidationError(
                "canonical-components",
                "component must be a mapping",
                path="/spec/components/%d" % index,
            )
        pair = (component.get("name"), component.get("type"))
        pairs.append(pair)
        names.append(pair[0])
    if len(names) != len(set(name for name in names if isinstance(name, str))):
        raise ValidationError(
            "canonical-components",
            "component names must be unique",
            path="/spec/components",
        )
    has_aggregate = "Worker" in names
    has_disaggregated = "PrefillWorker" in names or "DecodeWorker" in names
    if has_aggregate and has_disaggregated:
        raise ValidationError(
            "canonical-components",
            "base mixes aggregate and disaggregated canonical workers",
            path="/spec/components",
        )
    if has_disaggregated:
        topology = "disagg"
        expected = (
            ("Frontend", "frontend"),
            ("PrefillWorker", "prefill"),
            ("DecodeWorker", "decode"),
        )
    elif has_aggregate:
        topology = "agg"
        expected = (("Frontend", "frontend"), ("Worker", "worker"))
    else:
        raise ValidationError(
            "canonical-components",
            "cannot determine aggregate or disaggregated topology",
            path="/spec/components",
        )
    for index, expected_pair in enumerate(expected):
        actual_pair = pairs[index] if index < len(pairs) else _MISSING
        if actual_pair != expected_pair:
            raise ValidationError(
                "canonical-components",
                "canonical component is missing, reordered, or has the wrong type",
                path="/spec/components/%d" % index,
                expected=expected_pair,
                actual=actual_pair,
            )
    return topology


def _decode_pointer_token(
    token: str, *, layer: Optional[str], op_index: Optional[int], path: str
) -> str:
    decoded: List[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in ("0", "1"):
            raise ValidationError(
                "replay-path",
                "invalid JSON Pointer escape",
                layer=layer,
                op_index=op_index,
                path=path,
            )
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _pointer_tokens(
    path: str, *, layer: Optional[str] = None, op_index: Optional[int] = None
) -> Tuple[str, ...]:
    if not path.startswith("/"):
        raise ValidationError(
            "replay-path",
            "JSON Pointer must start with /",
            layer=layer,
            op_index=op_index,
            path=path,
        )
    return tuple(
        _decode_pointer_token(token, layer=layer, op_index=op_index, path=path)
        for token in path[1:].split("/")
    )


def _list_index(
    token: str,
    size: int,
    *,
    allow_end: bool,
    context: Tuple[Optional[str], Optional[int], str],
) -> int:
    layer, op_index, path = context
    if not re.fullmatch(r"0|[1-9][0-9]*", token):
        raise ValidationError(
            "replay-path",
            "invalid list index %r" % token,
            layer=layer,
            op_index=op_index,
            path=path,
        )
    index = int(token)
    maximum = size if allow_end else size - 1
    if index < 0 or index > maximum:
        raise ValidationError(
            "replay-path",
            "list index %d is out of bounds for length %d" % (index, size),
            layer=layer,
            op_index=op_index,
            path=path,
        )
    return index


def _resolve_parent(
    document: Any,
    tokens: Sequence[str],
    *,
    layer: Optional[str],
    op_index: Optional[int],
    path: str,
) -> Tuple[Any, str]:
    current = document
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise ValidationError(
                    "replay-path",
                    "parent JSON Pointer does not exist",
                    layer=layer,
                    op_index=op_index,
                    path=path,
                    actual=token,
                )
            current = current[token]
        elif isinstance(current, list):
            index = _list_index(
                token, len(current), allow_end=False, context=(layer, op_index, path)
            )
            current = current[index]
        else:
            raise ValidationError(
                "replay-path",
                "JSON Pointer traverses a scalar",
                layer=layer,
                op_index=op_index,
                path=path,
            )
    return current, tokens[-1]


def _try_resolve_parent(document: Any, tokens: Sequence[str]) -> Tuple[Any, Any]:
    current = document
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                return _MISSING, _MISSING
            current = current[token]
        elif isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                return _MISSING, _MISSING
            index = int(token)
            if index >= len(current):
                return _MISSING, _MISSING
            current = current[index]
        else:
            return _MISSING, _MISSING
    return current, tokens[-1]


def _value_at(
    document: Any, tokens: Sequence[str], *, layer: str, op_index: int, path: str
) -> Any:
    parent, token = _resolve_parent(
        document, tokens, layer=layer, op_index=op_index, path=path
    )
    if isinstance(parent, dict):
        if token not in parent:
            raise ValidationError(
                "replay-path",
                "JSON Pointer does not exist",
                layer=layer,
                op_index=op_index,
                path=path,
            )
        return parent[token]
    if isinstance(parent, list):
        index = _list_index(
            token, len(parent), allow_end=False, context=(layer, op_index, path)
        )
        return parent[index]
    raise ValidationError(
        "replay-path",
        "JSON Pointer parent is a scalar",
        layer=layer,
        op_index=op_index,
        path=path,
    )


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _is_path_prefix(prefix: Sequence[str], path: Sequence[str]) -> bool:
    return len(prefix) <= len(path) and tuple(prefix) == tuple(path[: len(prefix)])


def _test_is_invalidated(
    tested_path: Sequence[str], mutation_path: Sequence[str], op_name: str
) -> bool:
    if _is_path_prefix(tested_path, mutation_path) or _is_path_prefix(
        mutation_path, tested_path
    ):
        return True
    if op_name not in ("add", "remove") or not mutation_path:
        return False
    mutation_index = mutation_path[-1]
    if not re.fullmatch(r"0|[1-9][0-9]*", mutation_index):
        return False
    parent = mutation_path[:-1]
    if len(tested_path) <= len(parent) or tuple(tested_path[: len(parent)]) != tuple(
        parent
    ):
        return False
    tested_index = tested_path[len(parent)]
    return bool(
        re.fullmatch(r"0|[1-9][0-9]*", tested_index)
        and int(tested_index) >= int(mutation_index)
    )


def _validate_guards(layer: _PatchLayer) -> None:
    tested: Dict[Tuple[str, ...], Any] = {}
    for index, operation in enumerate(layer.operations, start=1):
        path = operation["path"]
        tokens = _pointer_tokens(path, layer=layer.label, op_index=index)
        op_name = operation["op"]
        if op_name == "test":
            tested[tokens] = operation["value"]
            continue
        if tokens in (("spec",), ("spec", "components")):
            raise ValidationError(
                "patch-guard",
                "mutation cannot replace an ancestor of guarded component identities",
                layer=layer.label,
                op_index=index,
                path=path,
            )
        if len(tokens) >= 3 and tokens[:2] == ("spec", "components"):
            component_index = tokens[2]
            if not re.fullmatch(r"0|[1-9][0-9]*", component_index):
                raise ValidationError(
                    "patch-guard",
                    "component mutation requires a concrete component index",
                    layer=layer.label,
                    op_index=index,
                    path=path,
                )
            if len(tokens) == 3 or tokens[3:] in (
                ("podTemplate",),
                ("podTemplate", "spec"),
            ):
                raise ValidationError(
                    "patch-guard",
                    "mutation cannot replace an ancestor of guarded component or container identities",
                    layer=layer.label,
                    op_index=index,
                    path=path,
                )
            for identity in ("name", "type"):
                guard_path = ("spec", "components", component_index, identity)
                if guard_path not in tested:
                    raise ValidationError(
                        "patch-guard",
                        "component mutation requires preceding name and type tests",
                        layer=layer.label,
                        op_index=index,
                        path=path,
                    )
            if (
                len(tokens) >= 6
                and tokens[3:5] == ("podTemplate", "spec")
                and tokens[5] in CONTAINER_COLLECTIONS
            ):
                collection = tokens[5]
                collection_path = tokens[:6]
                collection_suffix = tokens[6:]
                if not collection_suffix:
                    if op_name in ("replace", "remove"):
                        raise ValidationError(
                            "patch-guard",
                            "mutation cannot replace a container identity collection",
                            layer=layer.label,
                            op_index=index,
                            path=path,
                        )
                elif collection_suffix[0] == "-":
                    value = operation.get("value", _MISSING)
                    if (
                        len(collection_suffix) != 1
                        or op_name != "add"
                        or not isinstance(value, dict)
                        or not isinstance(value.get("name"), str)
                        or not value["name"]
                    ):
                        raise ValidationError(
                            "patch-guard",
                            "%s/- requires an add of one complete container mapping "
                            "with a non-empty string name" % collection,
                            layer=layer.label,
                            op_index=index,
                            path=path,
                        )
                elif not re.fullmatch(r"0|[1-9][0-9]*", collection_suffix[0]):
                    raise ValidationError(
                        "patch-guard",
                        "container mutation requires a concrete container index",
                        layer=layer.label,
                        op_index=index,
                        path=path,
                    )
                else:
                    container_guard = collection_path + (
                        collection_suffix[0],
                        "name",
                    )
                    if container_guard not in tested:
                        raise ValidationError(
                            "patch-guard",
                            "container mutation requires a preceding container name test",
                            layer=layer.label,
                            op_index=index,
                            path=path,
                        )
        if op_name in ("replace", "remove") and tokens not in tested:
            raise ValidationError(
                "patch-guard",
                "%s requires a preceding test on the same path" % op_name,
                layer=layer.label,
                op_index=index,
                path=path,
            )
        tested = {
            tested_path: value
            for tested_path, value in tested.items()
            if not _test_is_invalidated(tested_path, tokens, op_name)
        }


def _walk_env_entries(
    value: Any, tokens: Tuple[str, ...] = ()
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_tokens = tokens + (str(key),)
            if key == "env" and isinstance(child, list):
                for index, entry in enumerate(child):
                    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                        yield _encode_pointer(child_tokens + (str(index),)), entry
            yield from _walk_env_entries(child, child_tokens)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_env_entries(child, tokens + (str(index),))


def _selected_env_append_names(layers: Sequence[_PatchLayer]) -> Set[str]:
    names: Set[str] = set()
    for layer in layers:
        if not layer.is_component:
            continue
        for index, operation in enumerate(layer.operations, start=1):
            if operation["op"] != "add":
                continue
            tokens = _pointer_tokens(
                operation["path"], layer=layer.label, op_index=index
            )
            if len(tokens) >= 2 and tokens[-2:] == ("env", "-"):
                value = operation["value"]
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("name"), str)
                    or not value["name"]
                ):
                    raise ValidationError(
                        "unsupported-manifest",
                        "env/- add requires a value with a non-empty string name",
                        layer=layer.label,
                        op_index=index,
                        path=operation["path"],
                    )
                names.add(value["name"])
    return names


def _validate_base_ownership(
    base_documents: Sequence[Any], dgd_index: int, layers: Sequence[_PatchLayer]
) -> None:
    dgd = base_documents[dgd_index]
    forbidden_names = set(FORBIDDEN_BASE_ENV_NAMES)
    forbidden_names.update(_selected_env_append_names(layers))
    for path, entry in _walk_env_entries(dgd):
        if entry["name"] in forbidden_names:
            raise ValidationError(
                "base-env-ownership",
                "base defines cluster-owned environment variable %s" % entry["name"],
                path=path + "/name",
            )
    for layer in layers:
        if not layer.is_component:
            continue
        target_matches = [
            document
            for document in base_documents
            if _matches_target(document, layer.target)
        ]
        if len(target_matches) != 1:
            continue
        target_document = target_matches[0]
        for index, operation in enumerate(layer.operations, start=1):
            if operation["op"] != "add":
                continue
            tokens = _pointer_tokens(
                operation["path"], layer=layer.label, op_index=index
            )
            parent, token = _try_resolve_parent(target_document, tokens)
            if isinstance(parent, dict) and token in parent:
                raise ValidationError(
                    "base-field-ownership",
                    "base already owns a whole mapping field selected Component adds",
                    layer=layer.label,
                    op_index=index,
                    path=operation["path"],
                    actual=parent[token],
                )


def _apply_operation(
    document: Any, layer: _PatchLayer, op_index: int, operation: Mapping[str, Any]
) -> None:
    op_name = operation["op"]
    path = operation["path"]
    tokens = _pointer_tokens(path, layer=layer.label, op_index=op_index)
    if op_name == "test":
        actual = _value_at(
            document, tokens, layer=layer.label, op_index=op_index, path=path
        )
        if not _json_equal(actual, operation["value"]):
            raise ValidationError(
                "replay-test",
                "JSON Patch test failed",
                layer=layer.label,
                op_index=op_index,
                path=path,
                expected=operation["value"],
                actual=actual,
            )
        return
    parent, token = _resolve_parent(
        document, tokens, layer=layer.label, op_index=op_index, path=path
    )
    if op_name == "add":
        value = copy.deepcopy(operation["value"])
        if isinstance(parent, dict):
            if token in parent:
                raise ValidationError(
                    "replay-derived-absence",
                    "whole-field add requires the mapping key to be absent",
                    layer=layer.label,
                    op_index=op_index,
                    path=path,
                    actual=parent[token],
                )
            parent[token] = value
            return
        if isinstance(parent, list):
            if token == "-":
                if len(tokens) >= 2 and tokens[-2] == "env":
                    if (
                        not isinstance(value, dict)
                        or not isinstance(value.get("name"), str)
                        or not value["name"]
                    ):
                        raise ValidationError(
                            "unsupported-manifest",
                            "env/- add requires a value with a non-empty string name",
                            layer=layer.label,
                            op_index=op_index,
                            path=path,
                        )
                    duplicate_index = next(
                        (
                            index
                            for index, entry in enumerate(parent)
                            if isinstance(entry, dict)
                            and entry.get("name") == value["name"]
                        ),
                        None,
                    )
                    if duplicate_index is not None:
                        raise ValidationError(
                            "replay-duplicate-env",
                            "environment name %s already exists at index %d"
                            % (value["name"], duplicate_index),
                            layer=layer.label,
                            op_index=op_index,
                            path=path,
                        )
                parent.append(value)
                return
            index = _list_index(
                token,
                len(parent),
                allow_end=True,
                context=(layer.label, op_index, path),
            )
            parent.insert(index, value)
            return
        raise ValidationError(
            "replay-path",
            "add parent must be a mapping or list",
            layer=layer.label,
            op_index=op_index,
            path=path,
        )
    if isinstance(parent, dict):
        if token not in parent:
            raise ValidationError(
                "replay-path",
                "%s target does not exist" % op_name,
                layer=layer.label,
                op_index=op_index,
                path=path,
            )
        current = parent[token]
        if op_name == "replace":
            if _json_equal(current, operation["value"]):
                raise ValidationError(
                    "replay-no-change",
                    "replace must change the accumulated document",
                    layer=layer.label,
                    op_index=op_index,
                    path=path,
                    actual=current,
                )
            parent[token] = copy.deepcopy(operation["value"])
        else:
            del parent[token]
        return
    if isinstance(parent, list):
        index = _list_index(
            token,
            len(parent),
            allow_end=False,
            context=(layer.label, op_index, path),
        )
        current = parent[index]
        if op_name == "replace":
            if _json_equal(current, operation["value"]):
                raise ValidationError(
                    "replay-no-change",
                    "replace must change the accumulated document",
                    layer=layer.label,
                    op_index=op_index,
                    path=path,
                    actual=current,
                )
            parent[index] = copy.deepcopy(operation["value"])
        else:
            del parent[index]
        return
    raise ValidationError(
        "replay-path",
        "%s parent must be a mapping or list" % op_name,
        layer=layer.label,
        op_index=op_index,
        path=path,
    )


def _missing_placement_affinity_parent(
    document: Any, layer: _PatchLayer, operation: Mapping[str, Any]
) -> bool:
    root_component = layer.root_component
    path = operation["path"]
    if (
        root_component is None
        or root_component.concern != "placement"
        or operation["op"] != "add"
        or not re.fullmatch(
            r"/spec/components/(0|[1-9][0-9]*)/podTemplate/spec/affinity/podAffinity",
            path,
        )
    ):
        return False
    tokens = tuple(path[1:].split("/"))
    pod_spec, affinity_key = _try_resolve_parent(document, tokens[:-1])
    return (
        isinstance(pod_spec, dict)
        and affinity_key == "affinity"
        and affinity_key not in pod_spec
    )


def _replay(base_documents: Sequence[Any], layers: Sequence[_PatchLayer]) -> List[Any]:
    result = copy.deepcopy(list(base_documents))
    for layer in layers:
        matches = [
            index
            for index, document in enumerate(result)
            if _matches_target(document, layer.target)
        ]
        if len(matches) != 1:
            raise ValidationError(
                "patch-target-count",
                "patch target must match exactly one accumulated resource; found %d"
                % len(matches),
                layer=layer.label,
            )
        target_document = result[matches[0]]
        for op_index, operation in enumerate(layer.operations, start=1):
            try:
                _apply_operation(target_document, layer, op_index, operation)
            except ValidationError as error:
                if error.code != "replay-path":
                    raise
                if not _missing_placement_affinity_parent(
                    target_document,
                    layer,
                    operation,
                ):
                    raise
                root_component = layer.root_component
                if root_component is None:
                    raise
                scheduling_reference = "components/scheduling/%s" % (
                    root_component.topology
                )
                raise ValidationError(
                    "component-dependency",
                    "placement Component %s requires scheduling Component %s to "
                    "establish the accumulated affinity parent before placement "
                    "adds podAffinity"
                    % (root_component.reference, scheduling_reference),
                    layer=layer.label,
                    op_index=op_index,
                    path=operation["path"],
                ) from error
    return result


def _encode_pointer(tokens: Sequence[str]) -> str:
    if not tokens:
        return "/"
    return "/" + "/".join(
        token.replace("~", "~0").replace("/", "~1") for token in tokens
    )


def _first_difference(
    expected: Any, actual: Any, tokens: Tuple[str, ...] = ()
) -> Optional[_Difference]:
    if _json_equal(expected, actual):
        return None
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys, key=str):
            return _Difference(
                _encode_pointer(tokens + (str(key),)), expected[key], _MISSING
            )
        for key in sorted(actual_keys - expected_keys, key=str):
            return _Difference(
                _encode_pointer(tokens + (str(key),)), _MISSING, actual[key]
            )
        for key in sorted(expected_keys, key=str):
            difference = _first_difference(
                expected[key], actual[key], tokens + (str(key),)
            )
            if difference is not None:
                return difference
    elif isinstance(expected, list) and isinstance(actual, list):
        common = min(len(expected), len(actual))
        for index in range(common):
            difference = _first_difference(
                expected[index], actual[index], tokens + (str(index),)
            )
            if difference is not None:
                return difference
        if len(expected) != len(actual):
            index = common
            return _Difference(
                _encode_pointer(tokens + (str(index),)),
                expected[index] if index < len(expected) else _MISSING,
                actual[index] if index < len(actual) else _MISSING,
            )
    return _Difference(_encode_pointer(tokens), expected, actual)


def _resource_identity(document: Mapping[str, Any], label: str, index: int) -> str:
    api_version = document.get("apiVersion")
    kind = document.get("kind")
    metadata = document.get("metadata")
    name = metadata.get("name") if isinstance(metadata, dict) else None
    namespace = metadata.get("namespace", "") if isinstance(metadata, dict) else ""
    if not all(isinstance(item, str) and item for item in (api_version, kind, name)):
        raise ValidationError(
            "render-equality",
            "%s document %d lacks apiVersion, kind, or metadata.name needed for semantic comparison"
            % (label, index),
            path="/documents/%d" % index,
        )
    if not isinstance(namespace, str):
        raise ValidationError(
            "render-equality",
            "%s document %d has a non-string metadata.namespace" % (label, index),
            path="/documents/%d/metadata/namespace" % index,
        )
    return "%s|%s|%s|%s" % (api_version, kind, namespace, name)


def _index_resources(
    documents: Sequence[Any], label: str
) -> Dict[str, Mapping[str, Any]]:
    resources: Dict[str, Mapping[str, Any]] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValidationError(
                "render-equality",
                "%s document %d is not a Kubernetes mapping" % (label, index),
                path="/documents/%d" % index,
            )
        identity = _resource_identity(document, label, index)
        if identity in resources:
            raise ValidationError(
                "render-equality",
                "%s contains duplicate resource identity %s" % (label, identity),
                path="/resources/%s" % identity.replace("/", "~1"),
            )
        resources[identity] = document
    return resources


def _version_output(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValidationError(
            "kustomize-version", "%s: %s" % (executable, error)
        ) from error
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise ValidationError(
            "kustomize-version", "failed to query Kustomize version: %s" % output
        )
    return output


def _require_kustomize_version(executable: str) -> None:
    output = _version_output(executable)
    versions = re.findall(
        r"(?<![0-9A-Za-z])v?[0-9]+\.[0-9]+\.[0-9]+(?![0-9A-Za-z.+-])",
        output,
    )
    normalized = [
        version if version.startswith("v") else "v" + version for version in versions
    ]
    if normalized != [REQUIRED_KUSTOMIZE_VERSION]:
        raise ValidationError(
            "kustomize-version",
            "validator requires exact Kustomize %s" % REQUIRED_KUSTOMIZE_VERSION,
            actual=output,
        )


def _run_kustomize(kustomization: Path, executable: str) -> _BuildResult:
    try:
        completed = subprocess.run(
            [
                executable,
                "build",
                str(kustomization.parent),
                "--load-restrictor",
                "LoadRestrictionsNone",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValidationError(
            "kustomize-build", "%s: %s" % (executable, error)
        ) from error
    return _BuildResult(completed.returncode, completed.stdout, completed.stderr)


def validate_case(
    base_path: Path,
    kustomization_path: Path,
    *,
    kustomize_bin: Optional[Union[str, Path]] = None,
) -> None:
    """Validate one scaffold case or raise :class:`ValidationError`."""

    base = Path(base_path).resolve()
    kustomization = Path(kustomization_path).resolve()
    executable = str(kustomize_bin or os.environ.get("KUSTOMIZE_BIN") or "kustomize")
    base_documents = _load_yaml_documents(base)
    if not base_documents or not all(
        isinstance(document, dict) for document in base_documents
    ):
        raise ValidationError(
            "yaml-parse", "%s must contain Kubernetes mapping documents" % base
        )
    dgd_index = _require_one_beta_dgd(base_documents, "base")
    _validate_canonical_components(base_documents[dgd_index])
    layers, _root_components = _collect_layers(base, kustomization)
    for layer in layers:
        _validate_guards(layer)
    _validate_base_ownership(base_documents, dgd_index, layers)

    _require_kustomize_version(executable)
    build = _run_kustomize(kustomization, executable)
    replay_error: Optional[ValidationError] = None
    replayed: Optional[List[Any]] = None
    try:
        replayed = _replay(base_documents, layers)
    except ValidationError as error:
        replay_error = error

    if replay_error is not None:
        if replay_error.code == "replay-test" and build.returncode != 0:
            renderer_error = build.stderr.strip() or build.stdout.strip()
            raise ValidationError(
                "kustomize-build",
                renderer_error or "Kustomize build failed during a JSON Patch test",
                layer=replay_error.layer,
                op_index=replay_error.op_index,
                path=replay_error.path,
                expected=replay_error.expected,
                actual=replay_error.actual,
            ) from replay_error
        raise replay_error
    if build.returncode != 0:
        renderer_error = build.stderr.strip() or build.stdout.strip()
        raise ValidationError(
            "kustomize-build", renderer_error or "Kustomize build failed"
        )
    assert replayed is not None
    try:
        rendered_documents = [
            document
            for document in yaml.safe_load_all(build.stdout)
            if document is not None
        ]
    except yaml.YAMLError as error:
        raise ValidationError(
            "render-yaml", "Kustomize output is invalid YAML: %s" % error
        ) from error
    if not rendered_documents or not all(
        isinstance(document, dict) for document in rendered_documents
    ):
        raise ValidationError(
            "render-yaml", "Kustomize output must contain Kubernetes mapping documents"
        )
    rendered_dgd_index = _require_one_beta_dgd(rendered_documents, "render")
    _validate_canonical_components(rendered_documents[rendered_dgd_index])
    replayed_resources = _index_resources(replayed, "replay")
    rendered_resources = _index_resources(rendered_documents, "render")
    difference = _first_difference(
        replayed_resources, rendered_resources, ("resources",)
    )
    if difference is not None:
        raise ValidationError(
            "render-equality",
            "sequential replay does not equal parsed Kustomize output",
            path=difference.path,
            expected=difference.expected,
            actual=difference.actual,
        )
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a beta recipe Kustomization by sequential JSON Patch replay."
    )
    parser.add_argument("base_yaml", type=Path, metavar="BASE_YAML")
    parser.add_argument("kustomization_yaml", type=Path, metavar="KUSTOMIZATION_YAML")
    parser.add_argument(
        "--kustomize-bin",
        default=os.environ.get("KUSTOMIZE_BIN") or "kustomize",
        help="Kustomize v5.8.1 executable (default: KUSTOMIZE_BIN or PATH)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_case(
            args.base_yaml,
            args.kustomization_yaml,
            kustomize_bin=args.kustomize_bin,
        )
    except ValidationError as error:
        print(error.diagnostic(), file=sys.stderr)
        return 1
    print(
        "OK: validation passed; sequential replay equals Kustomize %s render"
        % REQUIRED_KUSTOMIZE_VERSION
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
