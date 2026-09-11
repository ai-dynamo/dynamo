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
used by the recipe scaffold: one multi-document base, ordered Components, and
``patches`` written either as JSON 6902 operations or as strategic merge
patches. Merge patches are lowered into guarded JSON 6902 operations by name
against the accumulated document, so one replay contract covers both styles.
It uses only the Python standard library and PyYAML, and verifies its replay
against Kustomize v5.8.1.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
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
    {"apiVersion", "kind", "components", "patches", "openapi"}
)
SCHEMA_COMPONENT_NAME = "dynamo-openapi"
SCHEMA_COMPONENT_REFERENCE = "components/" + SCHEMA_COMPONENT_NAME
MERGE_PATCH_TOP_LEVEL_KEYS = frozenset({"apiVersion", "kind", "metadata", "spec"})
CANONICAL_COMPONENT_ORDER = (
    SCHEMA_COMPONENT_NAME,
    "cache-binding",
    "registry-credentials",
    "probes",
    "scheduling",
    "network-interface",
    "placement",
)
_NETWORK_ROOT_CONCERNS = frozenset(
    {"network-generic", "network-provider", "network-private"}
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
    merge: Optional[Mapping[str, Any]] = None


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
    index: int,
    root_component: Optional[_RootComponent],
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
    if isinstance(raw_operations, dict):
        return _PatchLayer(
            label=label,
            source=source,
            target=target,
            operations=(),
            is_component=is_component,
            root_component=root_component,
            merge=_parse_merge_patch(raw_operations, target, label=label),
        )
    return _PatchLayer(
        label=label,
        source=source,
        target=target,
        operations=_parse_operations(raw_operations, label=label),
        is_component=is_component,
        root_component=root_component,
    )


def _reject_merge_directives(
    value: Any, *, label: str, tokens: Tuple[str, ...]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValidationError(
                    "merge-patch",
                    "merge patch keys must be strings",
                    layer=label,
                    path=_encode_pointer(tokens),
                )
            if key.startswith("$"):
                raise ValidationError(
                    "merge-patch",
                    "strategic merge directives such as %s are not supported" % key,
                    layer=label,
                    path=_encode_pointer(tokens + (key,)),
                )
            if child is None:
                raise ValidationError(
                    "merge-patch",
                    "null values are not supported; omit the field instead",
                    layer=label,
                    path=_encode_pointer(tokens + (key,)),
                )
            _reject_merge_directives(child, label=label, tokens=tokens + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if child is None:
                raise ValidationError(
                    "merge-patch",
                    "null list items are not supported",
                    layer=label,
                    path=_encode_pointer(tokens + (str(index),)),
                )
            _reject_merge_directives(child, label=label, tokens=tokens + (str(index),))


def _parse_merge_patch(
    raw: Mapping[str, Any], target: _Target, *, label: str
) -> Mapping[str, Any]:
    """Accept one strategic merge patch document for the exact DGD target."""

    extra = sorted(set(raw).difference(MERGE_PATCH_TOP_LEVEL_KEYS))
    if extra:
        raise ValidationError(
            "merge-patch",
            "unsupported merge patch fields: %s" % ", ".join(extra),
            layer=label,
        )
    expected_api_version = "%s/%s" % (target.group, target.version)
    if raw.get("apiVersion") != expected_api_version or raw.get("kind") != target.kind:
        raise ValidationError(
            "merge-patch",
            "merge patch apiVersion and kind must match the patch target",
            layer=label,
            expected={"apiVersion": expected_api_version, "kind": target.kind},
            actual={"apiVersion": raw.get("apiVersion"), "kind": raw.get("kind")},
        )
    metadata = raw.get("metadata")
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"name"}
        or not isinstance(metadata.get("name"), str)
        or not metadata["name"]
    ):
        raise ValidationError(
            "merge-patch",
            "merge patch metadata must contain only a placeholder name; "
            "the target selector chooses the DGD",
            layer=label,
        )
    spec = raw.get("spec")
    if not isinstance(spec, dict) or not spec:
        raise ValidationError(
            "merge-patch",
            "merge patch requires a non-empty spec mapping",
            layer=label,
        )
    if not _is_json_value(spec):
        raise ValidationError(
            "merge-patch",
            "merge patch spec must contain JSON-compatible values",
            layer=label,
        )
    _reject_merge_directives(spec, label=label, tokens=("spec",))
    return raw


def _classify_root_component(reference: str) -> Optional[Tuple[str, str]]:
    if reference == SCHEMA_COMPONENT_REFERENCE:
        return "openapi", ""
    segments = reference.split("/")
    if (
        len(segments) == 3
        and segments[0] == "components"
        and segments[1] in CANONICAL_COMPONENT_ORDER
        and segments[1] != SCHEMA_COMPONENT_NAME
        and segments[2] in ("agg", "disagg")
    ):
        concern = (
            "network-generic" if segments[1] == "network-interface" else segments[1]
        )
        return concern, segments[2]
    if (
        len(segments) == 4
        and segments[0] == "components"
        and segments[1] == "provider-networking"
        and segments[2]
        and segments[3] in ("agg", "disagg")
    ):
        return "network-provider", segments[3]
    if (
        len(segments) == 3
        and segments[0] == "components"
        and segments[1] == "networking"
        and segments[2] in ("agg", "disagg")
    ):
        return "network-private", segments[2]
    return None


def _canonical_root_concern(concern: str) -> str:
    if concern in _NETWORK_ROOT_CONCERNS:
        return "network-interface"
    if concern == "openapi":
        return SCHEMA_COMPONENT_NAME
    return concern


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
    if root_component.concern == "openapi":
        _validate_schema_component(document, component_kustomization)
        return []
    if "openapi" in document:
        raise ValidationError(
            "unsupported-manifest",
            "%s declares openapi; only %s supplies the strategic merge schema"
            % (component_kustomization, SCHEMA_COMPONENT_REFERENCE),
        )
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
        classified = _classify_root_component(reference)
        if classified is None:
            raise ValidationError(
                "unsupported-manifest",
                "root Component path must be a canonical concern, a qualified "
                "provider-networking concern, or a private networking concern",
                actual=reference,
            )
        concern, topology = classified
        root_components.append(
            _RootComponent(
                index=index,
                reference=reference,
                resolved=(kustomization.parent / reference).resolve(),
                concern=concern,
                topology=topology,
            )
        )

    networking_components = [
        component
        for component in root_components
        if component.concern in _NETWORK_ROOT_CONCERNS
    ]
    if len(networking_components) > 1:
        raise ValidationError(
            "networking-slot",
            "select at most one generic, provider, or private networking Component",
            actual=[component.reference for component in networking_components],
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

    if networking_components:
        networking = networking_components[0]
        scheduling_indices = [
            component.index
            for component in root_components
            if component.concern == "scheduling"
        ]
        placement_indices = [
            component.index
            for component in root_components
            if component.concern == "placement"
        ]
        if (scheduling_indices and networking.index <= max(scheduling_indices)) or (
            placement_indices and networking.index >= min(placement_indices)
        ):
            raise ValidationError(
                "networking-slot",
                "networking must follow scheduling and precede placement",
                actual=networking.reference,
            )

    previous: Optional[_RootComponent] = None
    for current in root_components:
        if previous is not None:
            previous_rank = rank_by_concern[_canonical_root_concern(previous.concern)]
            current_rank = rank_by_concern[_canonical_root_concern(current.concern)]
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


def _validate_schema_component(
    document: Mapping[str, Any], kustomization_path: Path
) -> Path:
    openapi = document.get("openapi")
    if (
        not isinstance(openapi, dict)
        or set(openapi) != {"path"}
        or not isinstance(openapi.get("path"), str)
        or not openapi["path"]
    ):
        raise ValidationError(
            "merge-patch",
            "%s must declare openapi.path and no other openapi field"
            % kustomization_path,
        )
    for key in ("patches", "components"):
        if key in document:
            raise ValidationError(
                "merge-patch",
                "%s must not declare %s; it only supplies the merge schema"
                % (kustomization_path, key),
            )
    schema_path = (kustomization_path.parent / openapi["path"]).resolve()
    if not schema_path.is_file():
        raise ValidationError(
            "merge-patch",
            "OpenAPI schema file does not exist",
            actual=str(schema_path),
        )
    return schema_path


def _load_schema(
    root_components: Sequence[_RootComponent],
) -> Optional[Dict[str, Any]]:
    schema_component = next(
        (root for root in root_components if root.concern == "openapi"), None
    )
    if schema_component is None:
        return None
    kustomization_path = _find_kustomization(schema_component.resolved)
    document = _load_one_mapping(kustomization_path, "Component")
    schema_path = _validate_schema_component(document, kustomization_path)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValidationError(
            "merge-patch", "cannot read OpenAPI schema %s: %s" % (schema_path, error)
        ) from error
    if not isinstance(schema, dict) or not isinstance(schema.get("definitions"), dict):
        raise ValidationError(
            "merge-patch",
            "OpenAPI schema must contain a definitions mapping",
            actual=str(schema_path),
        )
    return schema


def _schema_property(node: Any, key: str) -> Any:
    if not isinstance(node, dict):
        return None
    properties = node.get("properties")
    if isinstance(properties, dict) and key in properties:
        return properties[key]
    additional = node.get("additionalProperties")
    return additional if isinstance(additional, dict) else None


def _schema_items(node: Any) -> Any:
    items = node.get("items") if isinstance(node, dict) else None
    return items if isinstance(items, dict) else None


def _schema_merge_key(node: Any) -> Optional[str]:
    if isinstance(node, dict) and node.get("x-kubernetes-patch-strategy") == "merge":
        key = node.get("x-kubernetes-patch-merge-key")
        if isinstance(key, str) and key:
            return key
    return None


def _merge_value(patch: Any, current: Any, schema: Any) -> Any:
    """Compute Kustomize's strategic merge of patch into current.

    Keyed lists place the patch elements first in patch order, merged with
    their matches, followed by the untouched current elements. Unkeyed lists
    and scalars are replaced. Mappings merge key by key.
    """

    if isinstance(patch, dict) and isinstance(current, dict):
        merged = copy.deepcopy(current)
        for key, value in patch.items():
            if key in current:
                merged[key] = _merge_value(
                    value, current[key], _schema_property(schema, key)
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(patch, list) and isinstance(current, list):
        merge_key = _schema_merge_key(schema)
        if merge_key is None:
            return copy.deepcopy(patch)
        item_schema = _schema_items(schema)
        result: List[Any] = []
        consumed: Set[int] = set()
        for item in patch:
            key = item.get(merge_key) if isinstance(item, dict) else None
            match = next(
                (
                    index
                    for index, element in enumerate(current)
                    if index not in consumed
                    and isinstance(element, dict)
                    and element.get(merge_key) == key
                ),
                None,
            )
            if match is None:
                result.append(copy.deepcopy(item))
            else:
                consumed.add(match)
                result.append(_merge_value(item, current[match], item_schema))
        result.extend(
            copy.deepcopy(element)
            for index, element in enumerate(current)
            if index not in consumed
        )
        return result
    return copy.deepcopy(patch)


class _MergeLowering:
    """Lower one strategic merge patch into guarded JSON 6902 operations.

    The lowered operations replay to exactly the document Kustomize renders,
    and they carry the same component and container identity tests that the
    hand-written JSON 6902 Components use, so every downstream contract check
    applies unchanged.
    """

    def __init__(self, layer: _PatchLayer, document: Mapping[str, Any]) -> None:
        self.layer = layer
        self.document = document
        self.operations: List[Dict[str, Any]] = []
        self._tested: Set[Tuple[str, ...]] = set()

    def _error(
        self, message: str, tokens: Tuple[str, ...], **details: Any
    ) -> ValidationError:
        return ValidationError(
            "merge-patch",
            message,
            layer=self.layer.label,
            path=_encode_pointer(tokens),
            **details,
        )

    def _emit(
        self, op_name: str, tokens: Tuple[str, ...], value: Any = _MISSING
    ) -> None:
        operation: Dict[str, Any] = {"op": op_name, "path": _encode_pointer(tokens)}
        if value is not _MISSING:
            operation["value"] = copy.deepcopy(value)
        self.operations.append(operation)
        if op_name == "test":
            self._tested.add(tuple(tokens))
        else:
            self._tested = {
                tested
                for tested in self._tested
                if not _test_is_invalidated(tested, tokens, op_name)
            }

    def _test_current(self, tokens: Tuple[str, ...]) -> None:
        if tuple(tokens) in self._tested:
            return
        parent, token = _try_resolve_parent(self.document, tokens)
        value: Any = _MISSING
        if isinstance(parent, dict) and token in parent:
            value = parent[token]
        elif isinstance(parent, list) and _is_list_slot(token) and token != "-":
            if int(token) < len(parent):
                value = parent[int(token)]
        if value is _MISSING:
            raise self._error(
                "merge patch requires an identity value the accumulated document lacks",
                tokens,
            )
        self._emit("test", tokens, value)

    def _identity(self, tokens: Tuple[str, ...]) -> None:
        """Emit the component and container identity tests guarding tokens."""

        if (
            len(tokens) < 3
            or tokens[:2] != ("spec", "components")
            or not re.fullmatch(r"0|[1-9][0-9]*", tokens[2])
        ):
            return
        component = tokens[:3]
        for identity in ("name", "type"):
            self._test_current(component + (identity,))
        if (
            len(tokens) >= 7
            and tokens[3:5] == ("podTemplate", "spec")
            and tokens[5] in CONTAINER_COLLECTIONS
            and re.fullmatch(r"0|[1-9][0-9]*", tokens[6])
        ):
            self._test_current(tokens[:7] + ("name",))

    def lower_mapping(
        self,
        patch: Mapping[str, Any],
        current: Mapping[str, Any],
        tokens: Tuple[str, ...],
        schema: Any,
    ) -> None:
        for key, value in patch.items():
            child = tokens + (key,)
            child_schema = _schema_property(schema, key)
            if key not in current:
                self._identity(child)
                self._emit("add", child, value)
                continue
            existing = current[key]
            if isinstance(value, dict):
                if not isinstance(existing, dict):
                    raise self._error(
                        "merge patch mapping cannot merge into a non-mapping field",
                        child,
                    )
                self.lower_mapping(value, existing, child, child_schema)
            elif isinstance(value, list):
                if not isinstance(existing, list):
                    raise self._error(
                        "merge patch list cannot merge into a non-list field", child
                    )
                merge_key = _schema_merge_key(child_schema)
                if merge_key is None:
                    if not _json_equal(existing, value):
                        self._identity(child)
                        self._emit("test", child, existing)
                        self._emit("replace", child, value)
                else:
                    self.lower_list(
                        value, existing, child, merge_key, _schema_items(child_schema)
                    )
            elif not _json_equal(existing, value):
                self._identity(child)
                self._emit("test", child, existing)
                self._emit("replace", child, value)

    def lower_list(
        self,
        patch_items: Sequence[Any],
        current: Sequence[Any],
        tokens: Tuple[str, ...],
        merge_key: str,
        item_schema: Any,
    ) -> None:
        keys: List[str] = []
        for position, item in enumerate(patch_items):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get(merge_key), str)
                or not item[merge_key]
            ):
                raise self._error(
                    "merge patch list items require a non-empty string %s" % merge_key,
                    tokens + (str(position),),
                )
            if item[merge_key] in keys:
                raise self._error(
                    "merge patch repeats %s %s" % (merge_key, item[merge_key]),
                    tokens + (str(position),),
                )
            keys.append(item[merge_key])
        current_keys = [
            element.get(merge_key) if isinstance(element, dict) else None
            for element in current
        ]
        if tokens == ("spec", "components"):
            if keys != current_keys[: len(keys)]:
                raise self._error(
                    "merge patch components must form an ordered prefix of the base "
                    "component list; unmatched names would append new components",
                    tokens,
                    expected=current_keys,
                    actual=keys,
                )
        elif (
            len(tokens) == 6
            and tokens[:2] == ("spec", "components")
            and tokens[3:5] == ("podTemplate", "spec")
            and tokens[5] in CONTAINER_COLLECTIONS
        ):
            if any(key not in current_keys for key in keys):
                raise self._error(
                    "merge patch may only address containers the base defines",
                    tokens,
                    expected=current_keys,
                    actual=keys,
                )
            if [key for key in current_keys if key in keys] != keys:
                raise self._error(
                    "merge patch must list containers in base order",
                    tokens,
                    expected=current_keys,
                    actual=keys,
                )
        working = list(current)
        for position, item in enumerate(patch_items):
            key = item[merge_key]
            match = next(
                (
                    index
                    for index in range(position, len(working))
                    if isinstance(working[index], dict)
                    and working[index].get(merge_key) == key
                ),
                None,
            )
            element = tokens + (str(position),)
            if match is None:
                self._identity(element)
                self._emit("add", element, item)
                working.insert(position, item)
                continue
            existing = working[match]
            merged = _merge_value(item, existing, item_schema)
            if match != position:
                moved = tokens + (str(match),)
                self._identity(moved)
                self._emit("test", moved, existing)
                self._emit("remove", moved)
                del working[match]
                self._identity(element)
                self._emit("add", element, merged)
                working.insert(position, merged)
                continue
            if _json_equal(merged, existing):
                continue
            self._identity(element)
            identity = element + (merge_key,)
            if identity not in self._tested:
                self._emit("test", identity, key)
            self.lower_mapping(item, existing, element, item_schema)
            working[position] = merged


def _lower_merge_patch(
    layer: _PatchLayer, document: Mapping[str, Any], schema: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], ...]:
    assert layer.merge is not None
    definition_key = "%s.%s.%s" % (
        layer.target.group,
        layer.target.version,
        layer.target.kind,
    )
    definition = schema["definitions"].get(definition_key)
    if not isinstance(definition, dict):
        raise ValidationError(
            "merge-patch",
            "OpenAPI schema lacks a definition for %s" % definition_key,
            layer=layer.label,
        )
    spec = document.get("spec")
    if not isinstance(spec, dict):
        raise ValidationError(
            "merge-patch",
            "target document lacks a spec mapping",
            layer=layer.label,
        )
    lowering = _MergeLowering(layer, document)
    lowering.lower_mapping(
        layer.merge["spec"], spec, ("spec",), _schema_property(definition, "spec")
    )
    if not lowering.operations:
        raise ValidationError(
            "merge-patch",
            "merge patch changes nothing in the accumulated document",
            layer=layer.label,
        )
    return tuple(lowering.operations)


def _target_document(documents: Sequence[Any], layer: _PatchLayer) -> Any:
    matches = [
        index
        for index, document in enumerate(documents)
        if _matches_target(document, layer.target)
    ]
    if len(matches) != 1:
        raise ValidationError(
            "patch-target-count",
            "patch target must match exactly one accumulated resource; found %d"
            % len(matches),
            layer=layer.label,
        )
    return documents[matches[0]]


def _lower_merge_layers(
    base_documents: Sequence[Any],
    layers: Sequence[_PatchLayer],
    schema: Optional[Mapping[str, Any]],
) -> Tuple[_PatchLayer, ...]:
    """Replace merge layers with lowered operations against the evolving document.

    Lowering follows the replay order so that a later merge patch resolves
    names against the document that earlier Components already changed. If an
    earlier layer fails to apply, lowering stops and the ordinary replay
    reports that failure.
    """

    if not any(layer.merge is not None for layer in layers):
        return tuple(layers)
    if schema is None:
        raise ValidationError(
            "merge-patch",
            "strategic merge patches require %s as the first root Component"
            % SCHEMA_COMPONENT_REFERENCE,
        )
    working = copy.deepcopy(list(base_documents))
    lowered: List[_PatchLayer] = []
    halted = False
    for layer in layers:
        if layer.merge is not None and not halted:
            target = _target_document(working, layer)
            layer = dataclass_replace(
                layer, operations=_lower_merge_patch(layer, target, schema)
            )
        lowered.append(layer)
        if halted:
            continue
        try:
            target = _target_document(working, layer)
            for op_index, operation in enumerate(layer.operations, start=1):
                _apply_operation(target, layer, op_index, operation)
        except ValidationError:
            halted = True
    return tuple(lowered)


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


def _is_list_slot(token: str) -> bool:
    return token == "-" or bool(re.fullmatch(r"0|[1-9][0-9]*", token))


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


def _source_has_networking_path(layer: _PatchLayer) -> bool:
    normalized = layer.source.resolve().as_posix()
    return bool(
        re.search(
            r"/components/(?:network-interface/(?:agg|disagg)|"
            r"provider-networking/[^/]+/(?:agg|disagg)|"
            r"networking/(?:agg|disagg))(?:/|$)",
            normalized,
        )
    )


def _preceding_env_name_test(
    layer: _PatchLayer, op_index: int, tokens: Tuple[str, ...]
) -> Optional[str]:
    name_path = tokens[:-1] + ("name",)
    observed: Optional[str] = None
    for earlier_index, earlier in enumerate(layer.operations[: op_index - 1], start=1):
        if earlier["op"] != "test":
            continue
        earlier_tokens = _pointer_tokens(
            earlier["path"], layer=layer.label, op_index=earlier_index
        )
        if earlier_tokens == name_path and isinstance(earlier["value"], str):
            observed = earlier["value"]
    return observed


def _physical_network_env_name(
    layer: _PatchLayer,
    op_index: int,
    operation: Mapping[str, Any],
    tokens: Tuple[str, ...],
) -> Optional[str]:
    """Return the base-forbidden physical networking name an operation carries.

    The validator keeps no provider capability list. The only environment names
    it recognizes as networking-owned by value are the physical selections that
    portable bases may never define.
    """
    name: Any = None
    if len(tokens) >= 2 and tokens[-2] == "env" and _is_list_slot(tokens[-1]):
        value = operation.get("value")
        name = value.get("name") if isinstance(value, dict) else None
    elif (
        len(tokens) >= 3
        and tokens[-3] == "env"
        and re.fullmatch(r"0|[1-9][0-9]*", tokens[-2])
        and tokens[-1] == "value"
    ):
        name = _preceding_env_name_test(layer, op_index, tokens)
    if isinstance(name, str) and name in FORBIDDEN_BASE_ENV_NAMES:
        return name
    return None


_VOLUME_MOUNT_OPTIONAL_FIELDS: Mapping[str, type] = {
    "readOnly": bool,
    "mountPropagation": str,
    "subPath": str,
    "subPathExpr": str,
}


def _is_volume_mount_shape(value: Any) -> bool:
    """Accept a VolumeMount with its optional Kubernetes fields, checked by type."""

    if not isinstance(value, dict) or not {"name", "mountPath"} <= set(value):
        return False
    if not set(value) <= {"name", "mountPath", *_VOLUME_MOUNT_OPTIONAL_FIELDS}:
        return False
    if not (isinstance(value["name"], str) and value["name"]):
        return False
    if not (isinstance(value["mountPath"], str) and value["mountPath"].startswith("/")):
        return False
    for field, field_type in _VOLUME_MOUNT_OPTIONAL_FIELDS.items():
        if field not in value:
            continue
        if field_type is bool:
            if not isinstance(value[field], bool):
                return False
        elif not (isinstance(value[field], str) and value[field]):
            return False
    return True


def _network_operation_kind(
    layer: _PatchLayer,
    op_index: int,
    operation: Mapping[str, Any],
) -> Optional[str]:
    """Classify the shape of a worker networking operation.

    Classification is structural. Any annotation key, environment name,
    extended-resource key, or host path may be carried by a networking
    Component; the provider owns the concrete values.
    """
    tokens = _pointer_tokens(operation["path"], layer=layer.label, op_index=op_index)
    if len(tokens) < 4 or tokens[:2] != ("spec", "components"):
        return None
    component_index = tokens[2]
    pod_prefix = ("spec", "components", component_index, "podTemplate")
    main_prefix = pod_prefix + ("spec", "containers", "0")

    if tokens in (pod_prefix + ("metadata",), pod_prefix + ("metadata", "annotations")):
        value = operation.get("value")
        if tokens[-1] == "metadata":
            annotations = (
                value.get("annotations")
                if isinstance(value, dict) and set(value) == {"annotations"}
                else None
            )
        else:
            annotations = value
        if (
            isinstance(annotations, dict)
            and annotations
            and all(
                isinstance(key, str) and key and isinstance(item, str)
                for key, item in annotations.items()
            )
        ):
            return "annotation"
        return None

    if len(tokens) == 7 and tokens[:6] == pod_prefix + (
        "metadata",
        "annotations",
    ):
        if tokens[6] and isinstance(operation.get("value"), str):
            return "annotation"
        return None

    if tokens[:-1] == main_prefix + ("env",) and _is_list_slot(tokens[-1]):
        value = operation.get("value")
        if (
            isinstance(value, dict)
            and set(value) in ({"name", "value"}, {"name", "valueFrom"})
            and isinstance(value.get("name"), str)
            and value["name"]
            and (
                isinstance(value.get("value"), str)
                or isinstance(value.get("valueFrom"), dict)
            )
        ):
            return "env-append"
        return None

    if (
        len(tokens) == len(main_prefix) + 3
        and tokens[: len(main_prefix)] == main_prefix
        and tokens[len(main_prefix)] == "env"
        and re.fullmatch(r"0|[1-9][0-9]*", tokens[-2])
        and tokens[-1] == "value"
        and _preceding_env_name_test(layer, op_index, tokens) is not None
    ):
        return "env-override"

    resource_prefix = main_prefix + ("resources",)
    if (
        len(tokens) >= len(resource_prefix) + 1
        and tokens[: len(resource_prefix)] == resource_prefix
        and tokens[len(resource_prefix)] in ("requests", "limits")
    ):
        if len(tokens) == len(resource_prefix) + 2 and "/" not in tokens[-1]:
            # cpu, memory, ephemeral-storage, hugepages-*: core resources are
            # portable recipe fields, not networking.
            return None
        if (
            len(tokens) != len(resource_prefix) + 2
            or tokens[-1].startswith("/")
            or tokens[-1].endswith("/")
        ):
            return "resource-invalid"
        return "resource"

    if tokens[:-1] == main_prefix + ("volumeMounts",) and _is_list_slot(tokens[-1]):
        if _is_volume_mount_shape(operation.get("value")):
            return "host-mount"
        return None

    if tokens[:-1] == pod_prefix + ("spec", "volumes") and _is_list_slot(tokens[-1]):
        value = operation.get("value")
        host_path = value.get("hostPath") if isinstance(value, dict) else None
        if (
            isinstance(value, dict)
            and set(value) == {"name", "hostPath"}
            and isinstance(value.get("name"), str)
            and value["name"]
            and isinstance(host_path, dict)
            and set(host_path) <= {"path", "type"}
            and isinstance(host_path.get("path"), str)
            and host_path["path"].startswith("/")
        ):
            return "host-volume"
        return None
    return None


_NETWORK_SHAPE_KINDS = frozenset(
    {
        "annotation",
        "env-append",
        "resource",
        "resource-invalid",
        "host-mount",
        "host-volume",
    }
)
_KEYED_LIST_FIELDS = frozenset({"env", "volumeMounts", "volumes"})


def _canonical_worker_indices(topology: str) -> frozenset:
    """Component positions that hold canonical workers for a topology.

    Aggregate bases carry ``Frontend`` and ``Worker``; disaggregated bases carry
    ``Frontend``, ``PrefillWorker``, and ``DecodeWorker``. Optional components
    follow the canonical prefix and are never networking targets.
    """

    return frozenset({"1"}) if topology == "agg" else frozenset({"1", "2"})


def _is_canonical_worker_path(
    tokens: Tuple[str, ...], worker_indices: frozenset
) -> bool:
    return (
        len(tokens) >= 3
        and tokens[:2] == ("spec", "components")
        and tokens[2] in worker_indices
    )


def _root_adds_network_shape(
    layer: _PatchLayer,
    operation: Mapping[str, Any],
    tokens: Tuple[str, ...],
    kind: Optional[str],
    moved_entries: Set[Tuple[str, str]],
    worker_indices: frozenset,
) -> bool:
    """Recognize a root patch that adds networking-shaped fields to a worker.

    Root patches may replace or move values the canonical workers already
    carry, such as the framework hooks, but adding annotations, environment
    entries, extended resources, mounts, or volumes to a canonical worker is
    the networking Component's job. Moves are recognized by provenance: the
    lowering tests and removes the existing entry before re-adding it.
    """

    if (
        layer.root_component is not None
        or operation["op"] != "add"
        or kind not in _NETWORK_SHAPE_KINDS
        or not _is_canonical_worker_path(tokens, worker_indices)
    ):
        return False
    value = operation.get("value")
    if (
        kind in ("env-append", "host-mount", "host-volume")
        and isinstance(value, dict)
        and (tokens[2], value.get("name")) in moved_entries
    ):
        return False
    return True


def _root_replaces_component_value(
    layer: _PatchLayer,
    op_index: int,
    tokens: Tuple[str, ...],
    kind: Optional[str],
    env_entries: Set[Tuple[str, str]],
    annotation_keys: Set[Tuple[str, str]],
    resource_paths: Set[Tuple[str, str, str]],
) -> bool:
    """Return whether a root replace targets a value the networking Component added.

    Root patches may retune what the selected networking Component introduced,
    such as an RDMA resource quantity or a socket interface, but portable worker
    values such as ``nvidia.com/gpu`` belong to the recipe base and stay out of
    reach of case-local patches.
    """

    if len(tokens) < 3:
        return False
    component_index = tokens[2]
    if kind == "resource":
        return (component_index, tokens[-2], tokens[-1]) in resource_paths
    if kind == "annotation" and len(tokens) == 7:
        return (component_index, tokens[6]) in annotation_keys
    if kind == "env-override":
        name = _preceding_env_name_test(layer, op_index, tokens)
        return name is not None and (component_index, name) in env_entries
    return False


def _networking_error(
    layer: _PatchLayer,
    op_index: int,
    operation: Mapping[str, Any],
    message: str,
) -> ValidationError:
    return ValidationError(
        "networking-delta",
        message,
        layer=layer.label,
        op_index=op_index,
        path=operation["path"],
    )


def _validate_networking_contract(
    layers: Sequence[_PatchLayer],
    root_components: Sequence[_RootComponent],
    topology: str,
) -> None:
    worker_indices = _canonical_worker_indices(topology)
    # _validate_root_contract owns the networking slot count and ordering rules.
    network_root = next(
        (root for root in root_components if root.concern in _NETWORK_ROOT_CONCERNS),
        None,
    )

    resource_values: Dict[Tuple[str, str, str], Any] = {}
    mounts: Dict[Tuple[str, str], int] = {}
    volumes: Dict[Tuple[str, str], int] = {}
    component_env_entries: Set[Tuple[str, str]] = set()
    component_annotation_keys: Set[Tuple[str, str]] = set()
    component_resource_paths: Set[Tuple[str, str, str]] = set()
    for component_layer in layers:
        if not (
            component_layer.root_component is not None
            and component_layer.root_component.concern in _NETWORK_ROOT_CONCERNS
        ):
            continue
        for component_op_index, component_operation in enumerate(
            component_layer.operations, start=1
        ):
            if component_operation["op"] != "add":
                continue
            component_tokens = _pointer_tokens(
                component_operation["path"],
                layer=component_layer.label,
                op_index=component_op_index,
            )
            component_value = component_operation.get("value")
            if len(component_tokens) < 3:
                continue
            component_index = component_tokens[2]
            component_kind = _network_operation_kind(
                component_layer, component_op_index, component_operation
            )
            if component_kind == "env-append" and isinstance(component_value, dict):
                component_env_entries.add((component_index, component_value["name"]))
            elif component_kind == "resource":
                component_resource_paths.add(
                    (component_index, component_tokens[-2], component_tokens[-1])
                )
            elif component_kind == "annotation" and isinstance(component_value, dict):
                if component_tokens[-1] == "metadata":
                    added = component_value.get("annotations", {})
                else:
                    added = component_value
                component_annotation_keys.update(
                    (component_index, key) for key in added
                )
            elif component_kind == "annotation" and len(component_tokens) == 7:
                component_annotation_keys.add((component_index, component_tokens[6]))

    for layer in layers:
        owner_is_network = bool(
            layer.root_component is not None
            and layer.root_component.concern in _NETWORK_ROOT_CONCERNS
        )
        source_has_networking_path = _source_has_networking_path(layer)
        tested_entries: Dict[Tuple[str, ...], Tuple[str, str]] = {}
        moved_entries: Set[Tuple[str, str]] = set()
        for op_index, operation in enumerate(layer.operations, start=1):
            tokens = _pointer_tokens(
                operation["path"], layer=layer.label, op_index=op_index
            )
            if operation["op"] == "test":
                tested = operation.get("value")
                if (
                    len(tokens) >= 4
                    and tokens[:2] == ("spec", "components")
                    and tokens[-2] in _KEYED_LIST_FIELDS
                    and _is_list_slot(tokens[-1])
                    and isinstance(tested, dict)
                    and isinstance(tested.get("name"), str)
                ):
                    tested_entries[tokens] = (tokens[2], tested["name"])
                continue
            if operation["op"] == "remove" and tokens in tested_entries:
                moved_entries.add(tested_entries[tokens])
            kind = _network_operation_kind(layer, op_index, operation)
            physical_name = _physical_network_env_name(
                layer, op_index, operation, tokens
            )
            # A networking delta is recognized by its owner, its source path, a
            # base-forbidden physical name, an extended-resource shape, or a root
            # patch adding networking-shaped fields to a canonical worker; never
            # by a provider-specific allowlist. Optional components after the
            # canonical workers keep the documented case-local patch path.
            resource_signal = kind in {"resource", "resource-invalid"} and (
                layer.root_component is not None
                or _is_canonical_worker_path(tokens, worker_indices)
            )
            is_network_delta = (
                owner_is_network
                or source_has_networking_path
                or physical_name is not None
                or resource_signal
                or _root_adds_network_shape(
                    layer, operation, tokens, kind, moved_entries, worker_indices
                )
            )
            if not is_network_delta:
                continue

            if (
                len(tokens) >= 3
                and tokens[:2] == ("spec", "components")
                and tokens[2] not in worker_indices
            ):
                raise _networking_error(
                    layer,
                    op_index,
                    operation,
                    "networking may mutate only canonical worker positions",
                )
            if kind == "resource-invalid":
                raise ValidationError(
                    "networking-resource-pair",
                    "network resource paths require one RFC 6901-encoded extended-resource key",
                    layer=layer.label,
                    op_index=op_index,
                    path=operation["path"],
                )

            if owner_is_network:
                if kind is None:
                    raise _networking_error(
                        layer,
                        op_index,
                        operation,
                        "networking Component mutates a field outside its concern",
                    )
                if operation["op"] != "add" or kind == "env-override":
                    raise _networking_error(
                        layer,
                        op_index,
                        operation,
                        "networking Components may only add worker networking fields",
                    )
            elif layer.root_component is not None:
                raise _networking_error(
                    layer,
                    op_index,
                    operation,
                    "networking delta is nested under an unrelated root concern",
                )
            elif operation["op"] == "replace":
                if not _root_replaces_component_value(
                    layer,
                    op_index,
                    tokens,
                    kind,
                    component_env_entries,
                    component_annotation_keys,
                    component_resource_paths,
                ):
                    raise _networking_error(
                        layer,
                        op_index,
                        operation,
                        "root patches may only replace networking values that the "
                        "selected networking Component adds; portable worker "
                        "resources belong to the recipe base",
                    )
            else:
                if network_root is None:
                    raise _networking_error(
                        layer,
                        op_index,
                        operation,
                        "root patch cannot serve as the networking slot; select a "
                        "networking Component to add worker networking fields",
                    )
                root_append_name = (
                    operation.get("value", {}).get("name")
                    if isinstance(operation.get("value"), dict)
                    else None
                )
                repeats_component_env = (
                    kind == "env-append"
                    and operation["op"] == "add"
                    and (tokens[2], root_append_name) in component_env_entries
                )
                if not repeats_component_env:
                    raise _networking_error(
                        layer,
                        op_index,
                        operation,
                        "root networking patches may only replace values the "
                        "selected networking Component adds",
                    )

            if kind == "resource":
                component_index = tokens[2]
                scope = tokens[-2]
                resource_key = tokens[-1]
                resource_values[(component_index, resource_key, scope)] = operation[
                    "value"
                ]
            elif kind in {"host-mount", "host-volume"}:
                component_index = tokens[2]
                value = operation["value"]
                key = (component_index, value["name"])
                collection = mounts if kind == "host-mount" else volumes
                collection[key] = collection.get(key, 0) + 1

    resource_pairs = {(index, key) for index, key, _ in resource_values}
    for component_index, resource_key in sorted(resource_pairs):
        request = resource_values.get((component_index, resource_key, "requests"))
        limit = resource_values.get((component_index, resource_key, "limits"))
        if request is None or limit is None or not _json_equal(request, limit):
            raise ValidationError(
                "networking-resource-pair",
                "network resource request and limit keys and quantities must match",
                path=(
                    "/spec/components/%s/podTemplate/spec/containers/0/resources"
                    % component_index
                ),
                expected=request,
                actual=limit,
            )

    if mounts != volumes:
        raise ValidationError(
            "networking-delta",
            "networking host mount and volume blocks must be complete and name-matched",
            expected=mounts,
            actual=volumes,
        )


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
            if len(tokens) >= 2 and tokens[-2] == "env" and _is_list_slot(tokens[-1]):
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


def _reject_duplicate_env(
    parent: Sequence[Any],
    value: Any,
    *,
    layer: _PatchLayer,
    op_index: int,
    path: str,
) -> None:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("name"), str)
        or not value["name"]
    ):
        raise ValidationError(
            "unsupported-manifest",
            "env add requires a value with a non-empty string name",
            layer=layer.label,
            op_index=op_index,
            path=path,
        )
    duplicate_index = next(
        (
            index
            for index, entry in enumerate(parent)
            if isinstance(entry, dict) and entry.get("name") == value["name"]
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
            if len(tokens) >= 2 and tokens[-2] == "env":
                _reject_duplicate_env(
                    parent, value, layer=layer, op_index=op_index, path=path
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
    topology = _validate_canonical_components(base_documents[dgd_index])
    layers, root_components = _collect_layers(base, kustomization)
    schema = _load_schema(root_components)
    layers = _lower_merge_layers(base_documents, layers, schema)
    for layer in layers:
        _validate_guards(layer)
    _validate_base_ownership(base_documents, dgd_index, layers)
    _validate_networking_contract(layers, root_components, topology)

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
        description=(
            "Validate a beta recipe Kustomization by sequential patch replay; "
            "strategic merge patches are lowered to guarded JSON 6902 operations."
        )
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
