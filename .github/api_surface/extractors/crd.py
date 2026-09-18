# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRD surface extractor.

Reads CustomResourceDefinition manifests from
``<repo>/deploy/operator/config/crd/bases/*.yaml`` and emits one
:class:`SurfaceSymbol` per ``(Kind, version)`` plus one per
``(Kind, version, field dotpath)``. Symbol ids follow the grammar pinned in
:mod:`api_surface.models`::

    crd:<Kind>@<version>
    crd:<Kind>@<version>.<dotpath>

Field signatures include the OpenAPI type, requiredness, and canonical
compatibility constraints such as enum values, formats, ranges, patterns, and
array item shape. Array items contribute ``[]`` to the dotpath;
``additionalProperties`` maps stop traversal so the emitted surface stays
bounded.

Dynamo-owned vs inherited Kubernetes schema
-------------------------------------------
controller-gen inlines the full openAPI schema of every embedded upstream
Kubernetes core type (a ``PodSpec``, ``Container``, ``Affinity``, ``ObjectMeta``,
...) directly into the CRD, because CRDs cannot ``$ref`` external schemas. The
result is that a Dynamo CRD's field surface is dominated (≈86%) by inherited k8s
schema that changes with the Kubernetes version, not with Dynamo's own API. We
classify each field's ``origin`` so the count is meaningful:

- ``origin="dynamo"`` -- fields the operator defines and owns the schema of.
- ``origin="k8s"`` -- the *boundary* field where the schema crosses into an
  embedded upstream type (its property name is in :data:`K8S_EMBED_FIELDS`).

At a k8s boundary we emit the boundary field (it *is* a Dynamo-exposed knob --
"you may set ``affinity`` here") tagged ``origin="k8s"`` and then stop
descending, so the thousands of inherited sub-fields never enter the surface or
the ledger. Tracking the boundary, not its upstream subtree, keeps Kubernetes
version bumps from masquerading as Dynamo API changes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from api_surface.models import SurfaceSymbol
from api_surface.results import OperationResult, OpError

SURFACE = "crd"

_CRD_REL_PATH = "deploy/operator/config/crd/bases"
_OPERATION = "api_surface.extract.crd"

_CONTRACT_KEYS = (
    "type",
    "format",
    "enum",
    "const",
    "nullable",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "x-kubernetes-int-or-string",
    "x-kubernetes-list-type",
    "x-kubernetes-map-type",
)

# Property names that mark the seam into an embedded upstream Kubernetes core
# type (PodSpec / Container / scheduling / ObjectMeta). Reaching one of these
# means everything below is inherited k8s schema, so we tag the boundary
# ``origin="k8s"`` and prune the subtree. These names are stable k8s API field
# names; a Dynamo field that happens to share one (e.g. ``resources`` is k8s
# ``ResourceRequirements`` in these CRDs) is genuinely the upstream type.
K8S_EMBED_FIELDS = frozenset(
    {
        "affinity",
        "containers",
        "dnsConfig",
        "env",
        "envFrom",
        "ephemeralContainers",
        "hostAliases",
        "imagePullSecrets",
        "initContainers",
        "lifecycle",
        "livenessProbe",
        "metadata",
        "nodeSelector",
        "podSecurityContext",
        "ports",
        "readinessProbe",
        "resources",
        "securityContext",
        "startupProbe",
        "tolerations",
        "topologySpreadConstraints",
        "volumeMounts",
        "volumes",
        "extraPodSpec",
    }
)


def _is_k8s_embed(name: str, type_str: str) -> bool:
    """True when ``name`` is the boundary into an embedded upstream k8s type.

    Keyed on the property name (the stable k8s API field name) and gated to
    composite nodes, since only object/array embeds carry the inherited subtree
    we want to prune. Opaque ``x-kubernetes-preserve-unknown-fields`` nodes are
    intentionally *not* treated as k8s: they are Dynamo-owned free-form fields
    (raw config blobs, nested DGD specs) that have no properties to prune.
    """
    return name in K8S_EMBED_FIELDS and type_str in {"object", "array"}


def extract(repo_path: Path, release: str) -> OperationResult:
    """Extract the CRD surface from ``repo_path``.

    Walks every ``*.yaml`` under ``deploy/operator/config/crd/bases``, parses
    each document with :func:`yaml.safe_load_all`, and emits one version-level
    symbol per ``(Kind, version)`` plus one field symbol per leaf or branch in
    the openAPI v3 schema.

    The ``release`` argument is accepted for protocol parity (``SurfaceExtractor``
    is uniform across surfaces) but is not consulted; the snapshot is captured
    against whatever ref the caller has on disk.

    Returns an :class:`OperationResult` with ``data["symbols"]`` populated and
    ``metadata = {"surface": "crd", "covered": <bool>}``. A missing CRD bases
    directory yields ``covered=False`` and a single :class:`OpError` rather
    than raising, so the diff engine can record a coverage gap.
    """
    result = OperationResult(metadata={"surface": SURFACE, "covered": False})
    result.data["symbols"] = []

    crd_dir = repo_path / _CRD_REL_PATH
    if not crd_dir.is_dir():
        result.errors.append(
            OpError(
                operation=_OPERATION,
                message=f"CRD directory not found: {crd_dir}",
                details={
                    "repo_path": str(repo_path),
                    "expected": _CRD_REL_PATH,
                    "release": release,
                },
            )
        )
        return result

    symbols: list[dict[str, Any]] = []
    for yaml_path in sorted(crd_dir.glob("*.yaml")):
        try:
            docs = list(yaml.safe_load_all(yaml_path.read_text()))
        except yaml.YAMLError as exc:
            result.errors.append(
                OpError(
                    operation=_OPERATION,
                    message=f"YAML parse failed: {yaml_path.name}",
                    details={"path": str(yaml_path), "error": str(exc)},
                )
            )
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("kind") != "CustomResourceDefinition":
                continue
            symbols.extend(_emit_for_crd(doc))

    result.data["symbols"] = symbols
    result.metadata["covered"] = True
    return result


def _emit_for_crd(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield serialized symbols for one parsed CRD document."""
    spec = doc.get("spec") or {}
    if not isinstance(spec, dict):
        return []
    names = spec.get("names") or {}
    kind = names.get("kind") if isinstance(names, dict) else None
    if not isinstance(kind, str) or not kind:
        return []

    out: list[dict[str, Any]] = []
    for version in spec.get("versions") or []:
        if not isinstance(version, dict):
            continue
        version_name = version.get("name")
        if not isinstance(version_name, str) or not version_name:
            continue
        if not bool(version.get("served", False)):
            continue
        deprecated = bool(version.get("deprecated", False))
        deprecation_warning = str(version.get("deprecationWarning", ""))
        version_meta = {
            "version": version_name,
            "served": bool(version.get("served", False)),
            "storage": bool(version.get("storage", False)),
            "deprecated": deprecated,
            "deprecation_warning": deprecation_warning,
        }
        out.append(
            SurfaceSymbol(
                surface=SURFACE,
                kind="version",
                id=f"crd:{kind}@{version_name}",
                signature="",
                deprecated=deprecated,
                deprecated_note=deprecation_warning,
                metadata={**version_meta, "origin": "dynamo"},
            ).to_dict()
        )
        schema_root = version.get("schema") or {}
        schema = (
            schema_root.get("openAPIV3Schema")
            if isinstance(schema_root, dict)
            else None
        )
        if not isinstance(schema, dict):
            continue
        for sym in _walk_root(
            schema, kind=kind, version=version_name, version_meta=version_meta
        ):
            out.append(sym.to_dict())
    return out


def _walk_root(
    schema: dict[str, Any],
    *,
    kind: str,
    version: str,
    version_meta: dict[str, Any],
) -> Iterator[SurfaceSymbol]:
    """Walk the top-level openAPIV3Schema, emitting symbols for its properties.

    The root object itself is not emitted as a field; the version-level symbol
    already represents it.
    """
    if schema.get("type") != "object":
        return
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required_set = {r for r in (schema.get("required") or []) if isinstance(r, str)}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        yield from _walk_field(
            prop,
            kind=kind,
            version=version,
            name=name,
            dotpath=name,
            is_required=name in required_set,
            version_meta=version_meta,
        )


def _walk_field(
    node: dict[str, Any],
    *,
    kind: str,
    version: str,
    name: str,
    dotpath: str,
    is_required: bool,
    version_meta: dict[str, Any],
) -> Iterator[SurfaceSymbol]:
    """Emit the field for ``node`` and recurse into composite Dynamo-owned descendants.

    ``name`` is the field's own property name (the dotpath leaf), used to detect
    the seam into an embedded upstream Kubernetes type. At such a seam the field
    is emitted ``origin="k8s"`` and the inherited subtree is pruned.
    """
    type_str = node.get("type") or ""
    is_embed = _is_k8s_embed(name, type_str)
    base_signature = (
        f"{type_str} required"
        if is_required and type_str
        else ("required" if is_required else type_str)
    )
    constraints = _schema_constraints(node)
    constraints.pop("type", None)
    suffix = (
        " " + json.dumps(constraints, sort_keys=True, separators=(",", ":"))
        if constraints
        else ""
    )
    signature = base_signature + suffix

    yield SurfaceSymbol(
        surface=SURFACE,
        kind="field",
        id=f"crd:{kind}@{version}.{dotpath}",
        signature=signature,
        deprecated=bool(version_meta["deprecated"]),
        deprecated_note=str(version_meta["deprecation_warning"]),
        metadata={**version_meta, "origin": "k8s" if is_embed else "dynamo"},
    )

    if is_embed:
        return  # inherited k8s subtree -- tracked at the boundary, not below it

    if type_str == "object":
        properties = node.get("properties")
        if isinstance(properties, dict):
            req_set = {r for r in (node.get("required") or []) if isinstance(r, str)}
            for child_name, child_node in properties.items():
                if not isinstance(child_node, dict):
                    continue
                yield from _walk_field(
                    child_node,
                    kind=kind,
                    version=version,
                    name=child_name,
                    dotpath=f"{dotpath}.{child_name}",
                    is_required=child_name in req_set,
                    version_meta=version_meta,
                )
        # additionalProperties maps: stop traversal to keep the surface bounded.
    elif type_str == "array":
        items = node.get("items")
        if isinstance(items, dict):
            # Array items are not a named field; pass name="" so boundary
            # detection keys on the items' own child property names.
            yield from _walk_field(
                items,
                kind=kind,
                version=version,
                name="",
                dotpath=f"{dotpath}[]",
                is_required=False,
                version_meta=version_meta,
            )


def _schema_constraints(node: dict[str, Any]) -> dict[str, Any]:
    """Project one CRD schema node to compatibility-relevant constraints."""
    contract = {key: node[key] for key in _CONTRACT_KEYS if key in node}
    items = node.get("items")
    if isinstance(items, dict):
        contract["items"] = _schema_constraints(items)
    additional = node.get("additionalProperties")
    if isinstance(additional, dict):
        contract["additionalProperties"] = _schema_constraints(additional)
    for key in ("allOf", "anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            contract[key] = [
                _schema_constraints(item) for item in variants if isinstance(item, dict)
            ]
    return contract
