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
"""Convert nvidia.com CRD manifests between served apiVersions via the
operator's conversion webhook (no conversion logic is duplicated here)."""

import argparse
import copy
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import yaml

GROUP = "nvidia.com"
DEFAULT_TARGET = "nvidia.com/v1beta1"
KUBECTL_TIMEOUT = 30  # seconds


class ConversionError(Exception):
    """Raised when the conversion webhook reports failure or returns junk."""


_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"


class _NoDatesSafeLoader(yaml.SafeLoader):
    """SafeLoader that keeps date/timestamp-like scalars as plain strings.

    PyYAML's default implicit resolver turns values like ``2026-06-15`` or
    ``2026-06-15T00:00:00Z`` into ``datetime``/``date`` objects, which are not
    JSON-serializable and would break the webhook request. Disabling the
    timestamp resolver preserves them as strings.
    """


_NoDatesSafeLoader.yaml_implicit_resolvers = {
    ch: [(tag, regexp) for tag, regexp in resolvers if tag != _TIMESTAMP_TAG]
    for ch, resolvers in _NoDatesSafeLoader.yaml_implicit_resolvers.items()
}


def load_docs(text: str) -> list:
    """Parse a multi-document YAML string into a list of non-empty dicts."""
    return [d for d in yaml.load_all(text, Loader=_NoDatesSafeLoader) if d is not None]


def is_convertible(doc: dict) -> bool:
    """True if the document's apiVersion is in the nvidia.com API group
    (must be group/version form, e.g. nvidia.com/v1alpha1)."""
    api_version = doc.get("apiVersion", "")
    parts = api_version.split("/", 1)
    return len(parts) == 2 and parts[0] == GROUP


def build_conversion_review(objects: list, desired_api_version: str, uid: str) -> dict:
    """Build an apiextensions.k8s.io/v1 ConversionReview request."""
    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "ConversionReview",
        "request": {
            "uid": uid,
            "desiredAPIVersion": desired_api_version,
            "objects": objects,
        },
    }


def parse_conversion_response(review_response: dict, expected_uid: str) -> list:
    """Validate a ConversionReview response and return its convertedObjects."""
    resp = review_response.get("response") or {}
    if not isinstance(resp, dict):
        raise ConversionError(
            f"conversion response 'response' is not an object: {type(resp).__name__}"
        )
    result = resp.get("result") or {}
    if not isinstance(result, dict):
        raise ConversionError(
            f"conversion response 'result' is not an object: {type(result).__name__}"
        )
    if result.get("status") != "Success":
        raise ConversionError(
            f"conversion webhook failed: {result.get('message', 'unknown error')}"
        )
    if resp.get("uid") != expected_uid:
        raise ConversionError(
            f"conversion response uid mismatch: "
            f"expected {expected_uid!r}, got {resp.get('uid')!r}"
        )
    converted = resp.get("convertedObjects") or []
    if not isinstance(converted, list):
        raise ConversionError(
            f"conversion response 'convertedObjects' is not a list: "
            f"{type(converted).__name__}"
        )
    return converted


SERVER_METADATA_FIELDS = (
    "uid",
    "resourceVersion",
    "generation",
    "creationTimestamp",
    "managedFields",
    "selfLink",
)

_TOP_LEVEL_ORDER = ("apiVersion", "kind", "metadata", "spec")


def clean_for_authoring(obj: dict) -> dict:
    """Strip server-managed metadata and status so the converted object reads
    like an authorable manifest.

    Annotations are left untouched on purpose: the conversion webhook uses them
    as hub/spoke preservation annotations (part of the conversion contract), so
    dropping them could change the resource's meaning.
    """
    obj = copy.deepcopy(obj)
    obj.pop("status", None)

    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for field in SERVER_METADATA_FIELDS:
            metadata.pop(field, None)

    ordered = {k: obj[k] for k in _TOP_LEVEL_ORDER if k in obj}
    for k, v in obj.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def convert_docs(docs: list, target: str = DEFAULT_TARGET, webhook_fn=None) -> list:
    """Convert nvidia.com docs to ``target`` via webhook_fn, preserving the
    original document order. Non-nvidia.com docs and docs already at ``target``
    pass through unchanged.

    webhook_fn(review: dict, kind: str) -> ConversionReview-response dict.
    Defaults to the real cluster transport.
    """
    if webhook_fn is None:
        webhook_fn = default_webhook_fn

    results = [None] * len(docs)
    indices_by_kind: dict = {}
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ConversionError(
                f"document at index {i} is not a mapping: {type(doc).__name__}"
            )
        if is_convertible(doc) and doc.get("apiVersion") != target:
            if "kind" not in doc:
                raise ConversionError(
                    f"document at index {i} has an nvidia.com apiVersion "
                    "but no 'kind'"
                )
            indices_by_kind.setdefault(doc["kind"], []).append(i)
        else:
            results[i] = doc

    for kind, indices in indices_by_kind.items():
        uid = str(uuid.uuid4())
        objects = [docs[i] for i in indices]
        review = build_conversion_review(objects, target, uid)
        converted = parse_conversion_response(webhook_fn(review, kind), uid)
        if len(converted) != len(indices):
            raise ConversionError(
                f"webhook returned {len(converted)} objects for kind {kind}, "
                f"expected {len(indices)}"
            )
        for idx, obj in zip(indices, converted):
            results[idx] = clean_for_authoring(obj)

    return results


def _kubectl(args: list) -> str:
    """Run kubectl and return stdout, raising ConversionError on failure."""
    try:
        proc = subprocess.run(
            ["kubectl", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=KUBECTL_TIMEOUT,
        )
    except FileNotFoundError as exc:
        raise ConversionError("kubectl not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(
            f"kubectl {' '.join(args)} timed out after {KUBECTL_TIMEOUT}s"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ConversionError(
            f"kubectl {' '.join(args)} failed: {exc.stderr.strip()}"
        ) from exc
    return proc.stdout


def discover_conversion_service(kind: str, crd_list: dict = None) -> dict:
    """Find the conversion webhook Service for an nvidia.com CRD by kind.

    crd_list: parsed `kubectl get crd -o json` output; fetched live if None.
    """
    if crd_list is None:
        crd_list = json.loads(_kubectl(["get", "crd", "-o", "json"]))

    for crd in crd_list.get("items", []):
        spec = crd.get("spec", {})
        if spec.get("group") != GROUP or spec.get("names", {}).get("kind") != kind:
            continue
        client_config = (
            spec.get("conversion", {}).get("webhook", {}).get("clientConfig", {})
        )
        service = client_config.get("service")
        if not service:
            continue  # matching kind but no webhook service configured; keep looking
        return {
            "name": service["name"],
            "namespace": service["namespace"],
            "port": service.get("port", 443),
            "path": service.get("path", "/convert"),
        }
    raise ConversionError(
        f"no conversion webhook service configured for {kind}.{GROUP}"
    )


def call_webhook(review: dict, service: dict) -> dict:
    """POST a ConversionReview to the webhook via the API server service proxy."""
    path = service["path"].lstrip("/")
    proxy_path = (
        f"/api/v1/namespaces/{service['namespace']}"
        f"/services/https:{service['name']}:{service['port']}/proxy/{path}"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(review, tmp)
    try:
        out = _kubectl(["create", "--raw", proxy_path, "-f", tmp.name])
    finally:
        os.unlink(tmp.name)
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"webhook returned invalid JSON: {exc}") from exc


def default_webhook_fn(review: dict, kind: str) -> dict:
    """Real transport: discover the webhook service and POST via API proxy."""
    service = discover_conversion_service(kind)
    return call_webhook(review, service)


# Strings a YAML 1.1 parser (go-yaml, used by kubectl and the apiserver) reads
# as bool/null. PyYAML's emitter does not quote all of these on output (notably
# y/n/Y/N), so an unquoted value would be re-parsed as a bool and violate the
# "value must be a string" contract for e.g. container env vars. Force-quote them.
_YAML11_AMBIGUOUS = frozenset(
    {
        "y",
        "Y",
        "yes",
        "Yes",
        "YES",
        "n",
        "N",
        "no",
        "No",
        "NO",
        "true",
        "True",
        "TRUE",
        "false",
        "False",
        "FALSE",
        "on",
        "On",
        "ON",
        "off",
        "Off",
        "OFF",
        "null",
        "Null",
        "NULL",
        "~",
    }
)


class _ManifestDumper(yaml.SafeDumper):
    """SafeDumper that force-quotes strings a YAML 1.1 consumer would otherwise
    misread as bool/null, so converted manifests round-trip through kubectl."""


def _represent_str(dumper, data):
    """Represent a str, single-quoting values a YAML 1.1 reader would misread as bool/null."""
    if data in _YAML11_AMBIGUOUS:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_ManifestDumper.add_representer(str, _represent_str)


def dump_docs(docs: list) -> str:
    """Serialize docs to a multi-document YAML string preserving key order.

    Uses a dumper that force-quotes YAML 1.1 bool/null-ambiguous strings (e.g.
    env ``value: "y"``) so kubectl/the apiserver, which parse with a broader
    bool token set than PyYAML, do not re-read them as booleans.
    """
    return yaml.dump_all(
        docs, Dumper=_ManifestDumper, sort_keys=False, default_flow_style=False
    )


def main(argv: list = None) -> int:
    """CLI entry point: parse args, convert the input manifest, write the result."""
    parser = argparse.ArgumentParser(
        description="Convert nvidia.com CRD manifests between served "
        "apiVersions using the operator's conversion webhook."
    )
    parser.add_argument("-i", "--input", required=True, help="input manifest YAML")
    parser.add_argument("-o", "--output", required=True, help="output manifest YAML")
    parser.add_argument(
        "--to",
        default=DEFAULT_TARGET,
        help=f"target apiVersion (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args(argv)

    try:
        docs = load_docs(Path(args.input).read_text(encoding="utf-8"))
        converted = convert_docs(docs, target=args.to)
        Path(args.output).write_text(dump_docs(converted), encoding="utf-8")
    except (ConversionError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))  # prints "error: <msg>" and exits 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
