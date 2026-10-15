# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only external Grove compatibility check, executed by the Helm hook Job."""

import hashlib
import json
import os
import re
import ssl
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class Incompatible(Exception):
    pass


def version_key(value):
    """Compare SemVer, including numeric prerelease identifiers such as alpha.15."""
    match = re.fullmatch(
        r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
        r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
        value,
    )
    if not match:
        raise Incompatible(
            f"Unverifiable Grove version {value!r}; use a release-tagged controller image."
        )
    prerelease = []
    for part in (match[4] or "").split("."):
        if part.isdigit():
            if len(part) > 1 and part.startswith("0"):
                raise Incompatible(f"Invalid Grove prerelease version {value!r}.")
            prerelease.append((0, int(part)))
        else:
            prerelease.append((1, part))
    return tuple(map(int, match.group(1, 2, 3))), tuple(prerelease) if match[4] else (
        (2, ""),
    )


def image_version(image):
    # A digest without a release tag cannot establish the controller release.
    name = image.split("@", 1)[0].rsplit("/", 1)[-1]
    if ":" not in name:
        raise Incompatible(
            f"Unverifiable Grove controller image {image!r}; no release tag is present."
        )
    return name.rsplit(":", 1)[1]


def schema_digest(crd):
    # Descriptions do not affect the API contract and may change independently.
    def normalize(value, property_map=False):
        if isinstance(value, dict):
            return {
                key: item
                if key in ("default", "example", "enum")
                else normalize(
                    item,
                    not property_map
                    and key in ("properties", "patternProperties", "definitions"),
                )
                for key, item in value.items()
                if property_map or key != "description"
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    versions = [
        {"name": version["name"], "schema": normalize(version.get("schema", {}))}
        for version in crd["spec"]["versions"]
        if version.get("served")
    ]
    data = json.dumps(
        sorted(versions, key=lambda item: item["name"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(data.encode()).hexdigest()


def check(
    get, contract, namespace="", deployment_name="", container_name="grove-operator"
):
    minimum = contract["minimumVersion"]
    if deployment_name:
        path = f"/apis/apps/v1/namespaces/{quote(namespace, safe='')}/deployments/{quote(deployment_name, safe='')}"
        candidates = [get(path)]
    else:
        scope = f"/namespaces/{quote(namespace, safe='')}" if namespace else ""
        candidates = get(f"/apis/apps/v1{scope}/deployments")["items"]
        candidates = [
            item
            for item in candidates
            if any(
                container["name"] == container_name
                for container in item["spec"]["template"]["spec"]["containers"]
            )
        ]
    if len(candidates) != 1:
        raise Incompatible(
            f"Unverifiable external Grove: found {len(candidates)} controller Deployments. "
            "Set grovePreflight.namespace and grovePreflight.deploymentName to identify exactly one controller."
        )

    deployment = candidates[0]
    if deployment["metadata"].get("deletionTimestamp"):
        raise Incompatible(
            "External Grove controller is being deleted; finish installing Grove, then retry."
        )
    containers = [
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == container_name
    ]
    if len(containers) != 1:
        raise Incompatible(
            f"Unverifiable Grove controller container {container_name!r}."
        )
    image = containers[0]["image"]
    version = image_version(image)
    if version_key(version) < version_key(minimum):
        raise Incompatible(
            f"External Grove {version} is incompatible with this Dynamo release. "
            f"Upgrade Grove and matching CRDs to {minimum} or later before installing or upgrading Dynamo."
        )

    # A desired image alone does not establish that the running controller was upgraded.
    status = deployment.get("status", {})
    desired = deployment["spec"].get("replicas", 1)
    if (
        desired < 1
        or status.get("observedGeneration", 0) < deployment["metadata"]["generation"]
        or any(
            status.get(key, 0) != desired
            for key in ("replicas", "updatedReplicas", "availableReplicas")
        )
    ):
        raise Incompatible(
            "External Grove controller rollout is incomplete. Wait for Grove to finish upgrading, then retry."
        )

    for name, accepted in contract["crdSchemaDigests"].items():
        crd = get(f"/apis/apiextensions.k8s.io/v1/customresourcedefinitions/{name}")
        conditions = {
            item["type"]: item["status"]
            for item in crd.get("status", {}).get("conditions", [])
        }
        if conditions.get("Established") != "True" or crd["metadata"].get(
            "deletionTimestamp"
        ):
            raise Incompatible(
                f"External Grove CRD {name} is not established or is being deleted."
            )
        if schema_digest(crd) not in accepted:
            raise Incompatible(
                f"Unverifiable external Grove CRD {name}: schema does not match this chart's tested Grove schemas. "
                f"Install matching CRDs for Grove {minimum}, or use a Dynamo chart that supports your Grove release."
            )
    return (
        f"External Grove {version} ({image}) and required CRD schemas are compatible."
    )


def main():
    contract = json.loads(Path("/preflight/compatibility.json").read_text())
    service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    context = ssl.create_default_context(cafile=str(service_account / "ca.crt"))
    token = (service_account / "token").read_text().strip()

    def get(path):
        request = Request(
            "https://kubernetes.default.svc" + path,
            headers={"Authorization": "Bearer " + token},
        )
        try:
            with urlopen(request, context=context, timeout=10) as response:
                return json.load(response)
        except HTTPError as error:
            raise Incompatible(
                f"Cannot verify external Grove: Kubernetes API returned HTTP {error.code} for {path}. "
                "Check that Grove and its CRDs exist and the preflight ServiceAccount has read access."
            ) from error

    try:
        print(
            check(
                get,
                contract,
                os.getenv("GROVE_NAMESPACE", ""),
                os.getenv("GROVE_DEPLOYMENT", ""),
                os.getenv("GROVE_CONTAINER", "grove-operator"),
            ),
            flush=True,
        )
    except (Incompatible, OSError, ValueError, KeyError) as error:
        print(f"Grove preflight failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
