# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject unsupported Dynamo Helm upgrade paths before the release is changed."""

import json
import os
import re
import ssl
import sys
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

SERVICE_ACCOUNT_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount")
VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?$")


class PreflightError(Exception):
    """An upgrade condition that must stop the Helm release."""


def parse_version(value):
    """Parse a Dynamo semantic version into its numeric components."""
    match = VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise PreflightError(f"version {value!r} is not a supported semantic version")
    return tuple(int(component) for component in match.groups())


def validate_upgrade(installed_version, target_version):
    """Allow patch upgrades and exactly one forward minor step."""
    installed = parse_version(installed_version)
    target = parse_version(target_version)

    same_minor = target[:2] == installed[:2]
    next_minor = target[0] == installed[0] and target[1] == installed[1] + 1
    if same_minor or next_minor:
        return

    raise PreflightError(
        f"unsupported Dynamo upgrade from {installed_version} to {target_version}. "
        "Dynamo supports Helm upgrades only within the installed minor release or "
        "to the next minor release. Upgrade through every intermediate minor release."
    )


def installed_version(payload):
    """Extract the installed Dynamo version from one operator Deployment."""
    items = payload.get("items", [])
    if len(items) != 1:
        raise PreflightError(
            f"expected exactly one operator Deployment for this Helm release, found {len(items)}"
        )

    labels = items[0].get("metadata", {}).get("labels", {})
    version = labels.get("app.kubernetes.io/version")
    if not version:
        raise PreflightError(
            "the installed operator Deployment has no app.kubernetes.io/version label"
        )
    return version


def kubernetes_get_json(path):
    """Read JSON from the in-cluster Kubernetes API."""
    token = (SERVICE_ACCOUNT_PATH / "token").read_text(encoding="utf-8").strip()
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
    request = Request(f"https://{host}:{port}{path}")
    request.add_header("Authorization", f"Bearer {token}")
    context = ssl.create_default_context(cafile=str(SERVICE_ACCOUNT_PATH / "ca.crt"))
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
    with opener.open(request, timeout=15) as response:
        return json.load(response)


def main():
    """Check the installed release against the chart being applied."""
    namespace = os.environ["RELEASE_NAMESPACE"]
    release_name = os.environ["RELEASE_NAME"]
    target_version = os.environ["TARGET_VERSION"]
    selector = (
        f"app.kubernetes.io/instance={release_name},"
        "app.kubernetes.io/component=manager"
    )
    query = urlencode({"labelSelector": selector})
    path = f"/apis/apps/v1/namespaces/{quote(namespace, safe='')}/deployments?{query}"

    try:
        current_version = installed_version(kubernetes_get_json(path))
        print(f"Checking Dynamo Helm upgrade {current_version} -> {target_version}")
        validate_upgrade(current_version, target_version)
    except Exception as error:
        print(f"ERROR: Dynamo upgrade preflight failed: {error}", file=sys.stderr)
        print(
            "To bypass this check after validating the complete upgrade path, set "
            "dynamo-operator.upgradePreflight.enabled=false.",
            file=sys.stderr,
        )
        return 1

    print("Dynamo upgrade path is supported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
