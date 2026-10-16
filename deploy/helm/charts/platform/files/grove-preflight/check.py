# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only external Grove check for the Helm pre-install/pre-upgrade Job."""

import json
import re
import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen

CRDS = (
    "podcliquesets",
    "podcliques",
    "podcliquescalinggroups",
    "clustertopologybindings",
    "podgangmaps",
)


def version(tag):
    # Recognize Grove alpha releases and stable releases; custom/RC tags are unverifiable.
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:-alpha\.(\d+))?", tag)
    if not match:
        raise ValueError(
            f"Unverifiable Grove version {tag!r}; use a release-tagged image."
        )
    return tuple(int(n) for n in match.group(1, 2, 3)) + (
        int(match[4]) if match[4] else float("inf"),
    )


def check(get, minimum, namespace="", deployment=""):
    scope = f"/namespaces/{namespace}" if namespace else ""
    path = f"/apis/apps/v1{scope}/deployments"
    candidates = [get(f"{path}/{deployment}")] if deployment else get(path)["items"]
    controllers = [
        (item, container)
        for item in candidates
        for container in item["spec"]["template"]["spec"]["containers"]
        if container["name"] == "grove-operator"
    ]
    if len(controllers) != 1:
        raise ValueError(
            f"Found {len(controllers)} Grove controllers; set grovePreflight.namespace/deploymentName."
        )
    controller, container = controllers[0]
    image = container["image"].split("@", 1)[0].rsplit("/", 1)[-1]
    tag = image.split(":", 1)[1] if ":" in image else ""
    if version(tag) < version(minimum):
        raise ValueError(
            f"External Grove {tag} is incompatible. Upgrade Grove and matching CRDs to {minimum} or later."
        )

    # Do not accept a new desired image while an older controller is still running.
    status = controller.get("status", {})
    replicas = controller["spec"].get("replicas", 1)
    if (
        controller["metadata"].get("deletionTimestamp")
        or replicas < 1
        or status.get("observedGeneration", 0) < controller["metadata"]["generation"]
        or any(
            status.get(key, 0) != replicas
            for key in ("replicas", "updatedReplicas", "availableReplicas")
        )
    ):
        raise ValueError(
            "Grove rollout is incomplete; wait for it to finish, then retry."
        )
    for name in CRDS:
        crd = get(
            f"/apis/apiextensions.k8s.io/v1/customresourcedefinitions/{name}.grove.io"
        )
        established = any(
            c["type"] == "Established" and c["status"] == "True"
            for c in crd.get("status", {}).get("conditions", [])
        )
        if not established or crd["metadata"].get("deletionTimestamp"):
            raise ValueError(f"Grove CRD {name}.grove.io is not established.")
    print(f"Grove {tag}: controller rollout complete; required CRDs established.")


def main():
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    context = ssl.create_default_context(cafile=str(account / "ca.crt"))
    token = (account / "token").read_text().strip()

    def get(path):
        request = Request(
            "https://kubernetes.default.svc" + path,
            headers={"Authorization": "Bearer " + token},
        )
        with urlopen(request, context=context, timeout=10) as response:
            return json.load(response)

    check(get, *sys.argv[1:])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(
            f"Grove preflight failed: {error}. Check Grove installation and preflight read permissions."
        )
