# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only Dynamo version check for the Helm pre-install/pre-upgrade Job."""

import re
import sys
from urllib.parse import quote

VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?$")
MINOR_VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)$")


def version(value):
    """Parse a Dynamo semantic version into numeric components."""
    match = VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Version {value!r} is not a supported semantic version")
    return tuple(int(component) for component in match.groups())


def minor_version(value, field_name):
    """Parse an override that identifies one installed minor release."""
    match = MINOR_VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            f"{field_name} must be a minor version in N.M form, got {value!r}"
        )
    return tuple(int(component) for component in match.groups())


def validate_transition(
    installed_value,
    target_value,
    unsupported_upgrade_override="",
    unsupported_downgrade_override="",
):
    """Allow transitions within the same minor or to an adjacent minor."""
    installed = version(installed_value)
    target = version(target_value)
    if target[:2] == installed[:2] or (
        target[0] == installed[0] and abs(target[1] - installed[1]) == 1
    ):
        return

    is_upgrade = target[:2] > installed[:2]
    direction = "upgrade" if is_upgrade else "downgrade"
    field_name = (
        "unsupportedUpgradeOverride" if is_upgrade else "unsupportedDowngradeOverride"
    )
    override = (
        unsupported_upgrade_override if is_upgrade else unsupported_downgrade_override
    )
    if override and minor_version(override, field_name) == installed[:2]:
        print(
            f"WARNING: allowing unsupported Dynamo {direction} from {installed_value} to {target_value} "
            f"because {field_name}={override} matches the installed minor release."
        )
        return

    installed_minor = f"{installed[0]}.{installed[1]}"
    raise ValueError(
        f"Unsupported Dynamo {direction} from {installed_value} to {target_value}. "
        "Dynamo supports Helm transitions only within the installed minor release or to an adjacent minor release. "
        "Move through every intermediate minor release. "
        f"To allow this specific unsupported {direction}, set "
        f"dynamo-operator.{field_name}={installed_minor}. The override applies only while "
        f"the installed operator minor is {installed_minor}"
    )


def check(
    get, namespace, deployment, target, upgrade_override="", downgrade_override=""
):
    """Read the installed operator Deployment and validate the requested transition."""
    path = (
        f"/apis/apps/v1/namespaces/{quote(namespace, safe='')}/deployments/"
        f"{quote(deployment, safe='')}"
    )
    payload = get(path)
    if payload is None:
        print(
            f"No existing operator Deployment {namespace}/{deployment}; allowing initial installation."
        )
        return

    installed = (
        payload.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/version")
    )
    if not installed:
        raise ValueError(
            "Installed operator Deployment has no app.kubernetes.io/version label"
        )

    print(f"Checking Dynamo Helm transition {installed} -> {target}")
    validate_transition(installed, target, upgrade_override, downgrade_override)
    print("Dynamo version transition is supported.")


def main():
    get = globals()["kubernetes_client"]()
    check(lambda path: get(path, missing_ok=True), *sys.argv[1:])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"Dynamo version preflight failed: {error}.")
