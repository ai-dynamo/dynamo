#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_components():
    """
    Scan the components/src/dynamo directory to get the list of available components.
    Returns full paths to component directories and RuntimeError if no components are found.
    """
    components_dir = os.path.join(
        os.path.dirname(__file__), "components", "src", "dynamo"
    )

    if not os.path.exists(components_dir):
        raise RuntimeError(f"Components directory not found: {components_dir}")

    components = []
    for item in os.listdir(components_dir):
        item_path = os.path.join(components_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            components.append(item_path)

    if not components:
        raise RuntimeError(f"No components found in directory: {components_dir}")

    return components


# CI passes the full source commit explicitly because the Docker build context
# excludes .git (see .dockerignore), so `git rev-parse` cannot work there.
GIT_COMMIT_ENV = "DYNAMO_GIT_COMMIT"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def resolve_git_commit(root):
    """
    Return the full source commit SHA: DYNAMO_GIT_COMMIT if set, else
    `git rev-parse HEAD` in root, else "".
    """
    explicit = os.environ.get(GIT_COMMIT_ENV, "").strip().lower()
    if explicit:
        if not _FULL_SHA.fullmatch(explicit):
            raise ValueError(
                f"{GIT_COMMIT_ENV}={explicit!r} is not a full 40-character git commit SHA"
            )
        return explicit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


class VersionWriterHook(BuildHookInterface):
    """
    A Hatch build hook to write the project version to a file.
    """

    def initialize(self, version, build_data):
        """
        This method is called before the build process begins.
        """

        full_version = self.metadata.version
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True,
            )
            git_version = result.stdout.strip()
            if git_version:
                full_version += f"+{git_version}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        git_commit = resolve_git_commit(self.root)
        if not git_commit:
            message = (
                f"Could not determine the source commit: set {GIT_COMMIT_ENV} or "
                "build from a git checkout."
            )
            if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
                raise RuntimeError(message)
            self.app.display_warning(f'{message} Writing __git_commit__ = "".')

        version_content = f'#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n#  SPDX-License-Identifier: Apache-2.0\n\n# This file is auto-generated at build time\n__version__ = "{full_version}"\n__git_commit__ = "{git_commit}"\n'

        for component in get_components():
            version_file_path = os.path.join(component, "_version.py")
            with open(version_file_path, "w") as f:
                f.write(version_content)
