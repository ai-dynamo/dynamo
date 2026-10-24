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
    `git rev-parse HEAD` when root is the top of a git checkout, else "".
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
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    toplevel, head = result.stdout.splitlines()
    # Git searches parent directories, so an unpacked tree inside another
    # checkout would otherwise report that checkout's commit.
    if os.path.realpath(toplevel) != os.path.realpath(root):
        return ""
    return head


_EXISTING_COMMIT = re.compile(r'^__git_commit__ = "([0-9a-f]{40})"$', re.MULTILINE)


def read_existing_git_commit(components):
    """
    Return the full commit SHA an earlier build wrote into the components'
    _version.py files, or "" if none holds one. Raise if they disagree.
    """
    found = {}
    for component in components:
        path = os.path.join(component, "_version.py")
        try:
            with open(path) as f:
                match = _EXISTING_COMMIT.search(f.read())
        except FileNotFoundError:
            continue
        if match:
            found[path] = match.group(1)
    commits = set(found.values())
    if len(commits) > 1:
        details = ", ".join(f"{path}={sha}" for path, sha in sorted(found.items()))
        raise ValueError(
            f"Existing _version.py files record different commits ({details}); "
            f"set {GIT_COMMIT_ENV} or build from a git checkout."
        )
    return commits.pop() if commits else ""


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

        components = get_components()
        # A wheel built from an unpacked sdist has no .git; keep the commit the
        # sdist build already wrote into _version.py.
        git_commit = resolve_git_commit(self.root) or read_existing_git_commit(
            components
        )
        if not git_commit:
            # Warn, not fail: `uv build` builds the wheel from the unpacked sdist.
            # The wheel-smoke test fails CI when a published wheel has no commit.
            self.app.display_warning(
                f"Could not determine the source commit: set {GIT_COMMIT_ENV} or "
                'build from a git checkout. Writing __git_commit__ = "".'
            )

        version_content = f'#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n#  SPDX-License-Identifier: Apache-2.0\n\n# This file is auto-generated at build time\n__version__ = "{full_version}"\n__git_commit__ = "{git_commit}"\n'

        for component in components:
            version_file_path = os.path.join(component, "_version.py")
            with open(version_file_path, "w") as f:
                f.write(version_content)
