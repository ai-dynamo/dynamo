#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

COMPONENTS = [
    "common",
    "frontend",
    "vllm",
    "sglang",
    "trtllm",
    "mocker",
    "llama_cpp",
    "planner",
]


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

        version_content = f'#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n#  SPDX-License-Identifier: Apache-2.0\n\n# This file is auto-generated at build time\n__version__ = "{full_version}"\n'

        for component in COMPONENTS:
            version_file_path = os.path.join(
                self.root, f"components/src/dynamo/{component}/_version.py"
            )
            with open(version_file_path, "w") as f:
                f.write(version_content)
