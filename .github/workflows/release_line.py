# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Resolve the release in development on main and where its cycle starts.

The newest ``release/X.Y.0`` branch is the line that left main last, so main is
developing the next one. Minors run 0 to 9, then the major increments, so
``release/1.9.0`` leaves main on 2.0.0. The scan base is the commit where that
branch left main, which is the first commit of the current cycle. Release
branches only take cherry-picks and never merge back into main, so the
merge-base stays put for the whole cycle.

Model-launch branches such as ``release/1.6.0-<model>-dev.1`` and patch lines
such as ``release/1.6.1`` do not match, so neither moves main's release.
"""

import os
import re
import subprocess
import sys

RELEASE_BRANCH_RE = re.compile(r"^release/(\d+)\.(\d+)\.0$")


def newest_release_line(branches: list[str]) -> tuple[int, int] | None:
    """Return the (major, minor) of the newest minor release branch, if any."""
    lines = [
        (int(m.group(1)), int(m.group(2)))
        for m in (RELEASE_BRANCH_RE.match(b) for b in branches)
        if m
    ]
    return max(lines) if lines else None


def next_version(major: int, minor: int) -> str:
    """Return the bare version main develops after the (major, minor) line."""
    return f"{major + 1}.0.0" if minor >= 9 else f"{major}.{minor + 1}.0"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    branches = git(
        "for-each-ref", "--format=%(refname:strip=3)", "refs/remotes/origin/release/"
    ).splitlines()
    line = newest_release_line(branches)
    if line is None:
        print("::error::No release/X.Y.0 branch found; cannot resolve the release.")
        return 1
    branch = f"release/{line[0]}.{line[1]}.0"
    version = next_version(*line)
    base = git("merge-base", "HEAD", f"origin/{branch}")
    print(f"Main develops {version}; cycle starts at {base} where {branch} left main.")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as f:
            f.write(f"version={version}\nbase={base}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
