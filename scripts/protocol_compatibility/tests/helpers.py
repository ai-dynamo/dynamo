# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Git operations shared by source-only protocol integration tests."""

import os
import subprocess


def fixture_git(repo, *args):
    # Hooks can inherit GIT_DIR/INDEX_FILE from the caller. Never let a fixture
    # command modify the caller's worktree or repository configuration.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        env=environment,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    ).stdout.strip()
