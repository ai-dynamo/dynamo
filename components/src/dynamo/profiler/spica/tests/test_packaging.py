# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-package contracts for the experimental Spica feature."""

import importlib.metadata
import importlib.util
import subprocess
import sys


def test_spica_has_no_console_script():
    scripts = importlib.metadata.entry_points(group="console_scripts")

    assert all(entry.name != "spica" for entry in scripts)


def test_spica_has_no_top_level_compatibility_alias():
    assert importlib.util.find_spec("spica") is None


def test_profiler_does_not_eagerly_reexport_spica():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import dynamo.profiler; assert not hasattr(dynamo.profiler, 'spica')",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
