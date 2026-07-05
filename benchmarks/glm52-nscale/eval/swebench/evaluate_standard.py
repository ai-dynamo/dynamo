#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned standard evaluator with generation-image identity guards."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import docker

from capture_task_images import install_docker_image_guard


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-images", required=True, type=Path)
    known, evaluator_args = parser.parse_known_args()
    install_docker_image_guard(docker, known.task_images)
    sys.argv = ["swebench.harness.run_evaluation", *evaluator_args]
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
