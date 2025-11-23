#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("."), trim_blocks=True, lstrip_blocks=True)
template = env.get_template("Dockerfile.template")

rendered = template.render(
    framework="vllm",
    target="runtime"
)

with open("rendered.Dockerfile", "w") as f:
    f.write(rendered)
