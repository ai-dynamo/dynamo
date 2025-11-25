#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import yaml
from jinja2 import Environment, FileSystemLoader

def parse_args():
    parser = argparse.ArgumentParser(description="A script that greets a user.")
    parser.add_argument("--framework", type=str, help="Dockerfile framework to use.")
    parser.add_argument("--target", type=str, help="Dockerfile target to use.")
    parser.add_argument("--platform", type=str, help="Dockerfile platform to use.")
    args = parser.parse_args()
    return args

def render(framework, target, platform, context):
    env = Environment(loader=FileSystemLoader("."),trim_blocks=True,lstrip_blocks=True)
    template = env.get_template("Dockerfile.template")
    rendered = template.render(
        context=context,
        framework=framework,
        target=target,
        platform=platform
    )
    with open("rendered.Dockerfile", "w") as f:
        f.write(rendered)

    return

def main():
    args = parse_args()
    framework = args.framework
    target = args.target
    platform = args.platform

    with open("context.yaml", "r") as f:
        context = yaml.safe_load(f)
    render(framework, target, platform, context)

if __name__ == "__main__":
    main()
