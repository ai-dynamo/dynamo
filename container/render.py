#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Renders dynamo Dockerfiles from templates")
    parser.add_argument("--framework", type=str, default="vllm", help="Dockerfile framework to use.")
    parser.add_argument("--target", type=str, default="runtime", help="Dockerfile target to use.")
    parser.add_argument("--platform", type=str, default="amd64", help="Dockerfile platform to use.")
    parser.add_argument("--cuda-version", type=str, default="12.9", help="CUDA version to use.")
    parser.add_argument("--show-result", action='store_true', help="Prints the rendered Dockerfile to stdout.")
    args = parser.parse_args()
    return args

def validate_args(args):
    # TODO: Add validation logic
    return

def render(args, context, script_dir):
    env = Environment(loader=FileSystemLoader(script_dir),trim_blocks=False,lstrip_blocks=True)
    template = env.get_template("Dockerfile.template")
    rendered = template.render(
        context=context,
        framework=args.framework,
        target=args.target,
        platform=args.platform,
        cuda_version=args.cuda_version
    )
    with open(f"{script_dir}/rendered.Dockerfile", "w") as f:
        f.write(rendered)

    if args.show_result == True:
        print("##############")
        print("# Dockerfile #")
        print("##############")
        print(rendered)
        print("##############")

    return

def main():
    args = parse_args()
    validate_args(args)
    script_dir = Path(sys.argv[0]).parent
    with open(f"{script_dir}/context.yaml", "r") as f:
        context = yaml.safe_load(f)

    render(args, context, script_dir)

if __name__ == "__main__":
    main()
