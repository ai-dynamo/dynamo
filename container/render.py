#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 

import argparse
import re
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

def parse_args():
+    parser = argparse.ArgumentParser(
+        description="Renders dynamo Dockerfiles from templates"
+    )
+    parser.add_argument(
+        "--framework", type=str, default="vllm", help="Dockerfile framework to use."
+    )
+    parser.add_argument(
+        "--target", type=str, default="runtime", help="Dockerfile target to use."
+    )
+    parser.add_argument(
+        "--platform", type=str, default="amd64", help="Dockerfile platform to use."
+    )
+    parser.add_argument(
+        "--cuda-version", type=str, default="12.9", help="CUDA version to use."
+    )
+    parser.add_argument("--make-efa", action="store_true", help="Enable AWS EFA")
+    parser.add_argument(
+        "--short-output",
+        action="store_true",
+        help="Output filename is just rendered.Dockerfile",
+    )
+    parser.add_argument(
+        "--show-result",
+        action="store_true",
+        help="Prints the rendered Dockerfile to stdout.",
+    )
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
        cuda_version=args.cuda_version,
        make_efa=args.make_efa,
    )
    # Replace all instances of 3+ newlines with 2 newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", rendered)

    if args.short_output:
        filename = "rendered.Dockerfile"
    else:
        filename = f"{args.framework}-{args.target}-cuda{args.cuda_version}-{args.platform}-rendered.Dockerfile"

    with open(f"{script_dir}/{filename}", "w") as f:
        f.write(cleaned)

    if args.show_result:
        print("##############")
        print("# Dockerfile #")
        print("##############")
        print(cleaned)
        print("##############")

    return

def main():
    args = parse_args()
    validate_args(args)
    script_dir = Path(sys.argv[0]).parent
    with open(f"{script_dir}/context.yaml", "r") as f:
        context = yaml.safe_load(f)

    render(args, context, script_dir)

    if args.target == "local-dev":
+        print(
+            "INFO: Remember to add --build-arg values for USER_UID and USER_GID when building a local-dev image!"
+        )
+        print(
+            "      Recommendation: --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)"
+        )

if __name__ == "__main__":
    main()
