#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Render the portable CPU-Mocker campaign manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


class RenderError(ValueError):
    """The Mocker template cannot be rendered without changing its contract."""


def render_manifest(
    template: str,
    replacements: dict[str, str],
    *,
    model_cache_claim: str | None = None,
    image_pull_secret: str | None = None,
) -> str:
    """Render required scalars and apply optional cluster resource settings."""
    rendered = template
    for name, value in replacements.items():
        rendered = rendered.replace(f"${{{name}}}", value)
    if "${" in rendered:
        raise RenderError("template contains an unresolved variable")

    documents = [document for document in yaml.safe_load_all(rendered) if document]
    deployments = [
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(deployments) != 1:
        raise RenderError("template must contain one DynamoGraphDeployment")

    components = deployments[0].get("spec", {}).get("components", [])
    if not isinstance(components, list) or not components:
        raise RenderError("DynamoGraphDeployment has no components")
    for component in components:
        pod_spec = component.get("podTemplate", {}).get("spec")
        if not isinstance(pod_spec, dict):
            raise RenderError("component is missing podTemplate.spec")
        volumes = pod_spec.get("volumes")
        if not isinstance(volumes, list):
            raise RenderError("component is missing volumes")
        model_cache = next(
            (
                volume
                for volume in volumes
                if isinstance(volume, dict) and volume.get("name") == "model-cache"
            ),
            None,
        )
        if model_cache is None:
            raise RenderError("component is missing the model-cache volume")
        if model_cache_claim:
            model_cache.pop("emptyDir", None)
            model_cache["persistentVolumeClaim"] = {"claimName": model_cache_claim}
        else:
            model_cache.pop("persistentVolumeClaim", None)
            model_cache["emptyDir"] = {}
        pod_spec["imagePullSecrets"] = (
            [{"name": image_pull_secret}] if image_pull_secret else []
        )

    return yaml.safe_dump_all(
        documents,
        explicit_start=True,
        sort_keys=False,
        width=100,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dynamo-image", required=True)
    parser.add_argument("--dynamo-runtime-version", required=True)
    parser.add_argument("--mocker-max-num-seqs", required=True, type=int)
    parser.add_argument("--mocker-timing-flag", required=True)
    parser.add_argument("--mocker-timing-value", required=True)
    parser.add_argument("--model-cache-claim")
    parser.add_argument("--image-pull-secret")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render one manifest from command-line arguments."""
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.mocker_max_num_seqs <= 0:
        raise SystemExit("--mocker-max-num-seqs must be positive")

    replacements = {
        "NAMESPACE": args.namespace,
        "DYNAMO_IMAGE": args.dynamo_image,
        "DYNAMO_RUNTIME_VERSION": args.dynamo_runtime_version,
        "MOCKER_MAX_NUM_SEQS": str(args.mocker_max_num_seqs),
        "MOCKER_TIMING_FLAG": args.mocker_timing_flag,
        "MOCKER_TIMING_VALUE": args.mocker_timing_value,
    }
    try:
        output = render_manifest(
            args.template.read_text(encoding="utf-8"),
            replacements,
            model_cache_claim=args.model_cache_claim,
            image_pull_secret=args.image_pull_secret,
        )
        args.output.write_text(output, encoding="utf-8")
    except (OSError, RenderError, yaml.YAMLError) as error:
        raise SystemExit(f"failed to render Mocker manifest: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
