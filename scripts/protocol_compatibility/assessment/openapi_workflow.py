# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect retained OpenAPI documents, composition, and comparison."""

import hashlib
from pathlib import Path

import yaml

from ..acquisition.http import load_capture, server_name, write_json
from ..composition.openai import compose
from ..reporting.openapi import render
from .openapi import (
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    pointer,
    request_differences,
    request_document,
)
from .responses import response_differences, response_document

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE / "composition/async-openai-0.42.1.yaml"


def assess(args) -> int:
    """Retain inputs and report request/response candidates, never runtime parity.

    `args` supplies the acquisition paths, composition manifest/baseline, pinned
    comparator and a new output directory. Return 1 for any difference or coverage
    gap, otherwise 0. Invalid inputs, existing output, and tool failures propagate;
    a later failure may leave partial evidence in the newly created directory.
    """
    framework_name = server_name(args.framework_name)
    if framework_name == "dynamo":
        raise ValueError("Framework identity must differ from dynamo")
    dynamo_input = load_capture(args.dynamo, "dynamo")
    native_input = load_capture(args.framework, framework_name)
    manifest_bytes = args.manifest.read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    baseline_bytes = args.openai.read_bytes()
    composed = compose(dynamo_input.document, manifest, baseline_bytes=baseline_bytes)
    dynamo, dynamo_gaps = request_document(composed)
    native, native_gaps = request_document(native_input.document)
    gaps = [
        {"kind": "request", "side": side, **gap}
        for side, items in (("dynamo", dynamo_gaps), ("framework", native_gaps))
        for gap in items
    ]
    coverage_bytes = (PACKAGE / "assessment/coverage.yaml").read_bytes()
    gaps.extend(
        {"kind": "request", "side": "dynamo", **gap}
        for gap in yaml.safe_load(coverage_bytes)["gaps"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dynamo_input.save(args.output_dir / "dynamo")
    native_input.save(args.output_dir / "framework")
    (args.output_dir / "openai.yaml").write_bytes(baseline_bytes)
    (args.output_dir / "composition.yaml").write_bytes(manifest_bytes)
    (args.output_dir / "coverage.yaml").write_bytes(coverage_bytes)
    write_json(args.output_dir / "dynamo.composed.json", composed)
    dynamo_path, native_path = (
        args.output_dir / "dynamo.requests.json",
        args.output_dir / "framework.requests.json",
    )
    write_json(dynamo_path, dynamo)
    write_json(native_path, native)
    normalized_dynamo, dynamo_normalizations = normalize_nullable_strings(dynamo)
    normalized_native, native_normalizations = normalize_nullable_strings(native)
    normalizations = [
        {"side": side, **change}
        for side, changes in (
            ("dynamo", dynamo_normalizations),
            ("framework", native_normalizations),
        )
        for change in changes
    ]
    dynamo_path = args.output_dir / "dynamo.requests.normalized.json"
    native_path = args.output_dir / "framework.requests.normalized.json"
    write_json(dynamo_path, normalized_dynamo)
    write_json(native_path, normalized_native)
    write_json(args.output_dir / "normalization.json", normalizations)
    raw_diff, command = compare(
        args.oasdiff, native_path, dynamo_path, flatten=not (dynamo_gaps or native_gaps)
    )
    findings = request_differences(raw_diff)
    alias_matches, alias_gaps = match_input_aliases(
        normalized_dynamo, normalized_native
    )
    gaps.extend(alias_gaps)
    for index, match in enumerate(alias_matches):
        directory = args.output_dir / "alias-probes" / str(index)
        directory.mkdir(parents=True)
        for side, document, location in (
            ("dynamo", normalized_dynamo, match["dynamo_location"]),
            ("framework", normalized_native, match["backend_location"]),
        ):
            write_json(
                directory / f"{side}.json", alias_value_document(document, location)
            )
        value_diff, value_command = compare(
            args.oasdiff,
            directory / "framework.json",
            directory / "dynamo.json",
            flatten=not (dynamo_gaps or native_gaps),
        )
        value_findings = request_differences(value_diff)
        match["value_comparison"] = (
            "differences_found" if value_findings else "no_detected_differences"
        )
        match["value_differences"] = value_findings
        match["comparator_command"] = value_command
        match["evidence"] = str(directory.relative_to(args.output_dir) / "oasdiff.json")
        write_json(directory / "oasdiff.json", value_diff)
    write_json(args.output_dir / "alias-matches.json", alias_matches)
    response_paths, response_gaps, response_normalizations = [], [], []
    for side, document in (("framework", native_input.document), ("dynamo", composed)):
        projected, missing = response_document(document)
        response_gaps.extend(
            {"kind": "response", "side": side, **gap} for gap in missing
        )
        write_json(args.output_dir / f"{side}.responses.json", projected)
        roots = [
            pointer(
                "paths",
                endpoint,
                "post",
                "responses",
                status,
                "content",
                media,
                "schema",
            )
            for endpoint, path in projected["paths"].items()
            for status, response in path["post"]["responses"].items()
            for media in response.get("content", {})
        ]
        normalized, rewrites = normalize_nullable_strings(projected, locations=roots)
        response_normalizations.extend(
            {"side": side, **rewrite} for rewrite in rewrites
        )
        response_path = args.output_dir / f"{side}.responses.normalized.json"
        write_json(response_path, normalized)
        response_paths.append(response_path)
    response_diff, response_command = compare(
        args.oasdiff, *response_paths, flatten=not response_gaps
    )
    response_findings = response_differences(response_diff)
    gaps.extend(response_gaps)
    write_json(args.output_dir / "response-oasdiff.json", response_diff)
    write_json(args.output_dir / "response-normalization.json", response_normalizations)
    report = {
        "schema": "dynamo-framework-openapi-assessment/v2",
        "framework_name": framework_name,
        "status": (
            "incomplete_coverage"
            if gaps
            else (
                "differences_found"
                if findings or response_findings
                else "no_declared_differences"
            )
        ),
        "behavioral_conformance": "not_assessed",
        "differences": findings,
        "response_differences": response_findings,
        "response_representation": "SSE envelopes with encoding-tagged decoded data; comparison-only, not a wire schema",
        "representation_normalizations": normalizations,
        "alias_matches": alias_matches,
        "coverage_gaps": gaps,
        "exclusions": manifest["exclusions"],
        "inputs": {"dynamo": dynamo_input.metadata, "framework": native_input.metadata},
        "deployment_provenance": "not_verified",
        "composition_dependencies": {
            "expected": manifest["dependencies"],
            "verification": "caller_responsibility",
        },
        "composition_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "coverage_catalog_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "comparator": {
            "command": command,
            "response_command": response_command,
            "binary_sha256": hashlib.sha256(args.oasdiff.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output_dir / "oasdiff.json", raw_diff)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(render(report))
    print(
        f"{report['status']}: {len(findings)} request deltas; "
        f"{len(response_findings)} response deltas; {len(gaps)} coverage gaps. "
        f"Read {args.output_dir / 'report.md'}"
    )
    return int(bool(findings or response_findings or gaps))
