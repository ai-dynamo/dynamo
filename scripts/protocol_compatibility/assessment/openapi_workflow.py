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
    request_differences,
    request_document,
)

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE / "composition/async-openai-0.42.1.yaml"


def assess(args) -> int:
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
        {"side": side, **gap}
        for side, items in (("dynamo", dynamo_gaps), ("framework", native_gaps))
        for gap in items
    ]
    coverage_bytes = (PACKAGE / "assessment/coverage.yaml").read_bytes()
    gaps.extend(
        {"side": "dynamo", **gap} for gap in yaml.safe_load(coverage_bytes)["gaps"]
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
    report = {
        "schema": "dynamo-framework-openapi-assessment/v1",
        "framework_name": framework_name,
        "status": (
            "incomplete_coverage"
            if gaps
            else ("differences_found" if findings else "no_declared_differences")
        ),
        "behavioral_conformance": "not_assessed",
        "differences": findings,
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
            "binary_sha256": hashlib.sha256(args.oasdiff.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output_dir / "oasdiff.json", raw_diff)
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "report.md").write_text(render(report))
    print(
        f"{report['status']}: {len(findings)} endpoint request deltas; {len(gaps)} coverage gaps. "
        f"Read {args.output_dir / 'report.md'}"
    )
    return int(bool(findings or gaps))
