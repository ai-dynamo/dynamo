# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate one assessment and retain the evidence needed to inspect its findings.

assess loads local inputs, composes Dynamo's dependency schemas, and compares
request and response projections from framework to Dynamo. It also runs alias
value and selected-chunk probes. Schema transformations live in openapi.py and
responses.py; finding interpretation and report writing live in reporting/.

This module owns the new output directory and retains inputs, projections,
comparator deltas, and commands alongside reports. Failures after directory
creation can leave partial evidence; no runtime behavior is checked.
"""

import copy
import hashlib
from pathlib import Path

import jsonpointer
import yaml

from ..composition.openai import compose
from ..inputs import load_input, server_name, write_json
from ..reporting.openapi import load_projections, write_reports
from .openapi import (
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    pointer,
    request_differences,
    request_document,
)
from .responses import (
    chunk_document,
    prepare_response_source,
    response_differences,
    response_document,
    response_schema_roots,
)

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE / "composition/async-openai-0.42.1.yaml"


class AdapterLoader(yaml.SafeLoader):
    """Read adapter YAML without silently overwriting duplicate selections."""

    def construct_mapping(self, node, deep=False):
        """Build a YAML mapping; reject repeated keys instead of choosing a value."""
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate key in response adapter configuration")
        return super().construct_mapping(node, deep=deep)


def compare_responses(
    args, documents: dict, selections: dict
) -> tuple[list, list, list, list]:
    """Retain full projections and compare only mutually interpretable event layers.

    `args` supplies oasdiff and an existing output_dir. Documents and qualified
    selections use the keys "framework" and "dynamo"; selections come from
    prepare_response_source. Chunk-only pairs retain full projections, but omit
    both SSE body schemas from the event
    comparator and run an independent, explicitly selected chunk probe. Status
    and media declarations remain. Missing chunk evidence is a gap, never a pass.
    Return response deltas, gaps, chunk probe evidence and the response comparator
    command. Writes under output_dir; failures can leave partial evidence.
    Inputs are not mutated and only the pinned comparator performs schema diffs.
    """
    projections, gaps, probes, normalizations = {}, [], [], []
    for side, document in documents.items():
        projected, missing = response_document(document, selections[side])
        gaps.extend({"kind": "response", "side": side, **gap} for gap in missing)
        write_json(args.output_dir / f"{side}.responses.json", projected)
        projections[side] = projected
    event_media = {
        pointer(
            "paths",
            endpoint,
            "post",
            "responses",
            status,
            "content",
            "text/event-stream",
        )
        for projected in projections.values()
        for endpoint, path in projected["paths"].items()
        for status, response in path["post"]["responses"].items()
        if "text/event-stream" in response.get("content", {})
    }
    for media in sorted(event_media):
        for side, projected in projections.items():
            try:
                jsonpointer.resolve_pointer(projected, media[1:])
            except jsonpointer.JsonPointerException:
                gaps.append(
                    {
                        "kind": "response",
                        "side": side,
                        "endpoint": jsonpointer.JsonPointer(media[1:]).parts[1],
                        "location": media,
                        "layer": "event",
                        "reason": "SSE declaration absent; this does not establish absence of streaming behavior",
                    }
                )
    chunk_media = sorted(
        {
            media
            for streams in selections.values()
            for media, entry in streams.items()
            if entry["interpretation"] == "decoded-chunk"
        }
    )
    for index, media in enumerate(chunk_media):
        endpoint = jsonpointer.JsonPointer(media[1:]).parts[1]
        probe = {
            "endpoint": endpoint,
            "location": media,
            "layer": "chunk",
            "status": "not_compared",
            "sources": {},
        }
        paths, probe_gaps = [], []
        for side in ("framework", "dynamo"):
            # The retained full projection remains available for envelope/extension
            # inspection. Do not compare an event object against a decoded chunk.
            try:
                projected_media = jsonpointer.resolve_pointer(
                    projections[side], media[1:]
                )
                projected_media["schema"] = {}
            except jsonpointer.JsonPointerException:
                pass  # A missing media declaration remains visible in the full diff.
            selection = selections[side].get(media)
            try:
                if selection is None:
                    raise ValueError("No source-qualified shared chunk selector")
                probe["sources"][side] = copy.deepcopy(selection)
                projected, missing = chunk_document(documents[side], selection)
                probe_gaps.extend(
                    {"kind": "response", "side": side, **gap} for gap in missing
                )
                path = args.output_dir / "chunk-probes" / str(index) / f"{side}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                normalized, rewrites = normalize_nullable_strings(
                    projected, locations=response_schema_roots(projected)
                )
                write_json(path, normalized)
                write_json(path.with_suffix(".normalization.json"), rewrites)
                paths.append(path)
            except (ValueError, jsonpointer.JsonPointerException) as error:
                probe_gaps.append(
                    {
                        "kind": "response",
                        "side": side,
                        "endpoint": endpoint,
                        "location": media,
                        "layer": "chunk",
                        "reason": str(error),
                    }
                )
        if len(paths) == 2 and not probe_gaps:
            diff, command = compare(args.oasdiff, *paths, flatten=True)
            probe["differences"] = response_differences(diff)
            probe["status"] = (
                "differences_found"
                if probe["differences"]
                else "no_detected_chunk_differences"
            )
            probe["comparator_command"] = command
            evidence = paths[0].parent / "oasdiff.json"
            write_json(evidence, diff)
            probe["evidence"] = str(evidence.relative_to(args.output_dir))
        probe["coverage_gaps"] = probe_gaps
        gaps.extend(probe_gaps)
        probes.append(probe)
    response_paths = []
    for side in ("framework", "dynamo"):
        projected = projections[side]
        normalized, rewrites = normalize_nullable_strings(
            projected, locations=response_schema_roots(projected)
        )
        normalizations.extend({"side": side, **rewrite} for rewrite in rewrites)
        response_path = args.output_dir / f"{side}.responses.normalized.json"
        write_json(response_path, normalized)
        response_paths.append(response_path)
    diff, command = compare(args.oasdiff, *response_paths, flatten=not gaps)
    write_json(args.output_dir / "response-oasdiff.json", diff)
    write_json(args.output_dir / "response-normalization.json", normalizations)
    write_json(args.output_dir / "response-chunk-probes.json", probes)
    return response_differences(diff), gaps, probes, command


def assess(args) -> int:
    """Run a CLI-configured assessment and print its status and report path.

    `args` is the Namespace built by the package CLI; README.md documents its paths,
    identity, optional metadata/adapters, and pinned composition/comparator inputs.
    output_dir must not exist. Inputs are read-only; reports and evidence are written
    beneath that directory.
    Return 1 for any difference or coverage gap, otherwise 0.
    Invalid inputs, existing output, and tool failures propagate;
    a later failure may leave partial evidence in the newly created directory.
    """
    framework_name = server_name(args.framework_name)
    if framework_name == "dynamo":
        raise ValueError("Framework identity must differ from dynamo")
    dynamo_input = load_input(args.dynamo, "dynamo", args.dynamo_metadata)
    native_input = load_input(args.framework, framework_name, args.framework_metadata)
    manifest_bytes = args.manifest.read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    baseline_bytes = args.openai.read_bytes()
    composed = compose(dynamo_input.document, manifest, baseline_bytes=baseline_bytes)
    adapter_path = args.response_adapters
    adapter_bytes = adapter_path.read_bytes() if adapter_path else None
    adapter_config = (
        yaml.load(adapter_bytes, Loader=AdapterLoader)
        if adapter_bytes is not None
        else {"version": 1, "sources": {}}
    )
    if (
        not isinstance(adapter_config, dict)
        or set(adapter_config) != {"version", "sources"}
        or adapter_config["version"] != 1
        or type(adapter_config["version"]) is not int
        or not isinstance(adapter_config["sources"], dict)
        or set(adapter_config["sources"]) - {"dynamo", "framework"}
        or any(
            not isinstance(source, dict)
            for source in adapter_config["sources"].values()
        )
    ):
        raise ValueError(
            "Response adapters require version: 1 and dynamo/framework sources"
        )
    documents, selections, source_evidence = {}, {}, {}
    for side, document, snapshot in (
        ("dynamo", composed, dynamo_input),
        ("framework", native_input.document, native_input),
    ):
        (
            documents[side],
            selections[side],
            source_evidence[side],
        ) = prepare_response_source(
            document, adapter_config["sources"].get(side), snapshot.metadata["sha256"]
        )
        source_evidence[side]["native_item_schema_pointers"] = [
            pointer(
                "paths",
                endpoint,
                "post",
                "responses",
                status,
                "content",
                "text/event-stream",
                "itemSchema",
            )
            for endpoint in ("/v1/chat/completions", "/v1/completions")
            for status, response in documents[side]
            .get("paths", {})
            .get(endpoint, {})
            .get("post", {})
            .get("responses", {})
            .items()
            if isinstance(response, dict)
            and "itemSchema" in response.get("content", {}).get("text/event-stream", {})
        ]
    dynamo, dynamo_gaps = request_document(documents["dynamo"])
    native, native_gaps = request_document(documents["framework"])
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
    if adapter_bytes is not None:
        (args.output_dir / "response-adapters.yaml").write_bytes(adapter_bytes)
    write_json(args.output_dir / "response-sources.json", source_evidence)
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
    (
        response_findings,
        response_gaps,
        chunk_probes,
        response_command,
    ) = compare_responses(args, documents, selections)
    gaps.extend(response_gaps)
    for finding in [*findings, *response_findings]:
        finding["source_locations"] = {}
        for side in ("dynamo", "framework"):
            location = finding["location"]
            for route in source_evidence[side].get("path_mappings", []):
                projected = route["projection_operation"]
                if location == projected or location.startswith(projected + "/"):
                    location = route["source_operation"] + location[len(projected) :]
                    break
            finding["source_locations"][side] = location
    for gap in gaps:
        for route in source_evidence.get(gap["side"], {}).get("path_mappings", []):
            original, projected = (
                route["source_operation"],
                route["projection_operation"],
            )
            location = gap.get("location", "")
            if location == projected or location.startswith(projected + "/"):
                gap["projection_location"] = location
                gap["location"] = original + location[len(projected) :]
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
        "response_sources": source_evidence,
        "response_chunk_probes": chunk_probes,
        "response_representation": "Source-qualified decoded SSE events or selected chunks; comparison-only, not wire schemas",
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
    # Explain the retained comparator inputs without rerunning or changing the diff.
    write_reports(report, args.output_dir, load_projections(args.output_dir))
    print(
        f"{report['status']}: {len(findings)} request deltas; "
        f"{len(response_findings)} response deltas; "
        f"{sum(bool(probe.get('differences')) for probe in chunk_probes)} chunk probes with differences; "
        f"{len(gaps)} coverage gaps. "
        f"Read {args.output_dir / 'report.md'}"
    )
    return int(bool(findings or response_findings or gaps))
