#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate offline replay parity and performance between two Rust binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_TRACE_SHA256 = (
    "e2bec9bccd21978f69a2ace7886ee2e5192a64839f7aa2c4dd5d8766fd392510"
)
BOOTSTRAP_SEED = 0xD1A05EED
ORDER_SEED = 0xA11CE
WARMUPS = 5
INITIAL_PAIRS = 30
MAX_PAIRS = 60
REGRESSION_LIMIT = 1.05
BUILD_MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
REPLAY_BENCH_FEATURE = "replay-bench"
KVBM_FEATURE = "mocker-kvbm-offload"
REQUIRED_BUILD_MANIFEST_FIELDS = {
    "schema_version",
    "source_revision",
    "binary_sha256",
    "build_host",
    "rustc",
    "target",
    "profile",
    "features",
    "cargo_flags",
}


class ValidationFailure(RuntimeError):
    """A protocol failure that should produce a structured failing result."""


@dataclass(frozen=True)
class BuildArtifact:
    label: str
    binary: Path
    binary_sha256: str
    manifest_path: Path
    manifest: dict[str, Any]

    def result_record(self) -> dict[str, Any]:
        return {
            "binary": str(self.binary),
            "binary_sha256": self.binary_sha256,
            "manifest": str(self.manifest_path),
            "provenance": self.manifest,
        }


@dataclass(frozen=True)
class ReplayConfig:
    name: str
    serving_mode: str
    router_mode: str

    def arguments(self, *, kvbm_stress: bool = False) -> list[str]:
        arrival_speedup_ratio = "64" if kvbm_stress else "4"
        num_gpu_blocks = "3072" if kvbm_stress else "16384"
        args = [
            "--serving-mode",
            self.serving_mode,
            "--router-mode",
            self.router_mode,
            "--engine-type",
            "vllm",
            "--arrival-speedup-ratio",
            arrival_speedup_ratio,
            "--trace-block-size",
            "512",
            "--block-size",
            "512",
            "--num-gpu-blocks",
            num_gpu_blocks,
            "--max-num-seqs",
            "256",
            "--max-num-batched-tokens",
            "32768",
            "--speedup-ratio",
            "1",
            "--decode-speedup-ratio",
            "1",
            "--kv-bytes-per-token",
            "262144",
            "--kv-transfer-bandwidth",
            "100",
            "--kv-transfer-timing-mode",
            "full-prompt",
        ]
        if kvbm_stress:
            args.extend(
                [
                    "--num-g2-blocks",
                    "16384",
                    "--offload-batch-size",
                    "8",
                    "--bandwidth-g1-to-g2-gbps",
                    "32",
                    "--bandwidth-g2-to-g1-gbps",
                    "32",
                ]
            )
        if self.serving_mode == "aggregated":
            return [*args, "--num-workers", "4"]
        return [
            *args,
            "--num-prefill-workers",
            "2",
            "--num-decode-workers",
            "2",
        ]


CONFIGURATIONS = (
    ReplayConfig("aggregated_rr", "aggregated", "round-robin"),
    ReplayConfig("aggregated_kv", "aggregated", "kv-router"),
    ReplayConfig("disaggregated_rr", "disagg", "round-robin"),
    ReplayConfig("disaggregated_kv", "disagg", "kv-router"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace(trace: Path, results: dict[str, Any]) -> None:
    if not trace.is_file():
        raise ValidationFailure(f"trace does not exist: {trace}")
    trace_checksum = sha256_file(trace)
    results["trace"] = {
        "path": str(trace),
        "sha256": trace_checksum,
        "expected_sha256": EXPECTED_TRACE_SHA256,
    }
    results["trace_sha256"] = trace_checksum
    if trace_checksum != EXPECTED_TRACE_SHA256:
        raise ValidationFailure(
            f"trace checksum mismatch: expected {EXPECTED_TRACE_SHA256}, got {trace_checksum}"
        )


def require_string(manifest: dict[str, Any], field: str, manifest_path: Path) -> str:
    value = manifest[field]
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"{manifest_path}: {field} must be a non-empty string")
    return value


def require_string_list(
    manifest: dict[str, Any], field: str, manifest_path: Path
) -> list[str]:
    value = manifest[field]
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValidationFailure(
            f"{manifest_path}: {field} must be a list of unique non-empty strings"
        )
    return value


def load_build_artifact(label: str, binary: Path, manifest_path: Path) -> BuildArtifact:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValidationFailure(f"{label} binary is not executable: {binary}")
    if not manifest_path.is_file():
        raise ValidationFailure(
            f"{label} build manifest does not exist: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationFailure(
            f"failed to read {label} build manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValidationFailure(
            f"{manifest_path}: build manifest must be a JSON object"
        )

    missing = sorted(REQUIRED_BUILD_MANIFEST_FIELDS.difference(manifest))
    if missing:
        raise ValidationFailure(f"{manifest_path}: missing required fields: {missing}")
    if manifest["schema_version"] != BUILD_MANIFEST_SCHEMA_VERSION:
        raise ValidationFailure(
            f"{manifest_path}: schema_version must be {BUILD_MANIFEST_SCHEMA_VERSION}"
        )

    for field in (
        "source_revision",
        "binary_sha256",
        "build_host",
        "rustc",
        "target",
        "profile",
    ):
        require_string(manifest, field, manifest_path)
    features = sorted(require_string_list(manifest, "features", manifest_path))
    cargo_flags = require_string_list(manifest, "cargo_flags", manifest_path)
    declared_sha256 = manifest["binary_sha256"].lower()
    if len(declared_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in declared_sha256
    ):
        raise ValidationFailure(
            f"{manifest_path}: binary_sha256 must contain 64 hexadecimal characters"
        )

    actual_sha256 = sha256_file(binary)
    if declared_sha256 != actual_sha256:
        raise ValidationFailure(
            f"{label} binary checksum mismatch: manifest has {declared_sha256}, "
            f"binary has {actual_sha256}"
        )

    normalized_manifest = dict(manifest)
    normalized_manifest["binary_sha256"] = declared_sha256
    normalized_manifest["features"] = features
    normalized_manifest["cargo_flags"] = cargo_flags
    return BuildArtifact(
        label=label,
        binary=binary,
        binary_sha256=actual_sha256,
        manifest_path=manifest_path,
        manifest=normalized_manifest,
    )


def validate_feature_policy(
    artifact: BuildArtifact,
    *,
    required: set[str],
    forbidden: set[str],
) -> None:
    features = set(artifact.manifest["features"])
    missing = sorted(required.difference(features))
    present = sorted(forbidden.intersection(features))
    if missing or present:
        raise ValidationFailure(
            f"{artifact.label} feature mismatch: missing={missing}, forbidden_present={present}"
        )
    if artifact.manifest["profile"] != "release":
        raise ValidationFailure(
            f"{artifact.label} must use profile=release, got {artifact.manifest['profile']}"
        )


def validate_artifact_pair(
    pair_name: str,
    baseline: BuildArtifact,
    candidate: BuildArtifact,
    *,
    required_features: set[str],
    forbidden_features: set[str],
) -> None:
    validate_feature_policy(
        baseline, required=required_features, forbidden=forbidden_features
    )
    validate_feature_policy(
        candidate, required=required_features, forbidden=forbidden_features
    )
    if baseline.binary.resolve() == candidate.binary.resolve():
        raise ValidationFailure(
            f"{pair_name} baseline and candidate use the same binary path"
        )
    if baseline.binary_sha256 == candidate.binary_sha256:
        raise ValidationFailure(
            f"{pair_name} baseline and candidate binaries are identical"
        )
    if baseline.manifest["source_revision"] == candidate.manifest["source_revision"]:
        raise ValidationFailure(
            f"{pair_name} baseline and candidate source_revision values must differ"
        )

    for field in (
        "build_host",
        "rustc",
        "target",
        "profile",
        "features",
        "cargo_flags",
    ):
        if baseline.manifest[field] != candidate.manifest[field]:
            raise ValidationFailure(
                f"{pair_name} baseline and candidate manifest field differs: {field}"
            )


def validate_feature_variant(
    base: BuildArtifact,
    variant: BuildArtifact,
    *,
    added_feature: str,
) -> None:
    if base.binary.resolve() == variant.binary.resolve():
        raise ValidationFailure(
            f"{variant.label} must use a separate binary from {base.label}"
        )
    if base.binary_sha256 == variant.binary_sha256:
        raise ValidationFailure(f"{variant.label} binary must differ from {base.label}")
    if base.manifest["source_revision"] != variant.manifest["source_revision"]:
        raise ValidationFailure(
            f"{variant.label} must use the same source_revision as {base.label}"
        )
    for field in ("build_host", "rustc", "target", "profile"):
        if base.manifest[field] != variant.manifest[field]:
            raise ValidationFailure(
                f"{variant.label} must match {base.label} manifest field: {field}"
            )
    expected_features = sorted([*base.manifest["features"], added_feature])
    if variant.manifest["features"] != expected_features:
        raise ValidationFailure(
            f"{variant.label} features must equal {base.label} features plus {added_feature}"
        )


def run_checked(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("DYN_LOG", "warn")
    completed = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    stderr_tail = "\n".join(completed.stderr.splitlines()[-40:])
    raise RuntimeError(
        f"replay command failed with exit code {completed.returncode}:\n"
        f"{' '.join(command)}\n{stderr_tail}"
    )


def base_command(
    binary: Path,
    trace: Path,
    config: ReplayConfig,
    *,
    kvbm_stress: bool = False,
) -> list[str]:
    return [
        str(binary),
        str(trace),
        *config.arguments(kvbm_stress=kvbm_stress),
    ]


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def run_parity_matrix(
    matrix_name: str,
    baseline: BuildArtifact,
    candidate: BuildArtifact,
    trace: Path,
    root: Path,
    results: dict[str, Any],
    *,
    kvbm_stress: bool,
) -> None:
    matrix_root = root / matrix_name
    matrix_root.mkdir()
    matrix_results: dict[str, Any] = {}
    results["matrices"][matrix_name] = matrix_results

    for config in CONFIGURATIONS:
        config_result: dict[str, Any] = {
            "status": "running",
            "arguments": config.arguments(kvbm_stress=kvbm_stress),
        }
        matrix_results[config.name] = config_result
        revision_lines: dict[str, list[bytes]] = {}
        for revision, artifact in (
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            report_path = matrix_root / f"{config.name}-{revision}.jsonl"
            command = [
                *base_command(
                    artifact.binary,
                    trace,
                    config,
                    kvbm_stress=kvbm_stress,
                ),
                "--iterations",
                "20",
                "--canonical-reports-jsonl",
                str(report_path),
            ]
            config_result[f"{revision}_command"] = command[:-1] + [
                "<temporary-report-jsonl>"
            ]
            try:
                run_checked(command)
                lines = report_path.read_bytes().splitlines()
            except Exception:
                config_result["status"] = "error"
                config_result["failing_revision"] = revision
                raise
            hashes = [hashlib.sha256(line).hexdigest() for line in lines]
            config_result[f"{revision}_report_count"] = len(lines)
            config_result[f"{revision}_report_sha256"] = hashes
            if any(json.loads(line).get("replay_bench") is not True for line in lines):
                config_result["status"] = "fail"
                raise ValidationFailure(
                    f"{matrix_name} {config.name} {revision} was not emitted by a "
                    "replay-bench binary"
                )
            if len(lines) != 20:
                config_result["status"] = "fail"
                raise ValidationFailure(
                    f"{matrix_name} {config.name} {revision} emitted {len(lines)} "
                    "reports, expected 20"
                )
            if any(line != lines[0] for line in lines[1:]):
                config_result["status"] = "fail"
                raise ValidationFailure(
                    f"{matrix_name} {config.name} {revision} is not byte-deterministic"
                )
            revision_lines[revision] = lines

        baseline_bytes = revision_lines["baseline"][0]
        candidate_bytes = revision_lines["candidate"][0]
        baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
        candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
        config_result["baseline_canonical_report_sha256"] = baseline_hash
        config_result["candidate_canonical_report_sha256"] = candidate_hash
        config_result["canonical_report_bytes"] = len(baseline_bytes)
        if baseline_bytes != candidate_bytes:
            config_result["status"] = "fail"
            raise ValidationFailure(
                f"{matrix_name} {config.name} baseline and candidate canonical reports "
                f"differ: {baseline_hash} != {candidate_hash}"
            )
        config_result["status"] = "pass"


def parity(args: argparse.Namespace, results: dict[str, Any]) -> int:
    baseline = load_build_artifact(
        "baseline",
        args.baseline.resolve(),
        args.baseline_manifest.resolve(),
    )
    candidate = load_build_artifact(
        "candidate",
        args.candidate.resolve(),
        args.candidate_manifest.resolve(),
    )
    kvbm_baseline = load_build_artifact(
        "kvbm baseline",
        args.kvbm_baseline.resolve(),
        args.kvbm_baseline_manifest.resolve(),
    )
    kvbm_candidate = load_build_artifact(
        "kvbm candidate",
        args.kvbm_candidate.resolve(),
        args.kvbm_candidate_manifest.resolve(),
    )
    validate_artifact_pair(
        "normal parity",
        baseline,
        candidate,
        required_features={REPLAY_BENCH_FEATURE},
        forbidden_features={KVBM_FEATURE},
    )
    validate_artifact_pair(
        "KVBM parity",
        kvbm_baseline,
        kvbm_candidate,
        required_features={REPLAY_BENCH_FEATURE, KVBM_FEATURE},
        forbidden_features=set(),
    )
    validate_feature_variant(baseline, kvbm_baseline, added_feature=KVBM_FEATURE)
    validate_feature_variant(candidate, kvbm_candidate, added_feature=KVBM_FEATURE)

    results["iterations_per_revision"] = 20
    results["binaries"] = {
        "baseline": baseline.result_record(),
        "candidate": candidate.result_record(),
        "kvbm_baseline": kvbm_baseline.result_record(),
        "kvbm_candidate": kvbm_candidate.result_record(),
    }
    results["matrices"] = {}
    trace = args.trace.resolve()
    validate_trace(trace, results)
    with tempfile.TemporaryDirectory(prefix="offline-replay-parity-") as directory:
        root = Path(directory)
        run_parity_matrix(
            "normal",
            baseline,
            candidate,
            trace,
            root,
            results,
            kvbm_stress=False,
        )
        run_parity_matrix(
            "kvbm_stress",
            kvbm_baseline,
            kvbm_candidate,
            trace,
            root,
            results,
            kvbm_stress=True,
        )

    results["status"] = "pass"
    return 0


def elapsed_sample(
    binary: Path,
    trace: Path,
    config: ReplayConfig,
    output: Path,
    *,
    expected_replay_bench: bool,
) -> float:
    command = [
        *base_command(binary, trace, config),
        "--iterations",
        "1",
        "--timings-jsonl",
        str(output),
    ]
    run_checked(command)
    records = output.read_text().splitlines()
    if len(records) != 1:
        raise RuntimeError(f"expected one timing record from {' '.join(command)}")
    record = json.loads(records[0])
    if record.get("replay_bench") is not expected_replay_bench:
        raise ValidationFailure(
            f"{' '.join(command)} emitted replay_bench={record.get('replay_bench')!r}, "
            f"expected {expected_replay_bench}"
        )
    return float(record["wall_time_ms"])


def bootstrap_interval(
    baseline: list[float], candidate: list[float], seed: int
) -> tuple[float, float, float]:
    import numpy as np
    from scipy.stats import bootstrap

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)

    def median_ratio(base: Any, contender: Any) -> float:
        return float(np.median(contender / base))

    observed = median_ratio(baseline_array, candidate_array)
    interval = bootstrap(
        (baseline_array, candidate_array),
        median_ratio,
        paired=True,
        vectorized=False,
        n_resamples=100_000,
        confidence_level=0.90,
        method="percentile",
        rng=np.random.default_rng(seed),
    ).confidence_interval
    return observed, float(interval.low), float(interval.high)


def measure_configuration(
    baseline: Path,
    candidate: Path,
    trace: Path,
    config: ReplayConfig,
    root: Path,
    seed: int,
    result: dict[str, Any],
    *,
    expected_replay_bench: bool,
) -> dict[str, Any]:
    order_rng = random.Random(seed)
    result.update(
        {
            "status": "running",
            "pairs": 0,
            "warmups_per_revision": WARMUPS,
            "completed_warmup_pairs": 0,
            "baseline_wall_time_ms": [],
            "candidate_wall_time_ms": [],
            "arguments": config.arguments(),
        }
    )
    for warmup in range(WARMUPS):
        order = [("baseline", baseline), ("candidate", candidate)]
        order_rng.shuffle(order)
        for revision, binary in order:
            elapsed_sample(
                binary,
                trace,
                config,
                root / f"warmup-{warmup}-{revision}.jsonl",
                expected_replay_bench=expected_replay_bench,
            )
        result["completed_warmup_pairs"] = warmup + 1

    samples = {
        "baseline": result["baseline_wall_time_ms"],
        "candidate": result["candidate_wall_time_ms"],
    }
    target_pairs = INITIAL_PAIRS
    pair_index = 0
    while pair_index < target_pairs:
        order = [("baseline", baseline), ("candidate", candidate)]
        order_rng.shuffle(order)
        for revision, binary in order:
            sample = elapsed_sample(
                binary,
                trace,
                config,
                root / f"pair-{pair_index}-{revision}.jsonl",
                expected_replay_bench=expected_replay_bench,
            )
            samples[revision].append(sample)
        pair_index += 1
        result["pairs"] = pair_index

        if pair_index != target_pairs:
            continue
        observed, lower, upper = bootstrap_interval(
            samples["baseline"], samples["candidate"], seed + 1
        )
        if target_pairs == INITIAL_PAIRS and lower <= REGRESSION_LIMIT < upper:
            target_pairs = MAX_PAIRS

    observed, lower, upper = bootstrap_interval(
        samples["baseline"], samples["candidate"], seed + 1
    )
    if upper <= REGRESSION_LIMIT:
        status = "pass"
    elif lower > REGRESSION_LIMIT:
        status = "fail"
    else:
        status = "inconclusive"
    result.update(
        {
            "status": status,
            "median_candidate_over_baseline": observed,
            "one_sided_95_lower_bound": lower,
            "one_sided_95_upper_bound": upper,
        }
    )
    return result


def parse_text_size(output: str, section_name: str, binary: Path) -> int:
    matches = []
    for line in output.splitlines():
        columns = line.split()
        if columns and columns[0] == section_name:
            try:
                matches.append(int(columns[1]))
            except (IndexError, ValueError) as error:
                raise ValidationFailure(
                    f"invalid {section_name} size output for {binary}: {line}"
                ) from error
    if len(matches) != 1:
        raise ValidationFailure(
            f"expected one {section_name} section for {binary}, found {len(matches)}"
        )
    return matches[0]


def text_size(binary: Path, target: str) -> int:
    if platform.system() == "Darwin":
        architecture = target.split("-", maxsplit=1)[0]
        architecture = {"aarch64": "arm64"}.get(architecture, architecture)
        command = [
            "xcrun",
            "llvm-size",
            "-A",
            "-arch",
            architecture,
            str(binary),
        ]
        section_name = "__text"
    else:
        command = ["size", "-A", str(binary)]
        section_name = ".text"

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValidationFailure(
            f"failed to inspect text size for {binary} with {' '.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return parse_text_size(completed.stdout, section_name, binary)


def performance(args: argparse.Namespace, results: dict[str, Any]) -> int:
    baseline = load_build_artifact(
        "baseline",
        args.baseline.resolve(),
        args.baseline_manifest.resolve(),
    )
    candidate = load_build_artifact(
        "candidate",
        args.candidate.resolve(),
        args.candidate_manifest.resolve(),
    )
    validate_artifact_pair(
        "authoritative performance",
        baseline,
        candidate,
        required_features=set(),
        forbidden_features={REPLAY_BENCH_FEATURE, KVBM_FEATURE},
    )
    results["binaries"] = {
        "baseline": baseline.result_record(),
        "candidate": candidate.result_record(),
    }

    trace = args.trace.resolve()
    validate_trace(trace, results)
    baseline_size = baseline.binary.stat().st_size
    candidate_size = candidate.binary.stat().st_size
    target = baseline.manifest["target"]
    baseline_text = text_size(baseline.binary, target)
    candidate_text = text_size(candidate.binary, target)
    sizes = {
        "baseline_binary_bytes": baseline_size,
        "candidate_binary_bytes": candidate_size,
        "binary_ratio": candidate_size / baseline_size,
        "baseline_text_bytes": baseline_text,
        "candidate_text_bytes": candidate_text,
        "text_ratio": candidate_text / baseline_text,
    }
    sizes["status"] = (
        "pass"
        if sizes["binary_ratio"] <= REGRESSION_LIMIT
        and sizes["text_ratio"] <= REGRESSION_LIMIT
        else "fail"
    )

    results.update(
        {
            "regression_limit": REGRESSION_LIMIT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "order_seed": ORDER_SEED,
            "sizes": sizes,
            "configurations": {},
        }
    )
    seeded_values = (
        args.seeded_baseline,
        args.seeded_candidate,
        args.seeded_baseline_manifest,
        args.seeded_candidate_manifest,
    )
    if any(value is not None for value in seeded_values) and not all(
        value is not None for value in seeded_values
    ):
        raise ValidationFailure(
            "seeded baseline/candidate binaries and manifests must be provided together"
        )

    seeded_baseline = None
    seeded_candidate = None
    if all(value is not None for value in seeded_values):
        seeded_baseline = load_build_artifact(
            "seeded baseline",
            args.seeded_baseline.resolve(),
            args.seeded_baseline_manifest.resolve(),
        )
        seeded_candidate = load_build_artifact(
            "seeded candidate",
            args.seeded_candidate.resolve(),
            args.seeded_candidate_manifest.resolve(),
        )
        validate_artifact_pair(
            "seeded KV diagnostics",
            seeded_baseline,
            seeded_candidate,
            required_features={REPLAY_BENCH_FEATURE},
            forbidden_features={KVBM_FEATURE},
        )
        validate_feature_variant(
            baseline, seeded_baseline, added_feature=REPLAY_BENCH_FEATURE
        )
        validate_feature_variant(
            candidate, seeded_candidate, added_feature=REPLAY_BENCH_FEATURE
        )
        results["binaries"]["seeded_baseline"] = seeded_baseline.result_record()
        results["binaries"]["seeded_candidate"] = seeded_candidate.result_record()
        results["seeded_kv_diagnostics"] = {}

    with tempfile.TemporaryDirectory(prefix="offline-replay-performance-") as directory:
        root = Path(directory)
        for index, config in enumerate(CONFIGURATIONS):
            config_root = root / config.name
            config_root.mkdir()
            config_result: dict[str, Any] = {}
            results["configurations"][config.name] = config_result
            try:
                measure_configuration(
                    baseline.binary,
                    candidate.binary,
                    trace,
                    config,
                    config_root,
                    ORDER_SEED + index,
                    config_result,
                    expected_replay_bench=False,
                )
            except Exception:
                config_result["status"] = "error"
                raise
        if seeded_baseline is not None and seeded_candidate is not None:
            for index, config in enumerate(
                configuration
                for configuration in CONFIGURATIONS
                if configuration.router_mode == "kv-router"
            ):
                config_root = root / f"seeded-{config.name}"
                config_root.mkdir()
                config_result = {}
                results["seeded_kv_diagnostics"][config.name] = config_result
                try:
                    measure_configuration(
                        seeded_baseline.binary,
                        seeded_candidate.binary,
                        trace,
                        config,
                        config_root,
                        ORDER_SEED + 100 + index,
                        config_result,
                        expected_replay_bench=True,
                    )
                except Exception:
                    config_result["status"] = "error"
                    raise

    statuses = [value["status"] for value in results["configurations"].values()]
    size_pass = sizes["status"] == "pass"
    results["status"] = (
        "pass" if size_pass and all(status == "pass" for status in statuses) else "fail"
    )
    return 0 if results["status"] == "pass" else 1


def parser() -> argparse.ArgumentParser:
    manifest_example = """
Build manifest schema (all fields required):
  {
    "schema_version": 1,
    "source_revision": "<exact commit/tree/patch identity>",
    "binary_sha256": "<64 lowercase hex characters>",
    "build_host": "<host identity shared by the pair>",
    "rustc": "<complete rustc -Vv identity>",
    "target": "<Rust target triple>",
    "profile": "release",
    "features": ["replay-bench", "mocker-kvbm-offload"],
    "cargo_flags": ["<semantic build flags, excluding output paths>"]
  }
The validator requires identical build_host/rustc/target/profile/features/cargo_flags
within each A/B pair and binds each manifest to its binary checksum. Parity normal
binaries require replay-bench only; parity KVBM binaries add mocker-kvbm-offload.
Authoritative performance binaries forbid both features; optional seeded diagnostics
add replay-bench only.
"""
    argument_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=manifest_example,
    )
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    parity_command = subparsers.add_parser("parity")
    performance_command = subparsers.add_parser("performance")
    for command, function in (
        (parity_command, parity),
        (performance_command, performance),
    ):
        command.add_argument("--baseline", type=Path, required=True)
        command.add_argument("--candidate", type=Path, required=True)
        command.add_argument("--baseline-manifest", type=Path, required=True)
        command.add_argument("--candidate-manifest", type=Path, required=True)
        command.add_argument("--trace", type=Path, required=True)
        command.add_argument("--results", type=Path, required=True)
        command.set_defaults(function=function)
    parity_command.add_argument("--kvbm-baseline", type=Path, required=True)
    parity_command.add_argument("--kvbm-candidate", type=Path, required=True)
    parity_command.add_argument("--kvbm-baseline-manifest", type=Path, required=True)
    parity_command.add_argument("--kvbm-candidate-manifest", type=Path, required=True)
    performance_command.add_argument("--seeded-baseline", type=Path)
    performance_command.add_argument("--seeded-candidate", type=Path)
    performance_command.add_argument("--seeded-baseline-manifest", type=Path)
    performance_command.add_argument("--seeded-candidate-manifest", type=Path)
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    results: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "validation": args.command,
        "status": "running",
        "inputs": {
            key: str(value.resolve())
            for key, value in vars(args).items()
            if isinstance(value, Path) and key != "results"
        },
    }
    exit_code = 1
    try:
        exit_code = args.function(args, results)
    except ValidationFailure as error:
        results["status"] = "fail"
        results["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    except Exception as error:
        results["status"] = "error"
        results["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    result_path = args.results.resolve()
    write_results(result_path, results)
    output = json.dumps(results, sort_keys=True)
    print(output, file=sys.stdout if exit_code == 0 else sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
