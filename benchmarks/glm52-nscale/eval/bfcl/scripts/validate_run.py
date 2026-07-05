#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that a BFCL generation/evaluation run is structurally complete.

The upstream generator exits successfully after recording per-case inference
exceptions.  The upstream evaluator can also produce aggregates for a subset of
the selected result files.  This checker turns both conditions into hard campaign
failures and anchors completeness to the IDs in the pinned BFCL installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_binding import BindingError, make_wrapper  # noqa: E402
from source_provenance import (  # noqa: E402
    SourceProvenanceError,
    verify_source_provenance,
)


FULL_COLLECTION = ["all_scoring"]
FULL_EXPECTED_GENERATED_COUNT = 5_217  # Includes 111 memory prerequisite turns.
FULL_EXPECTED_SCORED_COUNT = 5_106
FULL_EXPECTED_GENERATED_IDS_SHA256 = (
    "75413b25cd8994c118b53d19f3de1df9c38c4399f746abedf5a0f7e27ee6f526"
)
FULL_EXPECTED_SCORED_IDS_SHA256 = (
    "e0ed6c1ac094b2dc6c831f59d59c7e57a02c589b535b0a1f0534751ee94e0417"
)
RESULT_PREFIX = "BFCL_v4_"
BFCL_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATCH = BFCL_ROOT / "patches" / "0001-glm52-openai-chat-completions.patch"
IMMUTABLE_METADATA_FIELDS = (
    "schema_version",
    "variant",
    "mode",
    "campaign_phase",
    "run_name",
    "categories",
    "model_registry_name",
    "served_model_name",
    "endpoint",
    "bfcl_gorilla_commit",
    "bfcl_patch_sha256",
    "bfcl_source_identity",
    "endpoint_models_sha256",
    "endpoint_model",
    "temperature",
    "max_tokens",
    "num_threads",
    "include_input_log",
    "glm52_openai_extra_body",
    "glm52_openai_default_headers_sha256",
    "runtime_binding",
    "campaign_source",
    "python_environment",
)
EXPECTED_TRACKED_DIFF_SHA256 = (
    "e075c864c9054095956a198475d09e179c7d2aa9d83f76ea51792fa5cf4650c4"
)
EXPECTED_NEW_HANDLER_SHA256 = (
    "f09a1999ef861c55cfd5a230a15e6536ce2f375b800fd837a5ac7a147b1920c5"
)
EXPECTED_SOURCE_STATUS = [
    " M berkeley-function-call-leaderboard/SUPPORTED_MODELS.md",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/supported_models.py",
    "?? berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/glm52_openai.py",
]
EXPECTED_CONSTRAINTS_SHA256 = (
    "1ae31cfcb689500018f8ce0239dfe9f43e471561c9d0b795b96fad96eaa83f04"
)
EXPECTED_PACKAGE_COUNT = 141
EXPECTED_FREEZE_SHA256 = (
    "829c4dc3b72a4ec6f160fc2cc681070147dc1ea8dbeeee283c92ff3e356287a7"
)


class ContractError(RuntimeError):
    """The run or pinned dataset violates the expected campaign contract."""


@dataclass(frozen=True)
class ExpectedPopulation:
    requested_categories: tuple[str, ...]
    expanded_categories: tuple[str, ...]
    generated_by_category: dict[str, frozenset[str]]
    scored_by_category: dict[str, frozenset[str]]

    @property
    def generated_ids(self) -> frozenset[str]:
        return frozenset().union(*self.generated_by_category.values())

    @property
    def scored_ids(self) -> frozenset[str]:
        return frozenset().union(*self.scored_by_category.values())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


def load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ContractError(f"Missing generation metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    categories = metadata.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or not all(isinstance(category, str) and category for category in categories)
    ):
        raise ContractError("metadata.json categories must be a non-empty string list")
    if not isinstance(metadata.get("model_registry_name"), str):
        raise ContractError("metadata.json is missing model_registry_name")
    return metadata


def id_category(test_id: str) -> str:
    if ":" in test_id:
        test_id = test_id.split(":", 1)[0]
    if "_" not in test_id:
        raise ContractError(f"Malformed BFCL test ID: {test_id!r}")
    return test_id.rsplit("_", 1)[0]


def build_expected_population(categories: Sequence[str]) -> ExpectedPopulation:
    # Import from the installed, pinned BFCL package rather than maintaining a
    # second hand-written category list.  The caller has already verified HEAD.
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

    expanded = tuple(parse_test_category_argument(list(categories)))
    generated_by_category: dict[str, set[str]] = defaultdict(set)
    scored_by_category: dict[str, set[str]] = defaultdict(set)
    generated_seen: Counter[str] = Counter()
    scored_seen: Counter[str] = Counter()

    for category in expanded:
        for entry in load_dataset_entry(category, include_prereq=True):
            test_id = entry.get("id")
            if not isinstance(test_id, str):
                raise ContractError(
                    f"Pinned dataset category {category} has an invalid ID"
                )
            generated_seen[test_id] += 1
            generated_by_category[id_category(test_id)].add(test_id)

        for entry in load_dataset_entry(category, include_prereq=False):
            test_id = entry.get("id")
            if not isinstance(test_id, str):
                raise ContractError(
                    f"Pinned dataset category {category} has an invalid ID"
                )
            scored_seen[test_id] += 1
            scored_by_category[id_category(test_id)].add(test_id)

    duplicate_generated = sorted(
        test_id for test_id, count in generated_seen.items() if count != 1
    )
    duplicate_scored = sorted(
        test_id for test_id, count in scored_seen.items() if count != 1
    )
    if duplicate_generated or duplicate_scored:
        raise ContractError(
            "Pinned BFCL population contains duplicate IDs: "
            f"generated={duplicate_generated[:10]}, scored={duplicate_scored[:10]}"
        )

    population = ExpectedPopulation(
        requested_categories=tuple(categories),
        expanded_categories=expanded,
        generated_by_category={
            category: frozenset(ids) for category, ids in generated_by_category.items()
        },
        scored_by_category={
            category: frozenset(ids) for category, ids in scored_by_category.items()
        },
    )
    if list(categories) == FULL_COLLECTION:
        if len(population.generated_ids) != FULL_EXPECTED_GENERATED_COUNT:
            raise ContractError(
                "Pinned all_scoring generated population changed: expected "
                f"{FULL_EXPECTED_GENERATED_COUNT}, got {len(population.generated_ids)}"
            )
        if len(population.scored_ids) != FULL_EXPECTED_SCORED_COUNT:
            raise ContractError(
                "Pinned all_scoring scored population changed: expected "
                f"{FULL_EXPECTED_SCORED_COUNT}, got {len(population.scored_ids)}"
            )
        generated_hash = ids_sha256(population.generated_ids)
        if generated_hash != FULL_EXPECTED_GENERATED_IDS_SHA256:
            raise ContractError(
                "Pinned all_scoring generated ID set changed: expected SHA-256 "
                f"{FULL_EXPECTED_GENERATED_IDS_SHA256}, got {generated_hash}"
            )
        scored_hash = ids_sha256(population.scored_ids)
        if scored_hash != FULL_EXPECTED_SCORED_IDS_SHA256:
            raise ContractError(
                "Pinned all_scoring scored ID set changed: expected SHA-256 "
                f"{FULL_EXPECTED_SCORED_IDS_SHA256}, got {scored_hash}"
            )
    return population


def build_configured_population(path: Path) -> ExpectedPopulation:
    try:
        configured = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read configured population {path}: {exc}") from exc
    if not isinstance(configured, dict) or not configured:
        raise ContractError("Configured population must be a non-empty JSON object")

    ids_by_category: dict[str, frozenset[str]] = {}
    seen: Counter[str] = Counter()
    for category, raw_ids in configured.items():
        if not isinstance(category, str) or not category:
            raise ContractError("Configured population has an invalid category")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or not all(isinstance(test_id, str) and test_id for test_id in raw_ids)
        ):
            raise ContractError(f"Configured category {category} has invalid IDs")
        ids = frozenset(raw_ids)
        if len(ids) != len(raw_ids):
            raise ContractError(f"Configured category {category} has duplicate IDs")
        for test_id in ids:
            if id_category(test_id) != category:
                raise ContractError(
                    f"Configured ID {test_id} does not belong to category {category}"
                )
            seen[test_id] += 1
        ids_by_category[category] = ids
    duplicates = sorted(test_id for test_id, count in seen.items() if count != 1)
    if duplicates:
        raise ContractError(f"Configured population has duplicate IDs: {duplicates}")
    categories = tuple(configured)
    return ExpectedPopulation(
        requested_categories=categories,
        expanded_categories=categories,
        generated_by_category=ids_by_category,
        scored_by_category=ids_by_category,
    )


def ids_sha256(ids: Iterable[str]) -> str:
    payload = "".join(f"{test_id}\n" for test_id in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {field: metadata.get(field) for field in IMMUTABLE_METADATA_FIELDS}


def expected_manifest(
    population: ExpectedPopulation, bfcl_commit: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bfcl_gorilla_commit": bfcl_commit,
        "requested_categories": list(population.requested_categories),
        "expanded_categories": list(population.expanded_categories),
        "generated": {
            "count": len(population.generated_ids),
            "ids_sha256": ids_sha256(population.generated_ids),
            "ids_by_result_category": {
                category: sorted(ids)
                for category, ids in sorted(population.generated_by_category.items())
            },
        },
        "scored": {
            "count": len(population.scored_ids),
            "ids_sha256": ids_sha256(population.scored_ids),
            "ids_by_category": {
                category: sorted(ids)
                for category, ids in sorted(population.scored_by_category.items())
            },
        },
    }


def category_from_filename(path: Path, suffix: str) -> str:
    if not path.name.startswith(RESULT_PREFIX) or not path.name.endswith(suffix):
        raise ContractError(f"Unexpected BFCL file name: {path}")
    return path.name[len(RESULT_PREFIX) : -len(suffix)]


def read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"{path}:{line_number}: expected a JSON object")
                    continue
                entries.append(entry)
    except OSError as exc:
        errors.append(f"Cannot read {path}: {exc}")
    return entries


def is_inference_error(entry: dict[str, Any]) -> bool:
    result = entry.get("result")
    return bool(entry.get("traceback")) or (
        isinstance(result, str) and result.startswith("Error during inference:")
    )


def validate_generation(
    run_dir: Path,
    model_dir_name: str,
    population: ExpectedPopulation,
) -> dict[str, Any]:
    errors: list[str] = []
    result_root = run_dir / "result" / model_dir_name
    files = (
        sorted(result_root.glob("**/BFCL_v4_*_result.json"))
        if result_root.is_dir()
        else []
    )
    if not files:
        errors.append(f"No BFCL result files found under {result_root}")

    occurrences: dict[str, list[str]] = defaultdict(list)
    inference_error_ids: list[str] = []
    file_categories: Counter[str] = Counter()
    for path in files:
        category = category_from_filename(path, "_result.json")
        file_categories[category] += 1
        for entry in read_jsonl(path, errors):
            test_id = entry.get("id")
            if not isinstance(test_id, str):
                errors.append(f"{path}: result entry is missing a string ID")
                continue
            occurrences[test_id].append(str(path.relative_to(run_dir)))
            try:
                actual_category = id_category(test_id)
            except ContractError as exc:
                errors.append(str(exc))
                continue
            if actual_category != category:
                errors.append(
                    f"Result ID {test_id} is in category file {category}, expected {actual_category}"
                )
            if is_inference_error(entry):
                inference_error_ids.append(test_id)

    actual_ids = frozenset(occurrences)
    expected_ids = population.generated_ids
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    duplicate_ids = {
        test_id: sources
        for test_id, sources in sorted(occurrences.items())
        if len(sources) != 1
    }
    duplicate_files = sorted(
        category for category, count in file_categories.items() if count != 1
    )
    expected_file_categories = set(population.generated_by_category)
    actual_file_categories = set(file_categories)
    missing_file_categories = sorted(expected_file_categories - actual_file_categories)
    extra_file_categories = sorted(actual_file_categories - expected_file_categories)
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} generated IDs")
    if extra_ids:
        errors.append(f"Found {len(extra_ids)} unexpected generated IDs")
    if duplicate_ids:
        errors.append(f"Found {len(duplicate_ids)} duplicate generated IDs")
    if duplicate_files:
        errors.append(f"Found duplicate result files for categories: {duplicate_files}")
    if missing_file_categories:
        errors.append(f"Missing result categories: {missing_file_categories}")
    if extra_file_categories:
        errors.append(f"Unexpected result categories: {extra_file_categories}")
    if inference_error_ids:
        errors.append(f"Found {len(inference_error_ids)} inference-error results")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "expected_count": len(expected_ids),
        "expected_ids_sha256": ids_sha256(expected_ids),
        "actual_count": len(actual_ids),
        "actual_entry_count": sum(len(sources) for sources in occurrences.values()),
        "actual_ids_sha256": ids_sha256(actual_ids),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "duplicate_ids": duplicate_ids,
        "missing_categories": missing_file_categories,
        "extra_categories": extra_file_categories,
        "inference_error_ids": sorted(set(inference_error_ids)),
    }


def validate_scores(
    run_dir: Path,
    model_dir_name: str,
    population: ExpectedPopulation,
) -> dict[str, Any]:
    errors: list[str] = []
    score_root = run_dir / "score" / model_dir_name
    files = (
        sorted(score_root.glob("**/BFCL_v4_*_score.json"))
        if score_root.is_dir()
        else []
    )
    if not files:
        errors.append(f"No BFCL score files found under {score_root}")

    files_by_category: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        files_by_category[category_from_filename(path, "_score.json")].append(path)

    expected_categories = set(population.scored_by_category)
    actual_categories = set(files_by_category)
    missing_categories = sorted(expected_categories - actual_categories)
    extra_categories = sorted(actual_categories - expected_categories)
    if missing_categories:
        errors.append(f"Missing score categories: {missing_categories}")
    if extra_categories:
        errors.append(f"Unexpected score categories: {extra_categories}")

    category_stats: dict[str, dict[str, Any]] = {}
    scored_count = 0
    for category in sorted(expected_categories & actual_categories):
        category_files = files_by_category[category]
        if len(category_files) != 1:
            errors.append(
                f"Expected one score file for {category}, found {len(category_files)}"
            )
            continue
        path = category_files[0]
        entries = read_jsonl(path, errors)
        if not entries:
            errors.append(f"Empty score file: {path}")
            continue
        header, failures = entries[0], entries[1:]
        expected_ids = population.scored_by_category[category]
        expected_count = len(expected_ids)
        total_count = header.get("total_count")
        correct_count = header.get("correct_count")
        accuracy = header.get("accuracy")
        if not isinstance(total_count, int) or isinstance(total_count, bool):
            errors.append(f"{path}: total_count is not an integer")
        elif total_count != expected_count:
            errors.append(
                f"{path}: total_count {total_count} != expected {expected_count}"
            )
        else:
            scored_count += total_count
        if not isinstance(correct_count, int) or isinstance(correct_count, bool):
            errors.append(f"{path}: correct_count is not an integer")
        elif not 0 <= correct_count <= expected_count:
            errors.append(f"{path}: invalid correct_count {correct_count}")
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            errors.append(f"{path}: accuracy is not numeric")
        elif isinstance(correct_count, int) and expected_count:
            expected_accuracy = correct_count / expected_count
            if not math.isclose(float(accuracy), expected_accuracy, abs_tol=1e-12):
                errors.append(
                    f"{path}: accuracy {accuracy} != {correct_count}/{expected_count}"
                )

        expected_failure_count = (
            expected_count - correct_count if isinstance(correct_count, int) else None
        )
        if (
            expected_failure_count is not None
            and len(failures) != expected_failure_count
        ):
            errors.append(
                f"{path}: {len(failures)} failure rows != expected {expected_failure_count}"
            )
        failure_ids: list[str] = []
        for failure in failures:
            test_id = failure.get("id")
            if not isinstance(test_id, str):
                errors.append(f"{path}: failure row is missing a string ID")
                continue
            failure_ids.append(test_id)
            if test_id not in expected_ids:
                errors.append(f"{path}: unexpected failure ID {test_id}")
        duplicate_failure_ids = sorted(
            test_id for test_id, count in Counter(failure_ids).items() if count != 1
        )
        if duplicate_failure_ids:
            errors.append(f"{path}: duplicate failure IDs {duplicate_failure_ids}")
        category_stats[category] = {
            "expected_count": expected_count,
            "total_count": total_count,
            "correct_count": correct_count,
            "failure_count": len(failures),
        }

    if scored_count != len(population.scored_ids):
        errors.append(
            f"Scored population {scored_count} != expected {len(population.scored_ids)}"
        )
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "expected_count": len(population.scored_ids),
        "expected_ids_sha256": ids_sha256(population.scored_ids),
        "scored_count": scored_count,
        "missing_categories": missing_categories,
        "extra_categories": extra_categories,
        "categories": category_stats,
    }


def perfect_score_errors(scores: dict[str, Any]) -> list[str]:
    expected = scores.get("expected_count")
    categories = scores.get("categories")
    if not isinstance(expected, int) or not isinstance(categories, dict):
        return ["Cannot evaluate perfect-score gate from malformed score summary"]
    correct = sum(
        category.get("correct_count", 0)
        for category in categories.values()
        if isinstance(category, dict) and isinstance(category.get("correct_count"), int)
    )
    if correct != expected:
        return [
            f"Smoke score gate requires {expected}/{expected}, got {correct}/{expected}"
        ]
    return []


def verify_metadata(
    metadata: dict[str, Any],
    expected_commit: str,
    expected_variant: str,
    require_full: bool,
    expected_mode: str = "full",
    expected_campaign_phase: str = "validation",
) -> list[str]:
    errors: list[str] = []
    expected_patch_sha256 = file_sha256(ADAPTER_PATCH)
    if metadata.get("schema_version") != 6:
        errors.append(
            f"Expected metadata schema_version 6, got {metadata.get('schema_version')!r}"
        )
    if metadata.get("bfcl_gorilla_commit") != expected_commit:
        errors.append(
            "Generation metadata commit mismatch: "
            f"{metadata.get('bfcl_gorilla_commit')} != {expected_commit}"
        )
    if metadata.get("mode") != expected_mode:
        errors.append(f"Expected {expected_mode} mode, got {metadata.get('mode')!r}")
    if metadata.get("variant") != expected_variant:
        errors.append(
            f"Generation metadata variant {metadata.get('variant')!r} != {expected_variant!r}"
        )
    if metadata.get("campaign_phase") != expected_campaign_phase:
        errors.append(
            "Generation metadata campaign phase "
            f"{metadata.get('campaign_phase')!r} != {expected_campaign_phase!r}"
        )
    run_name = metadata.get("run_name")
    normalized_run_name = (
        run_name.replace("_", "-").replace(".", "-")
        if isinstance(run_name, str)
        else ""
    )
    if expected_campaign_phase not in normalized_run_name.split("-"):
        errors.append(
            f"Generation metadata run_name must contain phase {expected_campaign_phase!r}"
        )
    if metadata.get("max_tokens") != 64_000:
        errors.append(f"Expected max_tokens 64000, got {metadata.get('max_tokens')!r}")
    if metadata.get("served_model_name") != "zai-org/GLM-5.2":
        errors.append(
            "Expected served_model_name 'zai-org/GLM-5.2', got "
            f"{metadata.get('served_model_name')!r}"
        )
    if metadata.get("bfcl_patch_sha256") != expected_patch_sha256:
        errors.append(
            "BFCL adapter patch digest mismatch: "
            f"{metadata.get('bfcl_patch_sha256')!r} != {expected_patch_sha256!r}"
        )
    expected_source_identity = {
        "head": expected_commit,
        "status": EXPECTED_SOURCE_STATUS,
        "tracked_diff_sha256": EXPECTED_TRACKED_DIFF_SHA256,
        "new_handler_sha256": EXPECTED_NEW_HANDLER_SHA256,
    }
    if metadata.get("bfcl_source_identity") != expected_source_identity:
        errors.append(
            "BFCL patched checkout identity does not match the campaign source"
        )
    if metadata.get("temperature") != 0:
        errors.append(f"Expected temperature 0, got {metadata.get('temperature')!r}")
    if metadata.get("num_threads") != 16:
        errors.append(f"Expected num_threads 16, got {metadata.get('num_threads')!r}")
    if metadata.get("include_input_log") is not True:
        errors.append(
            f"Expected include_input_log true, got {metadata.get('include_input_log')!r}"
        )
    if metadata.get("glm52_openai_extra_body") is not None:
        errors.append(
            "Baseline campaign requires glm52_openai_extra_body to be unset, got "
            f"{metadata.get('glm52_openai_extra_body')!r}"
        )
    if metadata.get("glm52_openai_default_headers_sha256") is not None:
        errors.append(
            "Baseline campaign requires GLM52_OPENAI_DEFAULT_HEADERS to be unset"
        )
    endpoint_model = metadata.get("endpoint_model")
    if not isinstance(endpoint_model, dict):
        errors.append("Endpoint model identity is missing")
    elif (
        endpoint_model.get("id") != "zai-org/GLM-5.2"
        or endpoint_model.get("context_window") != 409600
    ):
        errors.append(
            "Endpoint model identity must be zai-org/GLM-5.2 with context_window 409600"
        )
    digest = metadata.get("endpoint_models_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append("endpoint_models_sha256 is not a lowercase SHA-256 digest")
    runtime_wrapper = metadata.get("runtime_binding")
    if not isinstance(runtime_wrapper, dict):
        errors.append("runtime_binding is missing from immutable generation metadata")
    else:
        content = runtime_wrapper.get("content")
        deployment = content.get("deployment") if isinstance(content, dict) else None
        if not isinstance(deployment, dict):
            errors.append("runtime_binding content is missing deployment identity")
        else:
            recipe = deployment.get("recipe")
            source_commit = (
                recipe.get("source_commit") if isinstance(recipe, dict) else None
            )
            campaign_source = metadata.get("campaign_source")
            expected_source_fields = {
                "schema_version",
                "source_commit",
                "source_clean",
                "source_changed_path_count",
                "bundle_sha256",
                "source_tree_sha256",
                "eval_tree_sha256",
                "campaign_env_sha256",
                "source_file_count",
                "eval_file_count",
            }
            if (
                not isinstance(campaign_source, dict)
                or set(campaign_source) != expected_source_fields
            ):
                errors.append("campaign_source identity is missing or invalid")
            else:
                if (
                    campaign_source.get("schema_version") != 1
                    or campaign_source.get("source_commit") != source_commit
                    or campaign_source.get("source_clean") is not True
                    or campaign_source.get("source_changed_path_count") != 0
                ):
                    errors.append(
                        "campaign_source identity does not match the deployment"
                    )
                for field in (
                    "bundle_sha256",
                    "source_tree_sha256",
                    "eval_tree_sha256",
                    "campaign_env_sha256",
                ):
                    value = campaign_source.get(field)
                    if (
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(
                            character not in "0123456789abcdef" for character in value
                        )
                    ):
                        errors.append(
                            f"campaign_source.{field} is not a SHA-256 digest"
                        )
                source_count = campaign_source.get("source_file_count")
                eval_count = campaign_source.get("eval_file_count")
                if (
                    isinstance(source_count, bool)
                    or not isinstance(source_count, int)
                    or isinstance(eval_count, bool)
                    or not isinstance(eval_count, int)
                    or eval_count < 1
                    or source_count != eval_count + 1
                ):
                    errors.append("campaign_source file counts are invalid")
            evaluator = content.get("evaluator") if isinstance(content, dict) else None
            if (
                not isinstance(evaluator, dict)
                or evaluator.get("campaign_source") != campaign_source
            ):
                errors.append(
                    "runtime_binding evaluator campaign_source differs from metadata"
                )
            try:
                expected_wrapper = make_wrapper(
                    deployment,
                    evaluator=content.get("evaluator"),
                    variant=expected_variant,
                    campaign_phase=expected_campaign_phase,
                    endpoint=metadata.get("endpoint"),
                )
            except BindingError as error:
                errors.append(str(error))
            else:
                if runtime_wrapper != expected_wrapper:
                    errors.append("runtime_binding wrapper or digest mismatch")
    python_environment = metadata.get("python_environment")
    if not isinstance(python_environment, dict) or set(python_environment) != {
        "schema_version",
        "constraints_sha256",
        "freeze_sha256",
        "package_count",
        "python",
    }:
        errors.append("BFCL Python environment lock identity is missing or invalid")
    else:
        if python_environment.get("schema_version") != 1:
            errors.append("BFCL Python environment schema_version must be 1")
        if python_environment.get("constraints_sha256") != EXPECTED_CONSTRAINTS_SHA256:
            errors.append("BFCL constraints lock digest mismatch")
        if python_environment.get("package_count") != EXPECTED_PACKAGE_COUNT:
            errors.append("BFCL package count mismatch")
        freeze_sha256 = python_environment.get("freeze_sha256")
        if freeze_sha256 != EXPECTED_FREEZE_SHA256:
            errors.append("BFCL environment freeze digest is invalid")
        if (
            not isinstance(python_environment.get("python"), str)
            or not python_environment["python"]
        ):
            errors.append("BFCL Python version identity is invalid")
    endpoint = metadata.get("endpoint")
    if not isinstance(endpoint, str):
        errors.append(f"Expected a sanitized endpoint URL, got {endpoint!r}")
    else:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path.rstrip("/")[-3:] != "/v1"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            errors.append(f"Expected a sanitized OpenAI /v1 endpoint, got {endpoint!r}")
    if require_full and metadata.get("categories") != FULL_COLLECTION:
        errors.append(
            f"Full campaign requires categories {FULL_COLLECTION}, got {metadata.get('categories')!r}"
        )
    if require_full and metadata.get("model_registry_name") != "zai-org/GLM-5.2-FC":
        errors.append(
            "Full campaign requires model_registry_name 'zai-org/GLM-5.2-FC', got "
            f"{metadata.get('model_registry_name')!r}"
        )
    return errors


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def metadata_value(run_dir: Path, field: str) -> str:
    metadata = load_metadata(run_dir)
    if field == "categories":
        return ",".join(metadata["categories"])
    if field == "model":
        return metadata["model_registry_name"]
    raise ContractError(f"Unsupported metadata field: {field}")


def validate_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    metadata = load_metadata(run_dir)
    metadata_errors = verify_metadata(
        metadata,
        args.expected_commit,
        args.expected_variant,
        args.require_full,
        args.expected_mode,
        args.expected_campaign_phase,
    )
    wrapper = metadata.get("runtime_binding")
    content = wrapper.get("content") if isinstance(wrapper, dict) else None
    deployment = content.get("deployment") if isinstance(content, dict) else None
    recipe = deployment.get("recipe") if isinstance(deployment, dict) else None
    campaign_commit = recipe.get("source_commit") if isinstance(recipe, dict) else None
    if not isinstance(campaign_commit, str):
        metadata_errors.append(
            "Cannot verify campaign source without deployment commit"
        )
    else:
        try:
            live_campaign_source = verify_source_provenance(
                args.campaign_source_metadata,
                args.campaign_source_root,
                campaign_commit,
            )
        except (OSError, json.JSONDecodeError, SourceProvenanceError) as error:
            metadata_errors.append(f"Campaign source verification failed: {error}")
        else:
            if live_campaign_source != metadata.get("campaign_source"):
                metadata_errors.append(
                    "Live campaign source differs from immutable generation metadata"
                )
    if metadata.get("run_name") != run_dir.name:
        metadata_errors.append(
            f"Generation metadata run_name {metadata.get('run_name')!r} "
            f"does not match run directory {run_dir.name!r}"
        )
    runtime_binding_path = run_dir / "runtime-binding.json"
    if not runtime_binding_path.is_file():
        metadata_errors.append(
            f"Missing runtime binding evidence: {runtime_binding_path}"
        )
    else:
        try:
            recorded_deployment = read_json(runtime_binding_path)
        except ContractError as error:
            metadata_errors.append(str(error))
        else:
            wrapper = metadata.get("runtime_binding")
            content = wrapper.get("content") if isinstance(wrapper, dict) else None
            deployment = (
                content.get("deployment") if isinstance(content, dict) else None
            )
            if recorded_deployment != deployment:
                metadata_errors.append(
                    "runtime-binding.json does not match immutable metadata"
                )
    environment_lock_path = run_dir / "environment-lock.json"
    environment_freeze_path = run_dir / "environment.freeze.txt"
    if not environment_lock_path.is_file() or not environment_freeze_path.is_file():
        metadata_errors.append("Missing BFCL environment lock evidence")
    else:
        try:
            environment_lock = read_json(environment_lock_path)
        except ContractError as error:
            metadata_errors.append(str(error))
        else:
            if environment_lock != metadata.get("python_environment"):
                metadata_errors.append(
                    "BFCL environment-lock.json does not match immutable metadata"
                )
            elif file_sha256(environment_freeze_path) != environment_lock.get(
                "freeze_sha256"
            ):
                metadata_errors.append("BFCL environment.freeze.txt digest mismatch")
    endpoint_models_path = run_dir / "endpoint-models.json"
    if not endpoint_models_path.is_file():
        metadata_errors.append(
            f"Missing endpoint model evidence: {endpoint_models_path}"
        )
    else:
        endpoint_models_sha256 = file_sha256(endpoint_models_path)
        if endpoint_models_sha256 != metadata.get("endpoint_models_sha256"):
            metadata_errors.append(
                "Endpoint model evidence digest mismatch: "
                f"{endpoint_models_sha256} != {metadata.get('endpoint_models_sha256')!r}"
            )
        try:
            endpoint_models = read_json(endpoint_models_path)
        except ContractError as error:
            metadata_errors.append(str(error))
        else:
            matching_models = [
                entry
                for entry in endpoint_models.get("data", [])
                if isinstance(entry, dict) and entry.get("id") == "zai-org/GLM-5.2"
            ]
            if len(matching_models) != 1:
                metadata_errors.append(
                    "Endpoint evidence must advertise zai-org/GLM-5.2 exactly once"
                )
            elif {
                field: matching_models[0].get(field)
                for field in ("id", "object", "owned_by", "context_window")
            } != metadata.get("endpoint_model"):
                metadata_errors.append(
                    "Endpoint model evidence does not match immutable metadata"
                )
    try:
        if args.population_config:
            population = build_configured_population(args.population_config.resolve())
            if metadata["categories"] != list(population.requested_categories):
                metadata_errors.append(
                    "Generation metadata categories do not match configured population: "
                    f"{metadata['categories']!r} != {list(population.requested_categories)!r}"
                )
        else:
            population = build_expected_population(metadata["categories"])
    except Exception as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(
            f"Cannot derive expected IDs from pinned BFCL data: {exc}"
        ) from exc

    model_dir_name = metadata["model_registry_name"].replace("/", "_")
    manifest = expected_manifest(population, args.expected_commit)
    write_json(run_dir / "expected-ids.json", manifest)

    generation = validate_generation(run_dir, model_dir_name, population)
    all_errors = [*metadata_errors, *generation["errors"]]
    output: dict[str, Any] = {
        "schema_version": 2,
        "phase": args.phase,
        "campaign_phase": metadata.get("campaign_phase"),
        "status": "pass",
        "errors": [],
        "generation": generation,
        "run_identity": {
            "metadata_sha256": file_sha256(run_dir / "metadata.json"),
            "immutable": immutable_metadata(metadata),
        },
    }
    if args.phase == "complete":
        scores = validate_scores(run_dir, model_dir_name, population)
        output["scores"] = scores
        all_errors.extend(scores["errors"])
        if args.require_perfect:
            all_errors.extend(perfect_score_errors(scores))
    output["errors"] = all_errors
    output["status"] = "pass" if not all_errors else "fail"
    validation_path = run_dir / f"{args.phase}-validation.json"
    write_json(validation_path, output)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": output["status"],
                "generated_count": generation["actual_count"],
                "inference_error_count": len(generation["inference_error_ids"]),
                "scored_count": output.get("scores", {}).get("scored_count"),
                "validation": str(validation_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser(
        "metadata-value", help="Read an immutable generation setting"
    )
    metadata_parser.add_argument("--run-dir", type=Path, required=True)
    metadata_parser.add_argument(
        "--field", choices=("categories", "model"), required=True
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument(
        "--phase", choices=("generation", "complete"), required=True
    )
    validate_parser.add_argument("--expected-commit", required=True)
    validate_parser.add_argument("--expected-variant", required=True)
    validate_parser.add_argument(
        "--expected-campaign-phase", choices=("ab", "ba", "validation"), required=True
    )
    validate_parser.add_argument(
        "--expected-mode", choices=("smoke", "full"), default="full"
    )
    validate_parser.add_argument("--campaign-source-metadata", type=Path, required=True)
    validate_parser.add_argument("--campaign-source-root", type=Path, required=True)
    validate_parser.add_argument("--population-config", type=Path)
    validate_parser.add_argument("--require-perfect", action="store_true")
    validate_parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "metadata-value":
            print(metadata_value(args.run_dir.resolve(), args.field))
            return
        raise SystemExit(validate_command(args))
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
