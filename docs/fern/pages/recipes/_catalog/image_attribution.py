# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for effective-dated recipe image ownership."""

import re
from datetime import date

_IMAGE_FIELD_RE = re.compile(
    r"""^\s*(?:-\s*)?image:\s*(?:"([^"]+)"|'([^']+)'|([^\s#]+))\s*(?:#.*)?$"""
)
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_KINDS = {"deploy-asset", "github-release"}
_RELEASE_STATES = {"draft", "prerelease", "release"}


def _invalid_release_tag(value):
    return value is not None and (not isinstance(value, str) or not value.strip())


def _valid_choice(value, choices):
    return isinstance(value, str) and value in choices


def _is_iso_date(value):
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _deploy_images(paths):
    images = set()
    for path in paths:
        with open(path, "r") as source:
            for line in source:
                match = _IMAGE_FIELD_RE.match(line)
                if match:
                    images.add(
                        next(value for value in match.groups() if value is not None)
                    )
    return images


def _configured_images(artifacts, label):
    errors = []
    configured_images = artifacts.get("recipe_specific_images")
    if configured_images is not None and not isinstance(configured_images, list):
        errors.append("[%s] artifacts.recipe_specific_images must be an array" % label)
        configured_images = []
    elif isinstance(configured_images, list):
        if any(not isinstance(image, str) for image in configured_images):
            errors.append(
                "[%s] artifacts.recipe_specific_images entries must be strings" % label
            )
        configured_images = [
            image for image in configured_images if isinstance(image, str)
        ]
    return configured_images or [], errors


def _period_date_errors(period, label):
    start = period.get("effective_from")
    end = period.get("effective_to")
    errors = []
    if start is not None and not _is_iso_date(start):
        errors.append(
            "[%s] recipe-specific image period has invalid effective_from" % label
        )
    if end is not None and not _is_iso_date(end):
        errors.append(
            "[%s] recipe-specific image period has invalid effective_to" % label
        )
    if _is_iso_date(start) and _is_iso_date(end) and end < start:
        errors.append("[%s] recipe-specific image period ends before it starts" % label)
    return errors


def _release_field_errors(period, label):
    source_kind = period.get("source_kind")
    release_tag = period.get("release_tag")
    release_state = period.get("release_state")
    errors = []
    if source_kind is not None and not _valid_choice(source_kind, _SOURCE_KINDS):
        errors.append(
            "[%s] recipe-specific image period has invalid source_kind" % label
        )
    if _invalid_release_tag(release_tag):
        errors.append(
            "[%s] recipe-specific image period has invalid release_tag" % label
        )
    if release_state is not None and not _valid_choice(release_state, _RELEASE_STATES):
        errors.append(
            "[%s] recipe-specific image period has invalid release_state" % label
        )
    if source_kind != "github-release" and (release_tag is None) != (
        release_state is None
    ):
        errors.append(
            "[%s] release_tag and release_state must be declared together" % label
        )
    return errors


def _github_release_errors(period, label):
    if period.get("source_kind") != "github-release":
        return []
    errors = []
    release_state = period.get("release_state")
    if not period.get("release_tag"):
        errors.append("[%s] github-release image period is missing release_tag" % label)
    if not _valid_choice(release_state, _RELEASE_STATES):
        errors.append(
            "[%s] github-release image period has invalid release_state" % label
        )
    return errors


def _complete_release(period):
    tag = period.get("release_tag")
    release_state = period.get("release_state")
    return (
        period.get("source_kind") == "github-release"
        and isinstance(tag, str)
        and bool(tag.strip())
        and _valid_choice(release_state, _RELEASE_STATES)
    )


def _period_evidence(period, label):
    if not isinstance(period, dict):
        return (
            ["[%s] recipe-specific image period must be an object" % label],
            None,
            False,
            False,
        )
    image = period.get("image")
    revision = period.get("source_revision")
    errors = _period_date_errors(period, label)
    errors.extend(_release_field_errors(period, label))
    errors.extend(_github_release_errors(period, label))
    if not isinstance(image, str):
        errors.append("[%s] recipe-specific image period is missing image" % label)
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        errors.append(
            "[%s] recipe-specific image period has invalid source_revision" % label
        )
    return errors, image, period.get("effective_to") is None, _complete_release(period)


def _period_sets(periods, label):
    errors = []
    tracked_images = set()
    open_images = set()
    release_images = set()
    for period in periods:
        period_errors, image, is_open, release_complete = _period_evidence(
            period, label
        )
        errors.extend(period_errors)
        if isinstance(image, str):
            tracked_images.add(image)
            if is_open:
                open_images.add(image)
            if release_complete:
                release_images.add(image)
    return errors, tracked_images, open_images, release_images


def _deploy_evidence_errors(
    current_images, tracked_images, open_images, release_images, deploy_paths, label
):
    deployed_images = _deploy_images(deploy_paths)
    errors = [
        "[%s] recipe-specific image is not referenced by a deploy asset or "
        "complete GitHub release provenance: %s" % (label, image)
        for image in current_images
        if image not in deployed_images and image not in release_images
    ]
    errors.extend(
        "[%s] deployed recipe-specific image must have an open ownership period: %s"
        % (label, image)
        for image in sorted(deployed_images & tracked_images - open_images)
    )
    return errors


def _configured_periods(artifacts, label):
    periods = artifacts.get("recipe_specific_image_periods")
    if not isinstance(periods, list):
        return [], [
            "[%s] artifacts.recipe_specific_image_periods must be an array" % label
        ]
    if not periods:
        return [], [
            "[%s] artifacts.recipe_specific_image_periods must contain at least one item"
            % label
        ]
    return periods, []


def _evidence_errors(current_images, periods, deploy_paths, label):
    errors, tracked_images, open_images, release_images = _period_sets(periods, label)
    errors.extend(
        _deploy_evidence_errors(
            current_images,
            tracked_images,
            open_images,
            release_images,
            deploy_paths,
            label,
        )
    )
    if set(current_images) != open_images:
        errors.append(
            "[%s] current recipe-specific images must match open ownership periods"
            % label
        )
    return errors


def recipe_image_errors(artifacts, deploy_paths, label) -> list[str]:
    if artifacts is None:
        return []
    if not isinstance(artifacts, dict):
        return ["[%s] artifacts must be an object" % label]
    current_images, errors = _configured_images(artifacts, label)
    periods, period_errors = _configured_periods(artifacts, label)
    errors.extend(period_errors)
    errors.extend(_evidence_errors(current_images, periods, deploy_paths, label))
    return errors


def _period_ownership(recipe_id, periods):
    ownership = []
    for period in periods:
        if not isinstance(period, dict) or not isinstance(period.get("image"), str):
            continue
        start = period.get("effective_from")
        if isinstance(start, date):
            start = start.isoformat()
        elif not isinstance(start, str):
            start = "0001-01-01"
        end = period.get("effective_to")
        ownership.append((period["image"], start or "0001-01-01", end, recipe_id))
    return ownership


def _entry_ownership(recipe_id, obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("artifacts"), dict):
        return set(), []
    artifacts = obj["artifacts"]
    periods = artifacts.get("recipe_specific_image_periods")
    if isinstance(periods, list):
        return set(), _period_ownership(recipe_id, periods)
    images = artifacts.get("recipe_specific_images")
    if not isinstance(images, list):
        return set(), []
    return {image for image in images if isinstance(image, str)}, []


def _overlap_errors(image, periods):
    errors = []
    ordered = sorted(
        periods,
        key=lambda p: (
            p[0],
            p[1] if isinstance(p[1], str) else "9999-12-31",
            p[2],
        ),
    )
    for index, (start, end, recipe_id) in enumerate(ordered):
        upper = end if isinstance(end, str) else "9999-12-31"
        for other_start, other_end, other_recipe_id in ordered[index + 1 :]:
            other_upper = other_end if isinstance(other_end, str) else "9999-12-31"
            if start <= other_upper and other_start <= upper:
                errors.append(
                    "[recipes] overlapping ownership periods for %s (%s, %s)"
                    % (image, recipe_id, other_recipe_id)
                )
    return errors


def recipe_image_ownership_errors(entries) -> list[str]:
    legacy_owners = {}
    periods_by_image = {}
    for recipe_id, obj in entries.items():
        legacy_images, periods = _entry_ownership(recipe_id, obj)
        for image in legacy_images:
            legacy_owners.setdefault(image, set()).add(recipe_id)
        for image, start, end, owner in periods:
            periods_by_image.setdefault(image, []).append((start, end, owner))

    errors = []
    for image, recipe_ids in sorted(legacy_owners.items()):
        if len(recipe_ids) > 1:
            errors.append(
                "[recipes] recipe-specific image declared by multiple recipes: %s (%s)"
                % (image, ", ".join(sorted(recipe_ids)))
            )
    for image, periods in sorted(periods_by_image.items()):
        errors.extend(_overlap_errors(image, periods))
    return errors
