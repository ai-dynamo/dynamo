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


def recipe_image_errors(artifacts, deploy_paths, label):
    if artifacts is None:
        return []
    if not isinstance(artifacts, dict):
        return ["[%s] artifacts must be an object" % label]
    errors = []
    configured_images = artifacts.get("recipe_specific_images")
    if configured_images is not None and not isinstance(configured_images, list):
        errors.append("[%s] artifacts.recipe_specific_images must be an array" % label)
        configured_images = []
    current_images = configured_images or []
    deployed_images = _deploy_images(deploy_paths)
    for image in current_images:
        if image not in deployed_images:
            errors.append(
                "[%s] recipe-specific image is not referenced by a deploy asset: %s"
                % (label, image)
            )

    periods = artifacts.get("recipe_specific_image_periods")
    if not isinstance(periods, list):
        errors.append(
            "[%s] artifacts.recipe_specific_image_periods must be an array" % label
        )
        return errors
    open_images = set()
    for period in periods:
        if not isinstance(period, dict):
            errors.append("[%s] recipe-specific image period must be an object" % label)
            continue
        image = period.get("image")
        effective_from = period.get("effective_from")
        effective_to = period.get("effective_to")
        source_revision = period.get("source_revision")
        if not isinstance(image, str):
            errors.append("[%s] recipe-specific image period is missing image" % label)
        if effective_from is not None and not _is_iso_date(effective_from):
            errors.append(
                "[%s] recipe-specific image period has invalid effective_from" % label
            )
        if effective_to is not None and (not _is_iso_date(effective_to)):
            errors.append(
                "[%s] recipe-specific image period has invalid effective_to" % label
            )
        if (
            _is_iso_date(effective_from)
            and _is_iso_date(effective_to)
            and effective_to < effective_from
        ):
            errors.append(
                "[%s] recipe-specific image period ends before it starts" % label
            )
        if not isinstance(source_revision, str) or not _REVISION_RE.fullmatch(
            source_revision
        ):
            errors.append(
                "[%s] recipe-specific image period has invalid source_revision" % label
            )
        if isinstance(image, str) and effective_to is None:
            open_images.add(image)
    current_set = {image for image in current_images if isinstance(image, str)}
    if current_set != open_images:
        errors.append(
            "[%s] current recipe-specific images must match open ownership periods"
            % label
        )
    return errors


def recipe_image_ownership_errors(entries):
    legacy_owners = {}
    periods_by_image = {}
    for recipe_id, obj in entries.items():
        if not isinstance(obj, dict):
            continue
        artifacts = obj.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        configured_images = artifacts.get("recipe_specific_images")
        periods = artifacts.get("recipe_specific_image_periods")
        if isinstance(periods, list):
            for period in periods:
                if not isinstance(period, dict):
                    continue
                image = period.get("image")
                start = period.get("effective_from")
                end = period.get("effective_to")
                if isinstance(image, str) and (start is None or isinstance(start, str)):
                    periods_by_image.setdefault(image, []).append(
                        (
                            start or "0001-01-01",
                            end if isinstance(end, str) else None,
                            recipe_id,
                        )
                    )
        elif isinstance(configured_images, list):
            for image in configured_images:
                if isinstance(image, str):
                    legacy_owners.setdefault(image, set()).add(recipe_id)

    errors = []
    for image, recipe_ids in sorted(legacy_owners.items()):
        if len(recipe_ids) > 1:
            errors.append(
                "[recipes] recipe-specific image declared by multiple recipes: %s (%s)"
                % (image, ", ".join(sorted(recipe_ids)))
            )
    for image, periods in sorted(periods_by_image.items()):
        ordered = sorted(periods)
        for index, (start, end, recipe_id) in enumerate(ordered):
            upper = end or "9999-12-31"
            for other_start, other_end, other_recipe_id in ordered[index + 1 :]:
                other_upper = other_end or "9999-12-31"
                if start <= other_upper and other_start <= upper:
                    errors.append(
                        "[recipes] overlapping ownership periods for %s (%s, %s)"
                        % (image, recipe_id, other_recipe_id)
                    )
    return errors
