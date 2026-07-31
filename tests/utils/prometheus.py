# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small Prometheus text exposition helpers for tests."""

import re

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')


def _parse_labels(labels_text: str | None) -> dict[str, str]:
    if not labels_text:
        return {}
    return {
        match.group("key"): bytes(match.group("value"), "utf-8")
        .decode("unicode_escape")
        .replace(r"\/", "/")
        for match in _LABEL_RE.finditer(labels_text)
    }


def sum_metric_samples(
    content: str,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Sum Prometheus samples matching a metric name and label subset."""
    expected_labels = labels or {}
    total = 0.0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = _SAMPLE_RE.match(line)
        if not match or match.group("name") != metric_name:
            continue

        sample_labels = _parse_labels(match.group("labels"))
        if any(
            sample_labels.get(key) != value for key, value in expected_labels.items()
        ):
            continue

        total += float(match.group("value"))

    return total
