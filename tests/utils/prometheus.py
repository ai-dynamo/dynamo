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


def find_metric_samples(
    content: str,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> list[float]:
    """Return every Prometheus sample value matching a metric name and label subset.

    Use this instead of :func:`sum_metric_samples` when the assertion is about a
    metric being *registered* rather than about its value. An empty list means the
    metric is absent from the exposition, which a summed value of ``0.0`` cannot
    distinguish from a registered metric that simply has not been incremented.
    """
    expected_labels = labels or {}
    values = []

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

        values.append(float(match.group("value")))

    return values


def sum_metric_samples(
    content: str,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Sum Prometheus samples matching a metric name and label subset.

    Returns ``0.0`` both when the metric is absent and when it is present at zero;
    use :func:`find_metric_samples` if that distinction matters.
    """
    return sum(find_metric_samples(content, metric_name, labels))
