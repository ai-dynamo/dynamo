# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]

HELPER = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "helm_chart_version.py"
)


@pytest.fixture(scope="module")
def set_chart_versions():
    spec = importlib.util.spec_from_file_location("helm_chart_version", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.set_chart_versions


def test_rewrites_operator_pin_but_not_third_party_pin(tmp_path, set_chart_versions):
    chart = tmp_path / "Chart.yaml"
    chart.write_text(
        "apiVersion: v2\n"
        "name: dynamo-platform\n"
        "version: 1.3.2\n"
        'appVersion: "1.3.2"\n'
        "dependencies:\n"
        "  - name: dynamo-operator\n"
        "    repository: file://components/operator\n"
        '    version: "1.3.2"\n'
        "  - name: nats\n"
        "    repository: https://nats-io.github.io/k8s/helm/charts/\n"
        "    version: 1.3.2\n"
    )

    set_chart_versions(chart, "1.4.0-dev1", expect_operator_pin=True)

    assert chart.read_text() == (
        "apiVersion: v2\n"
        "name: dynamo-platform\n"
        "version: 1.4.0-dev1\n"
        'appVersion: "1.4.0-dev1"\n'
        "dependencies:\n"
        "  - name: dynamo-operator\n"
        "    repository: file://components/operator\n"
        '    version: "1.4.0-dev1"\n'
        "  - name: nats\n"
        "    repository: https://nats-io.github.io/k8s/helm/charts/\n"
        "    version: 1.3.2\n"
    )


def test_missing_expected_operator_pin_raises_before_write(
    tmp_path, set_chart_versions
):
    # An operator entry without a version must not borrow the next entry's pin.
    chart = tmp_path / "Chart.yaml"
    deps = (
        "dependencies:\n"
        "  - name: dynamo-operator\n"
        "    repository: file://components/operator\n"
        "  - name: nats\n"
        "    version: 1.3.2\n"
    )
    original = "apiVersion: v2\nversion: 1.3.2\n" + deps
    chart.write_text(original)

    with pytest.raises(RuntimeError, match="found 0"):
        set_chart_versions(chart, "1.4.0", expect_operator_pin=True)
    assert chart.read_text() == original

    # Release callers do not require the pin.
    set_chart_versions(chart, "1.4.0")
    assert chart.read_text() == "apiVersion: v2\nversion: 1.4.0\n" + deps


def test_missing_top_level_version_raises(tmp_path, set_chart_versions):
    chart = tmp_path / "Chart.yaml"
    chart.write_text("apiVersion: v2\nname: dynamo-platform\n")

    with pytest.raises(RuntimeError, match="no top-level version/appVersion"):
        set_chart_versions(chart, "1.4.0")
