# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_JSON_PATH = (
    REPO_ROOT / "deploy" / "observability" / "grafana_dashboards" / "disagg-dashboard.json"
)
CONFIGMAP_PATH = (
    REPO_ROOT / "deploy" / "observability" / "k8s" / "grafana-disagg-dashboard-configmap.yaml"
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TARGET_PANELS = {
    "Prefill Worker Processing Time": 'worker_type="prefill"',
    "Prefill Worker Throughput": 'worker_type="prefill"',
    "Component Latency - Prefill vs Decode": 'worker_type="decode"',
    "Decode Worker - Request Throughput": 'worker_type="decode"',
    "Decode Worker - Avg Request Duration": 'worker_type="decode"',
}


def _load_dashboard(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_configmap_dashboard(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return json.loads(data["data"]["disagg-dashboard.json"])


def _panel_targets_by_title(dashboard: dict) -> dict[str, list[str]]:
    panels = {}
    for panel in dashboard.get("panels", []):
        title = panel.get("title")
        if title:
            panels[title] = [
                target.get("expr", "")
                for target in panel.get("targets", [])
                if target.get("expr")
            ]
    return panels


def test_disagg_dashboard_uses_worker_type_for_prefill_and_decode_panels():
    dashboard_panels = _panel_targets_by_title(_load_dashboard(DASHBOARD_JSON_PATH))

    for title, expected_snippet in TARGET_PANELS.items():
        exprs = dashboard_panels[title]
        assert exprs, f"expected PromQL targets for {title}"
        assert any(expected_snippet in expr for expr in exprs), (
            f"expected {title} to include {expected_snippet}, got {exprs}"
        )


def test_disagg_dashboard_configmap_matches_dashboard_queries():
    dashboard_panels = _panel_targets_by_title(_load_dashboard(DASHBOARD_JSON_PATH))
    configmap_panels = _panel_targets_by_title(_load_configmap_dashboard(CONFIGMAP_PATH))

    for title in TARGET_PANELS:
        assert configmap_panels[title] == dashboard_panels[title]
