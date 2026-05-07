# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SMG_VALUES = ROOT / "deploy" / "production" / "addons" / "smg" / "values.yaml"
GITOPS_KUSTOMIZATION = (
    ROOT / "deploy" / "production" / "gitops" / "kustomization.yaml"
)


def test_smg_values_only_configure_http_gateway_features():
    values = yaml.safe_load(SMG_VALUES.read_text())
    router = values["router"]

    assert router["policy"] == "round_robin"
    assert router["workerUrls"] == [
        "http://deepseek-v32-reap-sglang-frontend.dynamo-system.svc.cluster.local:8000"
    ]

    assert "tokenizerPath" not in router
    assert "reasoningParser" not in router
    assert "toolCallParser" not in router
    assert "mcp" not in router
    assert router["extraEnv"] == []
    assert "history" not in values


def test_smg_root_app_does_not_deploy_history_or_tokenizer_support_apps():
    kustomization = yaml.safe_load(GITOPS_KUSTOMIZATION.read_text())
    resources = set(kustomization["resources"])

    assert "apps/70-smg.yaml" in resources
    assert "apps/65-smg-postgres.yaml" not in resources
    assert "apps/66-smg-secrets.yaml" not in resources
