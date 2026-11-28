# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual direct launch flags are checked without evaluating any env source."""

from __future__ import annotations

import pytest

from dynamo.planner.retune.live import frontend_config as fc
from dynamo.planner.tests.unit.test_live_retune_frontend_config import (
    capability,
    deployment,
)
from dynamo.planner.tests.unit.test_live_retune_state import live_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize(
    "args,field",
    [
        (["--router-session-affinity-ttl-secs=60"], "session_affinity_ttl_secs"),
        (["--router-session-affinity-ttl-s=60"], "session_affinity_ttl_secs"),
        (["--active-decode-blocks-threshold", "0.8"], "active_decode_blocks_threshold"),
        (["--active-prefill-tokens-threshold=100"], "active_prefill_tokens_threshold"),
        (
            ["--active-prefill-tokens-threshold-frac=0.8"],
            "active_prefill_tokens_threshold_frac",
        ),
        (["--load-aware"], "load_aware"),
        (["--router-policy-config=/not-read.yaml"], "router_policy_config"),
        (["--router-kv-overlap-score-weight=2"], "overlap_score_weight"),
        (["--router-kv-overlap-score-credit-decay=0.5"], "overlap_score_credit_decay"),
        (["--no-router-kv-events"], "use_kv_events"),
        (["--router-track-output-blocks"], "router_track_output_blocks"),
        (["--router-prefill-load-model=ais"], "router_prefill_load_model"),
        (["--router-approximate-cache-policy=lru"], "router_approximate_cache_policy"),
        (["--router-prefill-policy=custom"], "router_prefill_policy"),
        (["--tls-cert-path=/not-read.pem"], "tls_cert_path"),
    ],
)
def test_remaining_explicit_frontend_flags_are_checked(args, field):
    graph = deployment()
    graph["spec"]["components"][0]["podTemplate"] = {
        "spec": {"containers": [{"name": "main", "args": args}]}
    }
    report = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    assert not report.p4_ok
    assert any(field in reason for reason in report.reasons), report.reasons


def test_all_env_sources_and_references_remain_uninterpreted(monkeypatch):
    graph = deployment()
    before = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    graph["spec"]["env"] = [
        {"name": "DYN_ROUTER_MODE", "value": "random"},
        {"name": "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "value": "60"},
        {"name": "TTL", "value": "60"},
    ]
    graph["spec"]["components"][0]["podTemplate"] = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "args": ["--router-session-affinity-ttl-secs=$(TTL)"],
                    "env": [
                        {"name": "DYN_ROUTER_LOAD_AWARE", "value": "true"},
                        {
                            "name": "TTL",
                            "valueFrom": {
                                "configMapKeyRef": {"name": "settings", "key": "ttl"}
                            },
                        },
                    ],
                    "envFrom": [{"secretRef": {"name": "settings"}}],
                }
            ]
        }
    }
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "not-an-integer")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_MODEL", "invalid")
    after = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    assert before == after


def test_injected_flags_are_not_settings_sources_or_reparsed():
    graph = deployment()
    launch = capability()
    before = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=launch
    )
    graph["spec"]["components"][0]["podTemplate"] = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "args": [
                        "--router-mode=unknown",
                        "--no-router-replica-sync",
                        "--router-kv-overlap-score-credit=not-a-number",
                        "--router-prefill-load-scale=$(SCALE)",
                        "--router-temperature=99",
                    ],
                }
            ]
        }
    }
    assert (
        fc.evaluate_premises(graph, planner_config=live_config(), capability=launch)
        == before
    )


@pytest.mark.parametrize("role_templates", [False, True])
@pytest.mark.parametrize("router_flag", ["--router-mode=kv", "--router-m=kv"])
def test_explicit_worker_router_mode_is_rejected(role_templates, router_flag):
    graph = deployment()
    worker = graph["spec"]["components"][1]
    pod_template = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "command": ["python3", "-m", "dynamo.vllm"],
                    "args": [router_flag],
                }
            ]
        }
    }
    if role_templates:
        worker["roles"] = [
            {
                "name": "leader",
                "podTemplate": {"spec": {"containers": [{"name": "main"}]}},
            },
            {"name": "worker", "podTemplate": pod_template},
        ]
    else:
        worker["podTemplate"] = pod_template
    report = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    assert not report.p4_ok
    assert any(
        "worker" in reason and "--router-mode" in reason for reason in report.reasons
    )
    if role_templates:
        assert any("role worker" in reason for reason in report.reasons)


def test_explicit_empty_threshold_disables_rejection():
    graph = deployment()
    graph["spec"]["components"][0]["podTemplate"] = {
        "spec": {
            "containers": [
                {"name": "main", "args": ["--active-decode-blocks-threshold", ""]}
            ]
        }
    }
    report = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    assert report.spec_ok and report.p4_ok


def test_native_runtime_worker_and_opaque_scripts_are_not_interpreted():
    graph = deployment()
    graph["spec"]["components"][1]["podTemplate"] = {
        "spec": {
            "initContainers": [{"name": "runtime"}],
            "containers": [{"name": "main", "args": ["--router-mode=kv"]}],
        }
    }
    graph["spec"]["components"][0]["podTemplate"] = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "command": ["/bin/sh", "-c"],
                    "args": [
                        "exec python3 -m dynamo.frontend --router-session-affinity-ttl-secs=60"
                    ],
                }
            ]
        }
    }
    report = fc.evaluate_premises(
        graph, planner_config=live_config(), capability=capability()
    )
    assert report.spec_ok and report.p4_ok
