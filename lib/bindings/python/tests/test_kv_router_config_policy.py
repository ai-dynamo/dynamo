# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KvRouterConfig JSON serialization and resolved default-policy parameters."""

import json

import pytest

from dynamo.llm import KvRouterConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.router,
    pytest.mark.unit,
]

POLICY_PARAMETER_NAMES = {
    "overlap_score_credit",
    "overlap_score_credit_decay",
    "prefill_load_scale",
    "decode_active_request_weight",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "shared_cache_multiplier",
    "router_temperature",
}
CACHE_CREDIT_NAMES = (
    "overlap_score_credit",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "shared_cache_multiplier",
)

NON_DEFAULT_PARAMETERS = {
    "overlap_score_credit": 1.5,
    "overlap_score_credit_decay": 0.3,
    "prefill_load_scale": 2.0,
    "decode_active_request_weight": 0.4,
    "host_cache_hit_weight": 0.5,
    "disk_cache_hit_weight": 0.2,
    "shared_cache_multiplier": 0.3,
    "router_temperature": 0.7,
}


def _non_default_config() -> KvRouterConfig:
    return KvRouterConfig(
        **NON_DEFAULT_PARAMETERS,
        router_queue_threshold=0.8,
        router_queue_policy="wspt",
        router_track_output_blocks=True,
        router_ttl_secs=60.0,
    )


def test_to_json_round_trip_is_idempotent() -> None:
    config = _non_default_config()
    text = config.to_json()
    data = json.loads(text)
    assert {name: data[name] for name in NON_DEFAULT_PARAMETERS} == pytest.approx(
        NON_DEFAULT_PARAMETERS
    )
    assert data["router_queue_policy"] == "wspt"
    # Process-local fields are dropped by design.
    assert "router_prefill_policy" not in data
    # The object has no __eq__, so compare serialized forms.
    assert KvRouterConfig.from_json(text).to_json() == text


def test_policy_parameters_resolve_every_worker_type() -> None:
    config = _non_default_config()
    for worker_type in ("aggregated", "prefill"):
        parameters = config.policy_parameters(worker_type)
        assert set(parameters) == POLICY_PARAMETER_NAMES
        assert all(isinstance(value, float) for value in parameters.values())
        assert parameters == pytest.approx(NON_DEFAULT_PARAMETERS), worker_type

    # A plain decode worker has no cache inputs, so its cache credits are zero.
    decode = config.policy_parameters("decode")
    assert set(decode) == POLICY_PARAMETER_NAMES
    for name in CACHE_CREDIT_NAMES:
        assert decode[name] == 0.0, name
    for name in POLICY_PARAMETER_NAMES.difference(CACHE_CREDIT_NAMES):
        assert decode[name] == pytest.approx(NON_DEFAULT_PARAMETERS[name]), name

    for worker_type in ("encode", "Aggregated"):
        with pytest.raises(ValueError, match="worker_type"):
            config.policy_parameters(worker_type)


def test_policy_parameters_rejects_a_selected_policy_instance(tmp_path, monkeypatch):
    # A selected instance's own parameters (credit 3.0 here) are not resolved,
    # so the startup map is not reported for the role it serves.
    monkeypatch.delenv("DYN_ROUTER_WORKER_SELECTION_POLICY", raising=False)
    policy = tmp_path / "router_policy.yaml"
    policy.write_text(
        "worker_selection:\n"
        "  aggregated: tuned\n"
        "  instances:\n"
        "    - name: tuned\n"
        "      type: dynamo-default-cost-fn\n"
        "      parameters:\n"
        "        overlap_score_credit: 3.0\n",
        encoding="utf-8",
    )
    config = KvRouterConfig(router_policy_config=str(policy))

    with pytest.raises(ValueError, match="selects worker-selection policy instance"):
        config.policy_parameters("aggregated")
    assert config.policy_parameters("prefill") == KvRouterConfig().policy_parameters(
        "prefill"
    )

    monkeypatch.setenv("DYN_ROUTER_WORKER_SELECTION_POLICY", "tuned")
    with pytest.raises(ValueError, match="selects worker-selection policy instance"):
        KvRouterConfig().policy_parameters("decode")
