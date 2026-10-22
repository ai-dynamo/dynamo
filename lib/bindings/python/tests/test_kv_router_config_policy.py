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
WORKER_TYPES = ("aggregated", "prefill", "decode")

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


def _non_default_config(**overrides) -> KvRouterConfig:
    kwargs = {
        **NON_DEFAULT_PARAMETERS,
        "router_queue_threshold": 0.8,
        "router_queue_policy": "wspt",
        "router_track_output_blocks": True,
        "router_ttl_secs": 60.0,
        **overrides,
    }
    return KvRouterConfig(**kwargs)


@pytest.mark.parametrize(
    "config",
    [KvRouterConfig(), _non_default_config()],
    ids=["default", "non_default"],
)
def test_to_json_round_trip_is_idempotent(config: KvRouterConfig) -> None:
    text = config.to_json()
    assert isinstance(json.loads(text), dict)
    # The object has no __eq__, and the process-local fields are dropped by design, so
    # compare serialized forms.
    assert KvRouterConfig.from_json(text).to_json() == text


def test_to_json_preserves_serialized_fields() -> None:
    config = _non_default_config()
    data = json.loads(config.to_json())
    for name, value in NON_DEFAULT_PARAMETERS.items():
        assert data[name] == pytest.approx(value), name
    assert data["router_queue_threshold"] == pytest.approx(0.8)
    assert data["router_queue_policy"] == "wspt"
    assert data["router_track_output_blocks"] is True
    assert data["router_ttl_secs"] == pytest.approx(60.0)

    parsed = KvRouterConfig.from_json(config.to_json())
    assert parsed.overlap_score_credit == pytest.approx(1.5)
    assert parsed.overlap_score_credit_decay == pytest.approx(0.3)
    assert parsed.prefill_load_scale == pytest.approx(2.0)
    assert parsed.decode_active_request_weight == pytest.approx(0.4)


def test_to_json_omits_process_local_fields() -> None:
    config = KvRouterConfig(use_kv_events=False, router_approximate_cache_policy="lru")
    data = json.loads(config.to_json())
    for name in (
        "router_approximate_cache_policy",
        "router_prefill_policy",
        "router_decode_policy",
    ):
        assert name not in data
    # from_json restores the defaults of the omitted fields and still accepts the JSON.
    assert KvRouterConfig.from_json(config.to_json()).to_json() == config.to_json()


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


def test_policy_parameters_defaults() -> None:
    assert KvRouterConfig().policy_parameters("aggregated") == pytest.approx(
        {
            "overlap_score_credit": 1.0,
            "overlap_score_credit_decay": 0.0,
            "prefill_load_scale": 1.0,
            "decode_active_request_weight": 0.0,
            "host_cache_hit_weight": 0.75,
            "disk_cache_hit_weight": 0.25,
            "shared_cache_multiplier": 0.0,
            "router_temperature": 0.0,
        }
    )


def test_conditional_disagg_decode_keeps_cache_credits() -> None:
    config = _non_default_config(conditional_disagg_enabled=True)
    assert config.policy_parameters("decode") == pytest.approx(NON_DEFAULT_PARAMETERS)


def test_zero_credit_zeroes_inherited_cache_tier_weights() -> None:
    parameters = KvRouterConfig(
        overlap_score_credit=0.0, shared_cache_multiplier=0.5
    ).policy_parameters("aggregated")
    for name in CACHE_CREDIT_NAMES:
        assert parameters[name] == 0.0, name


def test_policy_parameters_follow_a_json_override() -> None:
    # The retune controller resolves a candidate as the startup JSON with candidate keys merged.
    config = _non_default_config()
    for worker_type in WORKER_TYPES:
        assert KvRouterConfig.from_json(config.to_json()).policy_parameters(
            worker_type
        ) == config.policy_parameters(worker_type)

    data = json.loads(config.to_json())
    data.update(overlap_score_credit=0.5, router_temperature=0.0)
    candidate = KvRouterConfig.from_json(json.dumps(data))
    parameters = candidate.policy_parameters("prefill")
    assert parameters["overlap_score_credit"] == pytest.approx(0.5)
    assert parameters["router_temperature"] == 0.0

    data["host_cache_hit_weight"] = 1.5
    with pytest.raises(ValueError, match="host_cache_hit_weight"):
        KvRouterConfig.from_json(json.dumps(data))


@pytest.mark.parametrize("worker_type", ["encode", "Aggregated", "agg", "bogus", ""])
def test_policy_parameters_rejects_unknown_worker_types(worker_type: str) -> None:
    with pytest.raises(ValueError, match="worker_type"):
        KvRouterConfig().policy_parameters(worker_type)


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
