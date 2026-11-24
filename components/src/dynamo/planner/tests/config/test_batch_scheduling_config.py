# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused validation tests for native batch scheduling configuration."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from dynamo.planner.config.planner_config import PlannerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def _clear_batch_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_PLANNER_BATCH_REDIS_URL", raising=False)


def _batch_config() -> dict[str, object]:
    return {
        "enabled": True,
        "gateway": {
            "base_url": "http://batch-gateway-apiserver:8000",
            "tenant": "planner-poc",
        },
        "metrics": {
            "frontend_metrics_url": "http://frontend:8000/metrics",
            "dispatcher_metrics_url": "http://llm-d-async:9090/metrics",
            "online_match_labels": {
                "model": "test-model",
                "request_type": "stream",
            },
        },
        "redis": {
            "url": "redis://batch-gateway-valkey:6379/0",
            "control_key": "llm-d-async:drain-limit:dynamo-batch",
        },
        "pool": {
            "pool_id": "dynamo-batch",
            "work_class": "gsm8k-128",
            "safe_rps_per_ready_replica": 10.0,
            "cold_start_margin_seconds": 15.0,
            "finalization_margin_seconds": 5.0,
            "max_observation_age_seconds": 20.0,
            "drain_lease_duration_seconds": 120.0,
            "min_replicas": 1,
            "max_replicas": 8,
            "scale_from_zero_replicas": 2,
            "max_batch_admission_rps": 25.0,
        },
    }


def _planner_config(
    batch_config: dict[str, object] | None = None,
    **overrides: object,
) -> PlannerConfig:
    raw: dict[str, object] = {
        "namespace": "test-ns",
        "environment": "kubernetes",
        "mode": "agg",
        "batch_scheduling": _batch_config() if batch_config is None else batch_config,
    }
    raw.update(overrides)
    return PlannerConfig.model_validate(raw)


def test_batch_scheduling_is_disabled_and_empty_by_default() -> None:
    config = PlannerConfig(namespace="test-ns")

    assert config.batch_scheduling.enabled is False
    assert config.batch_scheduling.gateway is None
    assert config.batch_scheduling.metrics is None
    assert config.batch_scheduling.redis is None
    assert config.batch_scheduling.pool is None


def test_enabled_batch_config_maps_exactly_to_policy_config() -> None:
    config = _planner_config()

    policy = config.batch_scheduling.to_policy_config()

    assert policy.pool_id == "dynamo-batch"
    assert policy.work_class == "gsm8k-128"
    assert policy.safe_rps_per_ready_replica == 10.0
    assert policy.cold_start_margin_s == 15.0
    assert policy.finalization_margin_s == 5.0
    assert policy.max_observation_age_s == 20.0
    assert policy.drain_lease_duration_s == 120.0
    assert policy.min_replicas == 1
    assert policy.max_replicas == 8
    assert policy.scale_from_zero_replicas == 2
    assert policy.max_batch_admission_rps == 25.0


def test_batch_pool_scale_from_zero_defaults_to_one_replica() -> None:
    batch = _batch_config()
    pool = batch["pool"]
    assert isinstance(pool, dict)
    pool.pop("scale_from_zero_replicas")

    config = _planner_config(batch)

    assert config.batch_scheduling.pool is not None
    assert config.batch_scheduling.pool.scale_from_zero_replicas == 1
    assert config.batch_scheduling.to_policy_config().scale_from_zero_replicas == 1


@pytest.mark.parametrize("scale_from_zero_replicas", [0, 9])
def test_batch_pool_rejects_invalid_scale_from_zero_floor(
    scale_from_zero_replicas: int,
) -> None:
    batch = _batch_config()
    pool = batch["pool"]
    assert isinstance(pool, dict)
    pool["scale_from_zero_replicas"] = scale_from_zero_replicas

    with pytest.raises(ValidationError, match="scale_from_zero_replicas"):
        _planner_config(batch)


def test_direct_metrics_config_requires_explicit_online_selector() -> None:
    batch = _batch_config()
    metrics = batch["metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("online_match_labels")

    with pytest.raises(ValidationError, match="online_match_labels"):
        _planner_config(batch)


def test_batch_gateway_has_bounded_observation_defaults() -> None:
    config = _planner_config()

    assert config.batch_scheduling.gateway is not None
    assert config.batch_scheduling.gateway.max_pages == 100
    assert config.batch_scheduling.gateway.max_jobs == 10_000
    assert config.batch_scheduling.gateway.collection_timeout_seconds == 15.0
    assert config.batch_scheduling.gateway.detail_concurrency == 16


def test_redis_url_loads_from_env_and_is_excluded_from_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_url = "redis://batch-gateway-valkey:6379/0"
    monkeypatch.setenv("DYN_PLANNER_BATCH_REDIS_URL", redis_url)
    batch = _batch_config()
    redis_config = batch["redis"]
    assert isinstance(redis_config, dict)
    redis_config.pop("url")

    config = _planner_config(batch)

    assert config.batch_scheduling.redis is not None
    assert config.batch_scheduling.redis.url is not None
    assert config.batch_scheduling.redis.url.get_secret_value() == redis_url
    dumped = config.model_dump(mode="json")
    dumped_batch = dumped["batch_scheduling"]
    assert isinstance(dumped_batch, dict)
    dumped_redis = dumped_batch["redis"]
    assert isinstance(dumped_redis, dict)
    assert "url" not in dumped_redis
    assert redis_url not in config.model_dump_json()


def test_redis_url_default_from_env_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DYN_PLANNER_BATCH_REDIS_URL", "not-a-redis-url")
    batch = _batch_config()
    redis_config = batch["redis"]
    assert isinstance(redis_config, dict)
    redis_config.pop("url")

    with pytest.raises(ValidationError, match="absolute redis\\(s\\) URL"):
        _planner_config(batch)


@pytest.mark.parametrize("missing", ["gateway", "metrics", "pool"])
def test_enabled_batch_config_requires_every_subtree(missing: str) -> None:
    batch = _batch_config()
    batch.pop(missing)

    with pytest.raises(ValidationError, match=missing):
        _planner_config(batch)


def test_enabled_batch_config_requires_redis_url_or_env() -> None:
    batch = _batch_config()
    redis_config = batch["redis"]
    assert isinstance(redis_config, dict)
    redis_config.pop("url")

    with pytest.raises(ValidationError, match="DYN_PLANNER_BATCH_REDIS_URL"):
        _planner_config(batch)


def test_enabled_non_advisory_batch_config_requires_redis_subtree() -> None:
    batch = _batch_config()
    batch.pop("redis")

    with pytest.raises(ValidationError, match="requires: redis"):
        _planner_config(batch)


@pytest.mark.parametrize("omit_redis", [False, True])
def test_advisory_batch_config_does_not_require_redis(
    omit_redis: bool,
) -> None:
    batch = _batch_config()
    if omit_redis:
        batch.pop("redis")
    else:
        redis_config = batch["redis"]
        assert isinstance(redis_config, dict)
        redis_config.pop("url")

    config = _planner_config(batch, advisory=True)

    assert config.advisory is True
    if omit_redis:
        assert config.batch_scheduling.redis is None


def test_enabled_batch_scheduling_is_kubernetes_only() -> None:
    with pytest.raises(ValidationError, match="environment='kubernetes'"):
        _planner_config(environment="virtual")


def test_enabled_batch_scheduling_is_aggregate_only() -> None:
    with pytest.raises(ValidationError, match="mode='agg'"):
        _planner_config(mode="disagg")


def test_batch_pool_ceiling_covers_effective_aggregate_minimum() -> None:
    batch = _batch_config()
    pool = batch["pool"]
    assert isinstance(pool, dict)
    pool["max_replicas"] = 2

    with pytest.raises(ValidationError, match="effective aggregate endpoint minimum"):
        _planner_config(batch, min_endpoint=3)


@pytest.mark.parametrize(
    ("tick_max_duration", "scale_interval", "metrics_timeout", "minimum_lease"),
    [
        (20.0, 5.0, 5.0, 76.0),
        (2.0, 5.0, 5.0, 40.0),
        (2.0, 5.0, 30.0, 55.0),
    ],
)
def test_drain_lease_covers_tick_deadline_and_two_tick_intervals(
    tick_max_duration: float,
    scale_interval: float,
    metrics_timeout: float,
    minimum_lease: float,
) -> None:
    batch = _batch_config()
    pool = batch["pool"]
    assert isinstance(pool, dict)
    pool["drain_lease_duration_seconds"] = minimum_lease - 0.001
    metrics = batch["metrics"]
    assert isinstance(metrics, dict)
    metrics["request_timeout_seconds"] = metrics_timeout

    with pytest.raises(ValidationError, match=f"{minimum_lease}s"):
        _planner_config(
            batch,
            scheduling={
                "tick_max_duration_seconds": tick_max_duration,
                "scale_interval_seconds": scale_interval,
            },
        )

    pool["drain_lease_duration_seconds"] = minimum_lease
    config = _planner_config(
        batch,
        scheduling={
            "tick_max_duration_seconds": tick_max_duration,
            "scale_interval_seconds": scale_interval,
        },
    )
    assert config.batch_scheduling.pool is not None
    assert config.batch_scheduling.pool.drain_lease_duration_seconds == minimum_lease


@pytest.mark.parametrize("subtree", ["batch", "gateway", "metrics", "redis", "pool"])
def test_batch_config_rejects_unknown_nested_fields(subtree: str) -> None:
    batch = deepcopy(_batch_config())
    if subtree == "batch":
        target = batch
    else:
        target = batch[subtree]
        assert isinstance(target, dict)
    target["unknown_field"] = "unsafe"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _planner_config(batch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frontend_metrics_url", "prometheus.monitoring:9090/metrics"),
        ("dispatcher_metrics_url", "ftp://llm-d-async/metrics"),
    ],
)
def test_metrics_urls_must_be_absolute_http_urls(field: str, value: str) -> None:
    batch = _batch_config()
    metrics = batch["metrics"]
    assert isinstance(metrics, dict)
    metrics[field] = value

    with pytest.raises(ValidationError, match=field):
        _planner_config(batch)


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"request_type": "stream"},
        {"not-a-label": "stream"},
        {"model": "test-model", "request_type": ""},
    ],
)
def test_online_metric_selector_must_be_explicit_and_valid(
    labels: dict[str, str],
) -> None:
    batch = _batch_config()
    metrics = batch["metrics"]
    assert isinstance(metrics, dict)
    metrics["online_match_labels"] = labels

    with pytest.raises(ValidationError, match="online_match_labels"):
        _planner_config(batch)


def test_policy_conversion_rejects_disabled_config() -> None:
    config = PlannerConfig(namespace="test-ns")

    with pytest.raises(ValueError, match="must be enabled"):
        config.batch_scheduling.to_policy_config()
