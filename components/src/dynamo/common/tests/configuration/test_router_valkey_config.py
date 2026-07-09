# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

import pytest
from dynamo._core import (
    KvRouterConfig,
    RouterConfig,
    RouterMode,
    TokenizerCacheConfig,
    normalize_router_valkey_config,
)

from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _normalized(raw_config: str) -> dict[str, object]:
    value = json.loads(normalize_router_valkey_config(raw_config))
    assert isinstance(value, dict)
    return value


def _build_router(config: KvRouterConfigBase) -> RouterConfig:
    return RouterConfig(
        RouterMode.KV,
        KvRouterConfig(**config.kv_router_kwargs()),
        router_valkey_config=config.parsed_router_valkey_config(),
    )


def test_tokenizer_binding_consumes_the_single_json_contract() -> None:
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "sentinel": {
                "urls": ["s0:26379", "s1:26379", "s2:26379"],
                "master_name": "router",
                "quorum": 2,
            },
            "tokenizer_cache": {
                "sentinel_master_name": "tokenizer",
                "scope": "shared-tokenizer",
            },
        }
    )

    TokenizerCacheConfig(raw_config)


def test_valkey_indexer_cli_and_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("DYN_ROUTER_VALKEY_CONNECTION_POOL_SIZE", "6")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--router-valkey-urls",
            "valkey://valkey-0:6379,valkey://valkey-1:6379",
            "--router-valkey-worker-events",
            "--router-valkey-authoritative-admission",
            "--router-valkey-index-scope",
            "shared-router",
            "--router-valkey-required-replica-acks",
            "1",
            "--router-valkey-sentinel-urls",
            "valkey://sentinel-0:26379,valkey://sentinel-1:26379,valkey://sentinel-2:26379",
            "--router-valkey-sentinel-master-name",
            "dynamo-primary",
            "--router-valkey-sentinel-quorum",
            "2",
            "--router-valkey-allow-insecure-plaintext",
            "--router-valkey-allow-degraded-writes",
            "--router-valkey-admission-lease-ms",
            "45000",
        ]
    )
    config = KvRouterConfigBase.from_cli_args(args)
    kwargs = config.kv_router_kwargs()

    assert kwargs["valkey_urls"] == "valkey://valkey-0:6379,valkey://valkey-1:6379"
    assert kwargs["valkey_index_scope"] == "shared-router"
    assert kwargs["valkey_connection_pool_size"] == 6
    assert kwargs["valkey_required_replica_acks"] == 1
    assert kwargs["valkey_sentinel_urls"].startswith("valkey://sentinel-0")
    assert kwargs["valkey_sentinel_master_name"] == "dynamo-primary"
    assert kwargs["valkey_sentinel_quorum"] == 2
    assert kwargs["valkey_allow_insecure_plaintext"] is True
    assert kwargs["valkey_allow_degraded_writes"] is True
    assert kwargs["valkey_worker_events"] is True
    assert kwargs["valkey_authoritative_admission"] is True
    assert kwargs["valkey_admission_lease_ms"] == 45000


def test_router_valkey_json_configures_router_and_tokenizer_sentinel_groups(
    monkeypatch,
) -> None:
    for name in (
        "DYN_TOKENIZER_CACHE",
        "DYN_TOKENIZER_CACHE_L2_URL",
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS",
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_MASTER_NAME",
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_QUORUM",
        "DYN_TOKENIZER_CACHE_L2_SCOPE",
        "DYN_TOKENIZER_CACHE_L2_TTL_SECONDS",
        "DYN_TOKENIZER_CACHE_L2_TIMEOUT_MS",
        "DYN_TOKENIZER_CACHE_L2_POOL_SIZE",
        "DYN_TOKENIZER_CACHE_L2_MAX_PENDING_WRITES",
        "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT",
        "DYN_TOKENIZER_CACHE_BYTES",
        "DYN_TOKENIZER_CACHE_EXTEND",
    ):
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://router-primary:6379"],
            "index_scope": "shared-router",
            "connection_pool_size": 6,
            "required_replica_acks": 1,
            "sentinel": {
                "urls": [
                    "valkey://sentinel-0:26379",
                    "valkey://sentinel-1:26379",
                    "valkey://sentinel-2:26379",
                ],
                "master_name": "dynamo-router",
                "quorum": 2,
            },
            "allow_degraded_writes": True,
            "worker_events": True,
            "authoritative_admission": True,
            "admission_lease_ms": 45000,
            "tokenizer_cache": {
                "enabled": True,
                "sentinel_master_name": "dynamo-tokenizer",
                "scope": "deployment-a",
                "key_prefix": "dynamo:tokenizer:v1",
                "ttl_seconds": 7200,
                "timeout_ms": 50,
                "connection_pool_size": 12,
                "max_pending_writes": 256,
                "l1_bytes": 33554432,
                "extend": False,
            },
        }
    )

    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )
    normalized = _normalized(config.tokenizer_cache_config_json())
    tokenizer = normalized["tokenizer_cache"]
    sentinel = normalized["sentinel"]
    assert isinstance(tokenizer, dict)
    assert isinstance(sentinel, dict)

    assert config.parsed_router_valkey_config() == raw_config
    _build_router(config)
    assert tokenizer["enabled"] is True
    assert normalized["allow_insecure_plaintext"] is True
    assert sentinel["urls"] == [
        "valkey://sentinel-0:26379",
        "valkey://sentinel-1:26379",
        "valkey://sentinel-2:26379",
    ]
    assert tokenizer["sentinel_master_name"] == "dynamo-tokenizer"
    assert sentinel["quorum"] == 2
    assert tokenizer["scope"] == "deployment-a"
    assert tokenizer["ttl_seconds"] == 7200
    assert tokenizer["timeout_ms"] == 50
    assert tokenizer["connection_pool_size"] == 12
    assert tokenizer["max_pending_writes"] == 256
    assert tokenizer["l1_bytes"] == 33554432
    assert tokenizer["extend"] is False


def test_router_valkey_json_overrides_legacy_valkey_flags() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://json-primary:6379"],
            "index_scope": "json-scope",
        }
    )

    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(
            [
                "--router-valkey-urls",
                "valkey://legacy-primary:6379",
                "--router-valkey-index-scope",
                "legacy-scope",
                "--router-valkey-worker-events",
                "--router-valkey-config",
                raw_config,
            ]
        )
    )

    kwargs = config.kv_router_kwargs()
    assert config.parsed_router_valkey_config() == raw_config
    assert "valkey_urls" not in kwargs
    assert "valkey_index_scope" not in kwargs
    assert "valkey_worker_events" not in kwargs
    # The combined Rust contract is applied after these legacy adapter values.
    _build_router(config)


def test_router_valkey_json_authoritative_admission_applies_runnable_preset() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://router-primary:6379"],
            "index_scope": "shared-router",
            "worker_events": True,
            "authoritative_admission": True,
        }
    )
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )

    kwargs = config.kv_router_kwargs()

    assert config.parsed_router_valkey_config() == raw_config
    # The kwargs intentionally retain ordinary router defaults; the canonical
    # Rust contract applies the authoritative preset beside its validation.
    assert kwargs["router_queue_threshold"] == 16.0
    _build_router(config)


@pytest.mark.parametrize(
    ("raw_config", "error"),
    [
        ("{", "config is invalid"),
        ("[]", "JSON object"),
        ('{"unknown": true}', "unknown field"),
        ('{"urls":[],"urls":[]}', "duplicate field"),
        ('{"urls": "valkey://router:6379"}', "expected a sequence"),
        ('{"urls": ["valkey://user@router:6379"]}', "without credentials"),
        ('{"urls": ["valkey://router:0"]}', "port in"),
        ('{"urls": ["valkey://router:not-a-port"]}', "port.*1..=65535"),
        ('{"index_scope":"bad/scope"}', "ASCII"),
        ('{"connection_pool_size":0}', "1..=64"),
        ('{"allow_degraded_writes":1}', "boolean"),
        (
            '{"tokenizer_cache":{"url":"valkey://tokenizer:6379,other:6379"}}',
            "one host:port",
        ),
        (
            '{"sentinel":{"urls":["valkey://s0:26379","s0:26379"],"master_name":"router"}}',
            "distinct",
        ),
        ('{"sentinel":[]}', "expected struct RouterValkeySentinelConfig"),
        (
            '{"sentinel":{"urls":["s0:26379"]}}',
            "missing field `master_name`",
        ),
        (
            '{"sentinel":{"urls":["s0:26379"],"master_name":"router","extra":1}}',
            "unknown field",
        ),
        (
            '{"tokenizer_cache":{"url":"valkey://tokenizer:6379/1"}}',
            "contain paths",
        ),
        (
            '{"sentinel":{"urls":["valkey://s0:26379","valkey://s1:26379"],"master_name":"router","quorum":1}}',
            "strict majority",
        ),
        (
            '{"tokenizer_cache":{"enabled":true,"sentinel_master_name":"tokenizer"}}',
            "requires top-level sentinel",
        ),
        ('{"tokenizer_cache":{"enabled":1}}', "expected a boolean"),
        (
            '{"sentinel":{"urls":["s0:26379"],"master_name":"router"},'
            '"tokenizer_cache":{"url":"tokenizer:6379","sentinel_master_name":"tokenizer"}}',
            "mutually exclusive",
        ),
        (
            '{"tokenizer_cache":{"url":"tokenizer:6379","l1_bytes":0}}',
            "1..=4294967295",
        ),
        (
            '{"tokenizer_cache":{"url":"tokenizer:6379","unknown":true}}',
            "unknown field",
        ),
    ],
)
def test_router_valkey_json_rejects_ambiguous_or_invalid_config(
    raw_config, error
) -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )

    with pytest.raises(ValueError, match=error):
        config.kv_router_kwargs()


def test_router_valkey_json_can_use_a_direct_tokenizer_cache_url() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://router-primary:6379"],
            "tokenizer_cache": {
                "url": "valkey://tokenizer-primary:6380",
            },
        }
    )
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )

    normalized = _normalized(config.tokenizer_cache_config_json())
    tokenizer = normalized["tokenizer_cache"]
    assert isinstance(tokenizer, dict)

    assert tokenizer["url"] == "valkey://tokenizer-primary:6380"
    assert tokenizer["sentinel_master_name"] is None


def test_tokenizer_only_sentinel_config_does_not_reach_router_adapter() -> None:
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "sentinel": {
                "urls": [
                    "valkey://sentinel-0:26379",
                    "valkey://sentinel-1:26379",
                    "valkey://sentinel-2:26379",
                ],
                "master_name": "unused-router-name",
                "quorum": 2,
            },
            "tokenizer_cache": {
                "sentinel_master_name": "dynamo-tokenizer",
            },
        }
    )
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )

    router = config.kv_router_kwargs()
    normalized = _normalized(config.tokenizer_cache_config_json())
    tokenizer = normalized["tokenizer_cache"]
    sentinel = normalized["sentinel"]
    assert isinstance(tokenizer, dict)
    assert isinstance(sentinel, dict)

    assert "valkey_urls" not in router
    assert "valkey_sentinel_urls" not in router
    assert "valkey_sentinel_master_name" not in router
    assert "valkey_sentinel_quorum" not in router
    _build_router(config)
    assert tokenizer["sentinel_master_name"] == "dynamo-tokenizer"
    assert sentinel["urls"][0] == "valkey://sentinel-0:26379"


def test_legacy_tokenizer_sentinel_environment_remains_supported(monkeypatch) -> None:
    monkeypatch.setenv("DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT", "true")
    monkeypatch.setenv(
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS",
        "valkey://s0:26379,valkey://s1:26379,valkey://s2:26379",
    )
    monkeypatch.setenv(
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_MASTER_NAME", "dynamo-tokenizer"
    )
    config = KvRouterConfigBase()

    raw_config = config.tokenizer_cache_config_json()
    TokenizerCacheConfig(raw_config)
    normalized = _normalized(raw_config)
    tokenizer = normalized["tokenizer_cache"]
    sentinel = normalized["sentinel"]
    assert isinstance(tokenizer, dict)
    assert isinstance(sentinel, dict)

    assert tokenizer["sentinel_master_name"] == "dynamo-tokenizer"
    assert sentinel["urls"] == [
        "valkey://s0:26379",
        "valkey://s1:26379",
        "valkey://s2:26379",
    ]
    assert sentinel["quorum"] == 2


def test_router_valkey_json_without_tokenizer_cache_ignores_legacy_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_TOKENIZER_CACHE", "1")
    monkeypatch.setenv("DYN_TOKENIZER_CACHE_L2_URL", "valkey://legacy-tokenizer:6379")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    raw_config = json.dumps(
        {
            "allow_insecure_plaintext": True,
            "urls": ["valkey://router-primary:6379"],
        }
    )
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(["--router-valkey-config", raw_config])
    )

    normalized = _normalized(config.tokenizer_cache_config_json())
    tokenizer = normalized["tokenizer_cache"]
    assert isinstance(tokenizer, dict)

    assert tokenizer["enabled"] is True
    assert tokenizer["url"] is None
    assert tokenizer["sentinel_master_name"] is None
    assert tokenizer["l1_bytes"] == 64 * 1024 * 1024
    assert tokenizer["extend"] is True
    assert os.environ["DYN_TOKENIZER_CACHE_L2_URL"] == "valkey://legacy-tokenizer:6379"


def test_router_valkey_json_can_explicitly_disable_tokenizer_cache(monkeypatch) -> None:
    monkeypatch.setenv("DYN_TOKENIZER_CACHE_L2_URL", "valkey://legacy:6379")
    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    config = KvRouterConfigBase.from_cli_args(
        parser.parse_args(
            [
                "--router-valkey-config",
                '{"tokenizer_cache":{"enabled":false}}',
            ]
        )
    )

    normalized = _normalized(config.tokenizer_cache_config_json())
    tokenizer = normalized["tokenizer_cache"]
    assert isinstance(tokenizer, dict)

    assert tokenizer["enabled"] is False
    assert tokenizer["l1_bytes"] == 64 * 1024 * 1024
    assert tokenizer["extend"] is True
    assert os.environ["DYN_TOKENIZER_CACHE_L2_URL"] == "valkey://legacy:6379"
