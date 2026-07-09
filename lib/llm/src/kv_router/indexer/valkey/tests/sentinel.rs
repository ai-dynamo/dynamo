// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{super::*, common::*};
use tokio::net::TcpListener;

#[test]
fn sentinel_config_requires_distinct_strict_majority_witnesses() {
    let config = ValkeySentinelConfig::new(
        "127.0.0.1:26379,127.0.0.1:26380,127.0.0.1:26381",
        "dynamo-primary",
        None,
    )
    .unwrap();
    assert_eq!(config.quorum, 2);
    config.validate_degraded_writes().unwrap();
    assert!(
        ValkeySentinelConfig::new("127.0.0.1:26379", "dynamo-primary", None)
            .unwrap()
            .validate_degraded_writes()
            .is_err()
    );
    assert!(
        ValkeySentinelConfig::new("127.0.0.1:26379,127.0.0.1:26379", "dynamo-primary", None,)
            .is_err()
    );
    assert!(
        ValkeySentinelConfig::new(
            "127.0.0.1:26379,127.0.0.1:26380,127.0.0.1:26381",
            "dynamo-primary",
            Some(1),
        )
        .is_err()
    );
    assert!(ValkeySentinelConfig::new("127.0.0.1:26379", "two words", None).is_err());
}

#[tokio::test]
async fn sentinel_resolver_requires_quorum_agreement() {
    let elected = "127.0.0.1:6379".to_string();
    let other = "127.0.0.1:6380".to_string();
    let (first_endpoint, first) = spawn_test_sentinel(vec![elected.clone()]).await;
    let (second_endpoint, second) = spawn_test_sentinel(vec![elected.clone()]).await;
    let (third_endpoint, third) = spawn_test_sentinel(vec![other]).await;
    let config = ValkeySentinelConfig::new(
        &[first_endpoint, second_endpoint, third_endpoint].join(","),
        "dynamo-primary",
        None,
    )
    .unwrap();

    assert_eq!(config.resolve_primary().await.unwrap(), elected);
    for server in [first, second, third] {
        server.await.unwrap();
    }
}

#[tokio::test]
async fn sentinel_resolver_rejects_a_split_vote() {
    let mut endpoints = Vec::new();
    let mut servers = Vec::new();
    for primary in ["127.0.0.1:6379", "127.0.0.1:6380", "127.0.0.1:6381"] {
        let (endpoint, server) = spawn_test_sentinel(vec![primary.to_string()]).await;
        endpoints.push(endpoint);
        servers.push(server);
    }
    let config = ValkeySentinelConfig::new(&endpoints.join(","), "dynamo-primary", None).unwrap();

    let error = config.resolve_primary().await.unwrap_err();
    assert!(error.to_string().contains("quorum not reached"));
    for server in servers {
        server.await.unwrap();
    }
}

#[tokio::test]
async fn sentinel_failover_rotates_endpoint_and_retries_command() {
    let old_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let old_endpoint = old_listener.local_addr().unwrap().to_string();
    let old_server = tokio::spawn(async move {
        accept_test_primary_role(&old_listener).await;
        let (mut stream, _) = old_listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut stream).await,
            vec![b"PING".to_vec()]
        );
        // Simulate a primary process dying after receiving the command but
        // before returning a reply.
    });

    let new_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let new_endpoint = new_listener.local_addr().unwrap().to_string();
    let new_server = tokio::spawn(async move {
        accept_test_primary_role(&new_listener).await;
        let (mut stream, _) = new_listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut stream).await,
            vec![b"PING".to_vec()]
        );
        stream.write_all(b"+PONG\r\n").await.unwrap();
    });

    let mut sentinel_endpoints = Vec::new();
    let mut sentinels = Vec::new();
    for _ in 0..3 {
        let (endpoint, server) =
            spawn_test_sentinel(vec![old_endpoint.clone(), new_endpoint.clone()]).await;
        sentinel_endpoints.push(endpoint);
        sentinels.push(server);
    }
    let sentinel =
        ValkeySentinelConfig::new(&sentinel_endpoints.join(","), "dynamo-primary", None).unwrap();
    let data_urls = format!("{old_endpoint},{new_endpoint}");
    let indexer = ValkeyIndexer::new_with_sentinel(
        &data_urls,
        1,
        Some(0),
        "test",
        "worker",
        Some("sentinel-failover"),
        None,
        16,
        CancellationToken::new(),
        sentinel,
        false,
    )
    .await
    .unwrap();
    assert_eq!(indexer.inner.primary.endpoint(), old_endpoint);
    assert_eq!(indexer.inner.primary.generation(), 0);

    let response = timeout(
        Duration::from_secs(2),
        indexer.inner.primary.command_read(&[b"PING"]),
    )
    .await
    .expect("Sentinel failover command timed out")
    .unwrap();
    assert!(matches!(response, RespValue::Simple(value) if value == "PONG"));
    assert_eq!(indexer.inner.primary.endpoint(), new_endpoint);
    assert_eq!(indexer.inner.primary.generation(), 1);

    old_server.await.unwrap();
    new_server.await.unwrap();
    for server in sentinels {
        server.await.unwrap();
    }
}

#[tokio::test]
async fn sentinel_confirmed_singleton_accepts_wait_zero_only_when_opted_in() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let data_server = tokio::spawn(async move {
        accept_test_primary_role(&listener).await;

        let (mut writer, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut writer).await,
            vec![b"DYNKV.TEST".to_vec(), b"index".to_vec()]
        );
        writer.write_all(b"+OK\r\n").await.unwrap();
        let wait = read_test_resp_request(&mut writer).await;
        assert_eq!(wait[0], b"WAIT");
        assert_eq!(wait[1], b"1");
        assert_eq!(wait[2], REPLICATION_WAIT_TIMEOUT_MS.to_string().as_bytes());
        writer.write_all(b":0\r\n").await.unwrap();

        // A degraded acknowledgement is accepted only after a fresh
        // Sentinel vote and ROLE check for this exact route generation.
        accept_test_primary_role(&listener).await;
    });

    let mut sentinel_endpoints = Vec::new();
    let mut sentinels = Vec::new();
    for _ in 0..3 {
        let (sentinel_endpoint, server) =
            spawn_test_sentinel(vec![endpoint.clone(), endpoint.clone()]).await;
        sentinel_endpoints.push(sentinel_endpoint);
        sentinels.push(server);
    }
    let sentinel =
        ValkeySentinelConfig::new(&sentinel_endpoints.join(","), "dynamo-primary", None).unwrap();
    let primary = ValkeyPrimary::new_with_sentinel(sentinel, 1, true)
        .await
        .unwrap();

    let response = primary
        .command_write_and_wait(&[b"DYNKV.TEST", b"index"], 1, false)
        .await
        .unwrap();
    assert!(matches!(response, RespValue::Simple(value) if value == "OK"));

    data_server.await.unwrap();
    for server in sentinels {
        server.await.unwrap();
    }
}

#[test]
fn degraded_failover_tuning_is_sentinel_and_generation_scoped() {
    let sentinel = ValkeySentinelConfig {
        endpoints: Arc::from(["127.0.0.1:26379".to_string()]),
        master_name: Arc::from(b"dynamo-primary".as_slice()),
        quorum: 1,
    };
    let degraded = ValkeyPrimary::with_endpoint(
        "127.0.0.1:6379".to_string(),
        1,
        Some(sentinel.clone()),
        true,
    );
    assert_eq!(
        degraded.write_retry_max_delay(),
        SENTINEL_WRITE_RETRY_MAX_DELAY
    );
    assert_eq!(
        degraded.replication_wait_timeout_ms(0),
        REPLICATION_WAIT_TIMEOUT_MS
    );
    assert_eq!(
        degraded.replication_wait_timeout_ms(1),
        DEGRADED_FAILOVER_WAIT_TIMEOUT_MS
    );
    assert!(degraded.first_degraded_warning_for_generation(1));
    assert!(!degraded.first_degraded_warning_for_generation(1));
    assert!(degraded.first_degraded_warning_for_generation(2));

    let strict =
        ValkeyPrimary::with_endpoint("127.0.0.1:6379".to_string(), 1, Some(sentinel), false);
    assert_eq!(
        strict.replication_wait_timeout_ms(1),
        REPLICATION_WAIT_TIMEOUT_MS
    );

    let fixed = ValkeyPrimary::new("127.0.0.1:6379".to_string(), 1);
    assert_eq!(fixed.write_retry_max_delay(), WRITE_RETRY_MAX_DELAY);
    assert_eq!(
        fixed.replication_wait_timeout_ms(1),
        REPLICATION_WAIT_TIMEOUT_MS
    );
}

#[test]
fn sentinel_role_validation_rejects_a_replica_candidate() {
    assert!(
        validate_primary_role_response(RespValue::Array(vec![RespValue::Bulk(b"master".to_vec())]))
            .is_ok()
    );
    assert!(
        validate_primary_role_response(RespValue::Array(vec![RespValue::Bulk(b"slave".to_vec())]))
            .is_err()
    );
}
