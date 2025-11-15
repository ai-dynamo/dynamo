// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for NATS transport
//!
//! These tests require a NATS server running on localhost:4222.
//! Start NATS server with: `nats-server` or `docker run -p 4222:4222 nats`

#[cfg(feature = "nats")]
mod common;

#[cfg(feature = "nats")]
use common::{NatsFactory, scenarios};

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_single_message_round_trip() {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        scenarios::single_message_round_trip::<NatsFactory>(),
    )
    .await;

    assert!(result.is_ok(), "Test timed out after 30 seconds");
    result.unwrap();
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_bidirectional_messaging() {
    scenarios::bidirectional_messaging::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_multiple_messages_same_connection() {
    scenarios::multiple_messages_same_connection::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_response_message_type() {
    scenarios::response_message_type::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_event_message_type() {
    scenarios::event_message_type::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_ack_message_type() {
    scenarios::ack_message_type::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_mixed_message_types() {
    scenarios::mixed_message_types::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_large_payload() {
    scenarios::large_payload::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_empty_header_and_payload() {
    scenarios::empty_header_and_payload::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_cluster_mesh_communication() {
    scenarios::cluster_mesh_communication::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_concurrent_senders() {
    scenarios::concurrent_senders::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_send_to_unregistered_peer() {
    scenarios::send_to_unregistered_peer::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_connection_reuse() {
    scenarios::connection_reuse::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_graceful_shutdown() {
    scenarios::graceful_shutdown::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_high_throughput() {
    scenarios::high_throughput::<NatsFactory>().await;
}

#[cfg(feature = "nats")]
#[tokio::test]
async fn test_zero_copy_efficiency() {
    scenarios::zero_copy_efficiency::<NatsFactory>().await;
}

// TODO: Implement health_check_online_peer scenario
// #[cfg(feature = "nats")]
// #[tokio::test]
// async fn test_health_check_online_peer() {
//     scenarios::health_check_online_peer::<NatsFactory>().await;
// }

// TODO: Implement health_check_offline_peer scenario
// #[cfg(feature = "nats")]
// #[tokio::test]
// async fn test_health_check_offline_peer() {
//     scenarios::health_check_offline_peer::<NatsFactory>().await;
// }
