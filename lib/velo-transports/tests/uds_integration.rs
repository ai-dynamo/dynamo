// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for UDS transport

#![cfg(unix)]

mod common;

use common::{UdsFactory, scenarios};

#[tokio::test]
async fn test_single_message_round_trip() {
    scenarios::single_message_round_trip::<UdsFactory>().await;
}

#[tokio::test]
async fn test_bidirectional_messaging() {
    scenarios::bidirectional_messaging::<UdsFactory>().await;
}

#[tokio::test]
async fn test_multiple_messages_same_connection() {
    scenarios::multiple_messages_same_connection::<UdsFactory>().await;
}

#[tokio::test]
async fn test_response_message_type() {
    scenarios::response_message_type::<UdsFactory>().await;
}

#[tokio::test]
async fn test_event_message_type() {
    scenarios::event_message_type::<UdsFactory>().await;
}

#[tokio::test]
async fn test_ack_message_type() {
    scenarios::ack_message_type::<UdsFactory>().await;
}

#[tokio::test]
async fn test_mixed_message_types() {
    scenarios::mixed_message_types::<UdsFactory>().await;
}

#[tokio::test]
async fn test_large_payload() {
    scenarios::large_payload::<UdsFactory>().await;
}

#[tokio::test]
async fn test_empty_header_and_payload() {
    scenarios::empty_header_and_payload::<UdsFactory>().await;
}

#[tokio::test]
async fn test_cluster_mesh_communication() {
    scenarios::cluster_mesh_communication::<UdsFactory>().await;
}

#[tokio::test]
async fn test_concurrent_senders() {
    scenarios::concurrent_senders::<UdsFactory>().await;
}

#[tokio::test]
async fn test_send_to_unregistered_peer() {
    scenarios::send_to_unregistered_peer::<UdsFactory>().await;
}

#[tokio::test]
async fn test_connection_reuse() {
    scenarios::connection_reuse::<UdsFactory>().await;
}

#[tokio::test]
async fn test_graceful_shutdown() {
    scenarios::graceful_shutdown::<UdsFactory>().await;
}

#[tokio::test]
async fn test_high_throughput() {
    scenarios::high_throughput::<UdsFactory>().await;
}

#[tokio::test]
async fn test_zero_copy_efficiency() {
    scenarios::zero_copy_efficiency::<UdsFactory>().await;
}

#[tokio::test]
async fn test_drain_rejects_messages() {
    scenarios::drain_rejects_messages::<UdsFactory>().await;
}

#[tokio::test]
async fn test_drain_accepts_responses() {
    scenarios::drain_accepts_responses::<UdsFactory>().await;
}

#[tokio::test]
async fn test_drain_accepts_events() {
    scenarios::drain_accepts_events::<UdsFactory>().await;
}

#[tokio::test]
async fn test_health_during_drain() {
    scenarios::health_during_drain::<UdsFactory>().await;
}
