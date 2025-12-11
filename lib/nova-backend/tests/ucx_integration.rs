// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for UCX transport

#![cfg(feature = "ucx")]

mod common;

use common::{UcxFactory, scenarios};

#[tokio::test]
async fn test_single_message_round_trip() {
    scenarios::single_message_round_trip::<UcxFactory>().await;
}

#[tokio::test]
async fn test_bidirectional_messaging() {
    scenarios::bidirectional_messaging::<UcxFactory>().await;
}

#[tokio::test]
async fn test_multiple_messages_same_connection() {
    scenarios::multiple_messages_same_connection::<UcxFactory>().await;
}

#[tokio::test]
async fn test_response_message_type() {
    scenarios::response_message_type::<UcxFactory>().await;
}

#[tokio::test]
async fn test_event_message_type() {
    scenarios::event_message_type::<UcxFactory>().await;
}

#[tokio::test]
async fn test_ack_message_type() {
    scenarios::ack_message_type::<UcxFactory>().await;
}

#[tokio::test]
async fn test_mixed_message_types() {
    scenarios::mixed_message_types::<UcxFactory>().await;
}

#[tokio::test]
async fn test_large_payload() {
    scenarios::large_payload::<UcxFactory>().await;
}

#[tokio::test]
async fn test_empty_header_and_payload() {
    scenarios::empty_header_and_payload::<UcxFactory>().await;
}

#[tokio::test]
async fn test_cluster_mesh_communication() {
    scenarios::cluster_mesh_communication::<UcxFactory>().await;
}

#[tokio::test]
async fn test_concurrent_senders() {
    scenarios::concurrent_senders::<UcxFactory>().await;
}

#[tokio::test]
async fn test_send_to_unregistered_peer() {
    scenarios::send_to_unregistered_peer::<UcxFactory>().await;
}

#[tokio::test]
async fn test_connection_reuse() {
    scenarios::connection_reuse::<UcxFactory>().await;
}

#[tokio::test]
async fn test_graceful_shutdown() {
    scenarios::graceful_shutdown::<UcxFactory>().await;
}

#[tokio::test]
async fn test_high_throughput() {
    scenarios::high_throughput::<UcxFactory>().await;
}

#[tokio::test]
async fn test_zero_copy_efficiency() {
    scenarios::zero_copy_efficiency::<UcxFactory>().await;
}
