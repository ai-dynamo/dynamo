// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for TCP transport

mod common;

use common::{TcpFactory, scenarios};

#[tokio::test]
async fn test_single_message_round_trip() {
    scenarios::single_message_round_trip::<TcpFactory>().await;
}

#[tokio::test]
async fn test_bidirectional_messaging() {
    scenarios::bidirectional_messaging::<TcpFactory>().await;
}

#[tokio::test]
async fn test_multiple_messages_same_connection() {
    scenarios::multiple_messages_same_connection::<TcpFactory>().await;
}

#[tokio::test]
async fn test_response_message_type() {
    scenarios::response_message_type::<TcpFactory>().await;
}

#[tokio::test]
async fn test_event_message_type() {
    scenarios::event_message_type::<TcpFactory>().await;
}

#[tokio::test]
async fn test_ack_message_type() {
    scenarios::ack_message_type::<TcpFactory>().await;
}

#[tokio::test]
async fn test_mixed_message_types() {
    scenarios::mixed_message_types::<TcpFactory>().await;
}

#[tokio::test]
async fn test_large_payload() {
    scenarios::large_payload::<TcpFactory>().await;
}

#[tokio::test]
async fn test_empty_header_and_payload() {
    scenarios::empty_header_and_payload::<TcpFactory>().await;
}

#[tokio::test]
async fn test_cluster_mesh_communication() {
    scenarios::cluster_mesh_communication::<TcpFactory>().await;
}

#[tokio::test]
async fn test_concurrent_senders() {
    scenarios::concurrent_senders::<TcpFactory>().await;
}

#[tokio::test]
async fn test_send_to_unregistered_peer() {
    scenarios::send_to_unregistered_peer::<TcpFactory>().await;
}

#[tokio::test]
async fn test_connection_reuse() {
    scenarios::connection_reuse::<TcpFactory>().await;
}

#[tokio::test]
async fn test_graceful_shutdown() {
    scenarios::graceful_shutdown::<TcpFactory>().await;
}

#[tokio::test]
async fn test_high_throughput() {
    scenarios::high_throughput::<TcpFactory>().await;
}

#[tokio::test]
async fn test_zero_copy_efficiency() {
    scenarios::zero_copy_efficiency::<TcpFactory>().await;
}
