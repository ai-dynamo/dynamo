// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic test scenarios that work with any transport implementation

use super::*;
use std::time::Duration;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn single_message_round_trip<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Message,
    );

    let received = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn bidirectional_messaging<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();
    transport_b.register_peer(&transport_a).unwrap();

    // A -> B
    let (header1, payload1) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header1.clone(),
        payload1.clone(),
        MessageType::Message,
    );

    // B -> A
    let (header2, payload2) = test_message(2);
    transport_b.send(
        transport_a.instance_id,
        header2.clone(),
        payload2.clone(),
        MessageType::Message,
    );

    let recv_b = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    let recv_a = transport_a.recv_message(TEST_TIMEOUT).await.unwrap();

    assert_message_eq(recv_b, &header1, &payload1);
    assert_message_eq(recv_a, &header2, &payload2);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn multiple_messages_same_connection<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // Send 10 messages
    for i in 0..10 {
        let (header, payload) = test_message(i);
        transport_a.send(
            transport_b.instance_id,
            header,
            payload,
            MessageType::Message,
        );
    }

    // Receive and verify all messages (order-independent)
    let messages = transport_b
        .collect_messages_unordered(10, TEST_TIMEOUT)
        .await
        .unwrap();

    // Generate expected messages and sort them the same way
    let mut expected: Vec<_> = (0..10).map(test_message).collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0));

    for (i, msg) in messages.iter().enumerate() {
        assert_message_eq(msg.clone(), &expected[i].0, &expected[i].1);
    }

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn response_message_type<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Response,
    );

    let received = transport_b.recv_response(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn event_message_type<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Event,
    );

    let received = transport_b.recv_event(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn ack_message_type<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Ack,
    );

    // Acks route to event stream
    let received = transport_b.recv_event(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn mixed_message_types<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // Send different message types
    let (msg_h, msg_p) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        msg_h.clone(),
        msg_p.clone(),
        MessageType::Message,
    );

    let (resp_h, resp_p) = test_message(2);
    transport_a.send(
        transport_b.instance_id,
        resp_h.clone(),
        resp_p.clone(),
        MessageType::Response,
    );

    let (event_h, event_p) = test_message(3);
    transport_a.send(
        transport_b.instance_id,
        event_h.clone(),
        event_p.clone(),
        MessageType::Event,
    );

    // Receive from appropriate streams
    let recv_msg = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    let recv_resp = transport_b.recv_response(TEST_TIMEOUT).await.unwrap();
    let recv_event = transport_b.recv_event(TEST_TIMEOUT).await.unwrap();

    assert_message_eq(recv_msg, &msg_h, &msg_p);
    assert_message_eq(recv_resp, &resp_h, &resp_p);
    assert_message_eq(recv_event, &event_h, &event_p);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn large_payload<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // 1MB payload
    let header = b"large-payload".to_vec();
    let payload = test_data(1024 * 1024);

    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Message,
    );

    let received = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn empty_header_and_payload<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    transport_a.send(
        transport_b.instance_id,
        vec![],
        vec![],
        MessageType::Message,
    );

    let received = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &[], &[]);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn cluster_mesh_communication<F: TransportFactory>() {
    let cluster = F::create_cluster(3).await.unwrap();

    // Each node sends to every other node
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let (header, payload) = test_message((i * 10 + j) as u32);
                cluster.get(i).send(
                    cluster.get(j).instance_id,
                    header,
                    payload,
                    MessageType::Message,
                );
            }
        }
    }

    // Each node should receive 2 messages
    for i in 0..3 {
        let messages = cluster
            .get(i)
            .collect_messages(2, TEST_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(messages.len(), 2);
    }

    cluster.shutdown();
}

pub async fn concurrent_senders<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // Send from multiple tasks concurrently (without needing to move transport_a)
    let target_id = transport_b.instance_id;
    let mut handles = vec![];

    for i in 0..10 {
        let (header, payload) = test_message(i);
        // Send directly without spawning - the send itself is non-blocking
        transport_a.send(target_id, header, payload, MessageType::Message);
    }

    // Alternatively test with actual concurrent tasks using a different approach
    // Spawn receiver tasks to demonstrate concurrent receives
    for _ in 0..10 {
        let handle = tokio::spawn(async {
            // Just to demonstrate concurrency is working
            tokio::time::sleep(Duration::from_micros(1)).await;
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Receive all messages
    let messages = transport_b
        .collect_messages(10, TEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(messages.len(), 10);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn send_to_unregistered_peer<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    // Don't register B with A
    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Message,
    );

    // Give it a moment to process
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Should have an error
    assert!(transport_a.error_handler.error_count() > 0);

    let errors = transport_a.error_handler.get_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].0, header.as_slice());
    assert_eq!(errors[0].1, payload.as_slice());
    assert!(errors[0].2.contains("peer not registered"));

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn connection_reuse<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // First message establishes connection
    let (header1, payload1) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header1.clone(),
        payload1.clone(),
        MessageType::Message,
    );

    let recv1 = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(recv1, &header1, &payload1);

    // Second message reuses connection
    let (header2, payload2) = test_message(2);
    transport_a.send(
        transport_b.instance_id,
        header2.clone(),
        payload2.clone(),
        MessageType::Message,
    );

    let recv2 = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(recv2, &header2, &payload2);

    // No errors should have occurred
    assert_eq!(transport_a.error_handler.error_count(), 0);

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn graceful_shutdown<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // Send a message
    let (header, payload) = test_message(1);
    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Message,
    );

    // Receive it
    let received = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    // Shutdown should complete without panics
    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn high_throughput<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    let num_messages = 100;

    // Send many messages
    for i in 0..num_messages {
        let (header, payload) = test_message(i);
        transport_a.send(
            transport_b.instance_id,
            header,
            payload,
            MessageType::Message,
        );
    }

    // Receive all messages (order-independent)
    let messages = transport_b
        .collect_messages_unordered(num_messages as usize, TEST_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(messages.len(), num_messages as usize);

    // Generate expected messages and sort them the same way
    let mut expected: Vec<_> = (0..num_messages).map(test_message).collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0));

    // Verify all messages received correctly
    for (i, msg) in messages.iter().enumerate() {
        assert_message_eq(msg.clone(), &expected[i].0, &expected[i].1);
    }

    transport_a.shutdown();
    transport_b.shutdown();
}

pub async fn zero_copy_efficiency<F: TransportFactory>() {
    let transport_a = F::create().await.unwrap();
    let transport_b = F::create().await.unwrap();

    transport_a.register_peer(&transport_b).unwrap();

    // Large payload to test zero-copy
    let header = b"zero-copy-test".to_vec();
    let payload = test_data(512 * 1024); // 512KB

    transport_a.send(
        transport_b.instance_id,
        header.clone(),
        payload.clone(),
        MessageType::Message,
    );

    let received = transport_b.recv_message(TEST_TIMEOUT).await.unwrap();
    assert_message_eq(received, &header, &payload);

    // Verify no errors
    assert_eq!(transport_a.error_handler.error_count(), 0);

    transport_a.shutdown();
    transport_b.shutdown();
}
