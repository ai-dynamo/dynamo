// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[allow(unused_imports)]
use super::*;

#[cfg(all(test, feature = "integration"))]
mod test_integration_publisher {
    use super::*;
    use crate::kv_router::KV_METRICS_SUBJECT;
    use dynamo_kv_router::protocols::ActiveLoad;
    use dynamo_runtime::distributed_test_utils::create_test_drt_async;
    use dynamo_runtime::transports::event_plane::EventSubscriber;

    #[tokio::test]
    #[ignore] // Mark as ignored as requested, because CI's integrations still don't have NATS
    async fn test_metrics_publishing_behavior() -> Result<()> {
        // Set up runtime and namespace
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns2001".to_string())?;

        // Create a subscriber for the metrics events
        let mut subscriber = EventSubscriber::for_namespace(&namespace, KV_METRICS_SUBJECT)
            .await
            .unwrap()
            .typed::<ActiveLoad>();

        // Create WorkerMetricsPublisher
        let publisher = WorkerMetricsPublisher::new().unwrap();
        let worker_id = 1234;

        // Start NATS metrics publishing
        publisher.start_nats_metrics_publishing(namespace.clone(), worker_id);

        // Allow some time for the background task to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Test 1: Publish 10 different metrics with 0.5ms intervals
        // Only the last one should be published after 1ms of stability
        for i in 0..10 {
            let value = (i * 100) as u64;
            publisher.publish(None, None, Some(value)).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait a bit more than 1ms to ensure the last metric is published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify we receive exactly one event with the last metric values
        let result =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), subscriber.next())
                .await
                .unwrap();

        let (_envelope, event) = result.unwrap().unwrap(); // Unwrap the Option and the Result
        assert_eq!(event.worker_id, worker_id);
        assert_eq!(event.active_decode_blocks, None); // Worker publisher sends kv_used_blocks
        assert_eq!(event.active_prefill_tokens, None); // Worker doesn't publish prefill tokens
        assert_eq!(event.kv_used_blocks, Some(900));

        // Ensure no more events are waiting
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(no_msg.is_err(), "Expected no more messages, but found one");

        // Test 2: Publish 10 more metrics with same active_decode_blocks - should not trigger publish
        for _ in 0..10 {
            publisher.publish(None, None, Some(900)).unwrap(); // Keep same as last published
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait to ensure no events are published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify no events are received
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(
            no_msg.is_err(),
            "Expected no messages when load metrics don't change"
        );

        drt.shutdown();

        Ok(())
    }
}
