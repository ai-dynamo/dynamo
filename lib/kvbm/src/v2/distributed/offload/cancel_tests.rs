// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive cancellation tests for the offload pipeline.
//!
//! These tests verify the cancellation invariants documented in README.md:
//! - P1: Container is the unit of cancellation
//! - P2: Token travels with container
//! - P3: Upgrade is the commitment boundary
//! - P4: Sweep before upgrade
//!
//! Key invariant: Cancellation is only confirmed when:
//! 1. All source block lists are removed from queues
//! 2. All in-flight transfers have completed

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use tokio::sync::Barrier;

    use crate::v2::distributed::offload::cancel::{CancelState, CancellationToken};
    use crate::v2::distributed::offload::handle::TransferId;
    use crate::v2::distributed::offload::queue::CancellableQueue;

    // =========================================================================
    // Draining Invariant Tests
    // =========================================================================

    /// Test that confirmation does NOT resolve while in-flight transfers remain.
    #[tokio::test]
    async fn test_confirmation_waits_for_in_flight_to_drain() {
        let (token, updater) = CancellationToken::new();

        // Request cancellation
        token.request();
        assert!(token.is_requested());

        // Set draining with 3 in-flight
        updater.set_draining(3);
        assert_eq!(token.state(), CancelState::Draining { in_flight: 3 });

        // Confirmation should NOT resolve yet
        let confirmation = token.wait_confirmed();
        let result = tokio::time::timeout(Duration::from_millis(50), confirmation.wait()).await;
        assert!(
            result.is_err(),
            "Confirmation should timeout while in-flight > 0"
        );

        // Drain to 1
        updater.update_draining(1);
        assert_eq!(token.state(), CancelState::Draining { in_flight: 1 });

        // Still should not resolve
        let confirmation = token.wait_confirmed();
        let result = tokio::time::timeout(Duration::from_millis(50), confirmation.wait()).await;
        assert!(
            result.is_err(),
            "Confirmation should timeout while in-flight > 0"
        );

        // Drain to 0 - this should trigger confirmation
        updater.update_draining(0);
        assert_eq!(token.state(), CancelState::Confirmed);

        // Now confirmation should resolve immediately
        let confirmation = token.wait_confirmed();
        tokio::time::timeout(Duration::from_millis(50), confirmation.wait())
            .await
            .expect("Confirmation should resolve when in-flight = 0");
    }

    /// Test that draining countdown correctly transitions to confirmed.
    #[tokio::test]
    async fn test_draining_countdown_to_confirmation() {
        let (token, updater) = CancellationToken::new();
        let in_flight = Arc::new(AtomicUsize::new(5));
        let in_flight_clone = in_flight.clone();

        token.request();
        updater.set_draining(in_flight.load(Ordering::SeqCst));

        // Spawn task to simulate transfers completing
        // We need to move updater into the spawned task
        tokio::spawn(async move {
            for _ in 0..5 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let remaining = in_flight_clone.fetch_sub(1, Ordering::SeqCst) - 1;
                updater.update_draining(remaining);
            }
        });

        // Wait for confirmation
        let confirmation = token.wait_confirmed();
        tokio::time::timeout(Duration::from_millis(200), confirmation.wait())
            .await
            .expect("Should confirm after all in-flight complete");

        assert!(token.is_confirmed());
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    /// Test concurrent cancellation requests are idempotent.
    #[tokio::test]
    async fn test_concurrent_cancel_requests() {
        let (token, updater) = CancellationToken::new();
        let barrier = Arc::new(Barrier::new(3));

        // Spawn multiple tasks requesting cancellation
        let token1 = token.clone();
        let barrier1 = barrier.clone();
        let t1 = tokio::spawn(async move {
            barrier1.wait().await;
            token1.request();
        });

        let token2 = token.clone();
        let barrier2 = barrier.clone();
        let t2 = tokio::spawn(async move {
            barrier2.wait().await;
            token2.request();
        });

        barrier.wait().await;
        token.request();

        t1.await.unwrap();
        t2.await.unwrap();

        // Should still be requested (idempotent)
        assert!(token.is_requested());

        // Confirm and verify
        updater.set_confirmed();
        assert!(token.is_confirmed());
    }

    // =========================================================================
    // Token-Based Cancellation Tests
    // =========================================================================

    /// Container that carries its own CancellationToken.
    struct MockContainer {
        id: usize,
        cancel_token: CancellationToken,
    }

    impl MockContainer {
        fn new(id: usize, token: CancellationToken) -> Self {
            Self {
                id,
                cancel_token: token,
            }
        }

        fn is_cancelled(&self) -> bool {
            self.cancel_token.is_requested()
        }
    }

    /// Test that container carries its own token and can check cancellation.
    #[test]
    fn test_container_carries_token() {
        let (token, _updater) = CancellationToken::new();
        let container = MockContainer::new(1, token.clone());

        assert!(!container.is_cancelled());

        // Cancel via the original token
        token.request();

        // Container should see cancellation via its cloned token
        assert!(container.is_cancelled());
    }

    /// Test multiple containers sharing same token (from same TransferHandle).
    #[test]
    fn test_multiple_containers_same_token() {
        let (token, _updater) = CancellationToken::new();

        let c1 = MockContainer::new(1, token.clone());
        let c2 = MockContainer::new(2, token.clone());
        let c3 = MockContainer::new(3, token.clone());

        assert!(!c1.is_cancelled());
        assert!(!c2.is_cancelled());
        assert!(!c3.is_cancelled());

        // Cancel via handle's token
        token.request();

        // All containers should see cancellation
        assert!(c1.is_cancelled());
        assert!(c2.is_cancelled());
        assert!(c3.is_cancelled());
    }

    /// Test containers from different handles have independent cancellation.
    #[test]
    fn test_independent_container_cancellation() {
        let (token1, _updater1) = CancellationToken::new();
        let (token2, _updater2) = CancellationToken::new();

        let c1 = MockContainer::new(1, token1.clone());
        let c2 = MockContainer::new(2, token2.clone());

        // Cancel only token1
        token1.request();

        assert!(c1.is_cancelled());
        assert!(!c2.is_cancelled());
    }

    // =========================================================================
    // Queue + Token Integration Tests
    // =========================================================================

    /// Wrapper that includes a CancellationToken for queue testing.
    struct TokenWrapper {
        data: i32,
        cancel_token: CancellationToken,
    }

    /// Test queue sweep using token-based cancellation check.
    #[test]
    fn test_queue_sweep_with_token_check() {
        let queue: CancellableQueue<TokenWrapper> = CancellableQueue::new();

        let (token1, _) = CancellationToken::new();
        let (token2, _) = CancellationToken::new();

        let id1 = TransferId::new();
        let id2 = TransferId::new();

        // Push items with different tokens
        queue.push(
            id1,
            TokenWrapper {
                data: 1,
                cancel_token: token1.clone(),
            },
        );
        queue.push(
            id2,
            TokenWrapper {
                data: 2,
                cancel_token: token2.clone(),
            },
        );
        queue.push(
            id1,
            TokenWrapper {
                data: 3,
                cancel_token: token1.clone(),
            },
        );

        assert_eq!(queue.len_approx(), 3);

        // Cancel token1 (and mark in queue for sweep)
        token1.request();
        queue.mark_cancelled(id1);

        // Sweep should remove token1's items
        let removed = queue.sweep();
        assert_eq!(removed, 2);
        assert_eq!(queue.len_approx(), 1);

        // Remaining item should be from token2
        let item = queue.pop().unwrap();
        assert_eq!(item.data.data, 2);
        assert!(!item.data.cancel_token.is_requested());
    }

    // =========================================================================
    // Batch Partial Cancellation Tests
    // =========================================================================

    /// Mock batch of containers for testing partial cancellation.
    struct MockBatch {
        containers: Vec<MockContainer>,
    }

    impl MockBatch {
        fn new(containers: Vec<MockContainer>) -> Self {
            Self { containers }
        }

        /// Remove cancelled containers, return count removed.
        fn sweep_cancelled(&mut self) -> usize {
            let before = self.containers.len();
            self.containers.retain(|c| !c.is_cancelled());
            before - self.containers.len()
        }

        fn len(&self) -> usize {
            self.containers.len()
        }

        fn is_empty(&self) -> bool {
            self.containers.is_empty()
        }
    }

    /// Test partial batch cancellation - some containers cancelled, others proceed.
    #[test]
    fn test_batch_partial_cancellation() {
        let (token1, _updater1) = CancellationToken::new();
        let (token2, _updater2) = CancellationToken::new();
        let (token3, _updater3) = CancellationToken::new();

        // Create container with cloned token
        let c1 = MockContainer::new(1, token1.clone());
        let c2 = MockContainer::new(2, token2.clone());
        let c3 = MockContainer::new(3, token3.clone());
        let c4 = MockContainer::new(4, token1.clone()); // Same token as c1

        // Verify tokens work before batching
        assert!(!c1.is_cancelled());
        assert!(!c4.is_cancelled());

        let mut batch = MockBatch::new(vec![c1, c2, c3, c4]);
        assert_eq!(batch.len(), 4);

        // Cancel token1 (affects containers 1 and 4)
        token1.request();
        assert!(token1.is_requested());

        // Verify containers in batch see the cancellation
        assert!(
            batch.containers[0].is_cancelled(),
            "Container 1 should be cancelled"
        );
        assert!(
            !batch.containers[1].is_cancelled(),
            "Container 2 should NOT be cancelled"
        );
        assert!(
            !batch.containers[2].is_cancelled(),
            "Container 3 should NOT be cancelled"
        );
        assert!(
            batch.containers[3].is_cancelled(),
            "Container 4 should be cancelled"
        );

        let removed = batch.sweep_cancelled();
        assert_eq!(removed, 2);
        assert_eq!(batch.len(), 2);

        // Remaining containers should be 2 and 3
        assert_eq!(batch.containers[0].id, 2);
        assert_eq!(batch.containers[1].id, 3);
    }

    /// Test batch where all containers are cancelled.
    #[test]
    fn test_batch_full_cancellation() {
        let (token, _updater) = CancellationToken::new();

        // Create containers with cloned tokens
        let c1 = MockContainer::new(1, token.clone());
        let c2 = MockContainer::new(2, token.clone());
        let c3 = MockContainer::new(3, token.clone());

        // Verify token clone works
        assert!(!c1.is_cancelled());
        assert!(!c2.is_cancelled());
        assert!(!c3.is_cancelled());

        let mut batch = MockBatch::new(vec![c1, c2, c3]);

        token.request();
        assert!(token.is_requested());

        // Verify containers see cancellation
        assert!(
            batch.containers[0].is_cancelled(),
            "Container 1 should be cancelled"
        );
        assert!(
            batch.containers[1].is_cancelled(),
            "Container 2 should be cancelled"
        );
        assert!(
            batch.containers[2].is_cancelled(),
            "Container 3 should be cancelled"
        );

        let removed = batch.sweep_cancelled();
        assert_eq!(removed, 3);
        assert!(batch.is_empty());
    }

    /// Test batch where no containers are cancelled.
    #[test]
    fn test_batch_no_cancellation() {
        let (token1, _updater1) = CancellationToken::new();
        let (token2, _updater2) = CancellationToken::new();

        let mut batch = MockBatch::new(vec![
            MockContainer::new(1, token1.clone()),
            MockContainer::new(2, token2.clone()),
        ]);

        // Don't cancel anything
        let removed = batch.sweep_cancelled();
        assert_eq!(removed, 0);
        assert_eq!(batch.len(), 2);
    }

    // =========================================================================
    // Select-Based Cancellation Tests
    // =========================================================================

    /// Simulate precondition awaiter with select on event OR cancel.
    #[tokio::test]
    async fn test_select_cancellation_during_wait() {
        let (token, _updater) = CancellationToken::new();
        let (event_tx, event_rx) = tokio::sync::oneshot::channel::<()>();

        // Verify initial state
        assert!(!token.is_requested());

        let token_clone = token.clone();
        let result = tokio::spawn(async move {
            // Poll-based cancellation check with timeout
            let cancel_check = async {
                loop {
                    if token_clone.is_requested() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };

            tokio::select! {
                biased;  // Prefer first branch to complete

                _ = event_rx => {
                    "event"
                }
                _ = cancel_check => {
                    "cancelled"
                }
            }
        });

        // Give the task time to start and enter select
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Cancel (don't send event)
        token.request();
        assert!(token.is_requested());

        let outcome = tokio::time::timeout(Duration::from_millis(200), result)
            .await
            .expect("Should complete within timeout")
            .expect("Task should not panic");

        assert_eq!(outcome, "cancelled");

        // Event sender still exists - wasn't used
        drop(event_tx);
    }

    /// Test that event completes before cancellation.
    #[tokio::test]
    async fn test_select_event_before_cancel() {
        let (token, _) = CancellationToken::new();
        let (event_tx, event_rx) = tokio::sync::oneshot::channel::<()>();

        let token_clone = token.clone();
        let result = tokio::spawn(async move {
            tokio::select! {
                _ = event_rx => {
                    "event"
                }
                _ = async {
                    loop {
                        if token_clone.is_requested() {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                } => {
                    "cancelled"
                }
            }
        });

        // Give the task time to start
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Send event (before cancellation)
        event_tx.send(()).unwrap();

        let outcome = tokio::time::timeout(Duration::from_millis(100), result)
            .await
            .expect("Should complete")
            .expect("Should not panic");

        assert_eq!(outcome, "event");
        assert!(!token.is_requested());
    }

    // =========================================================================
    // End-to-End Cancellation Flow Tests
    // =========================================================================

    /// Test complete cancellation flow: request → sweep → drain → confirm.
    #[tokio::test]
    async fn test_end_to_end_cancellation_flow() {
        let (token, updater) = CancellationToken::new();
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        // Simulate: 3 items in queue, 2 in-flight
        queue.push(id, 1);
        queue.push(id, 2);
        queue.push(id, 3);
        let in_flight = Arc::new(AtomicUsize::new(2));

        // Request cancellation
        token.request();
        assert!(token.is_requested());

        // Mark cancelled in queue
        queue.mark_cancelled(id);

        // Sweep queue
        let removed = queue.sweep();
        assert_eq!(removed, 3);
        assert_eq!(queue.len_approx(), 0);

        // Set draining for in-flight
        updater.set_draining(in_flight.load(Ordering::SeqCst));
        assert!(token.state().is_draining());

        // Simulate in-flight completing
        let in_flight_clone = in_flight.clone();
        tokio::spawn(async move {
            for _ in 0..2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let remaining = in_flight_clone.fetch_sub(1, Ordering::SeqCst) - 1;
                updater.update_draining(remaining);
            }
        });

        // Wait for confirmation
        let confirmation = token.wait_confirmed();
        tokio::time::timeout(Duration::from_millis(100), confirmation.wait())
            .await
            .expect("Should confirm after queue swept and in-flight drained");

        assert!(token.is_confirmed());
        assert_eq!(queue.len_approx(), 0);
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    /// Test cancellation with nothing in-flight (immediate confirmation after sweep).
    #[tokio::test]
    async fn test_cancellation_nothing_in_flight() {
        let (token, updater) = CancellationToken::new();
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        // Items only in queue, nothing in-flight
        queue.push(id, 1);
        queue.push(id, 2);

        // Request and sweep
        token.request();
        queue.mark_cancelled(id);
        let removed = queue.sweep();
        assert_eq!(removed, 2);

        // No in-flight, go directly to confirmed
        updater.update_draining(0); // This sets Confirmed when in_flight = 0

        assert!(token.is_confirmed());

        // Confirmation should resolve immediately
        let confirmation = token.wait_confirmed();
        tokio::time::timeout(Duration::from_millis(10), confirmation.wait())
            .await
            .expect("Should confirm immediately with nothing in-flight");
    }
}
