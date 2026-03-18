// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

pub mod sglang;
pub mod vllm;

use crate::common::protocols::DirectRequest;
use tokio::sync::mpsc;

pub use sglang::SglangScheduler;
pub use vllm::{MockerMetrics, Scheduler};

/// Engine-agnostic scheduler interface.
///
/// Both vLLM and SGLang schedulers implement this trait so that the engine
/// wrapper (`MockEngine`) can work with either backend through the same API.
pub trait SchedulerHandle: Send + Sync {
    /// Send a request to the scheduler's waiting queue.
    fn receive(&self, request: DirectRequest);

    /// Get a clone of the request sender channel for direct sending.
    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest>;

    /// Get a watch receiver for scheduler metrics (active decode blocks, etc.).
    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics>;
}

/// Shared test utilities for scheduler stress tests.
#[cfg(test)]
pub(crate) mod test_utils {
    use super::*;
    use crate::common::protocols::OutputSignal;
    use tokio::time::Duration;

    /// Send `num_requests` to a scheduler, collect all output signals, and assert
    /// that the scheduler produces exactly `num_requests * max_output_tokens` signals
    /// and returns to idle (0 active decode blocks).
    ///
    /// When `use_shared_tokens` is true, the first half of each request shares a
    /// common prefix to exercise prefix caching / radix tree reuse.
    pub async fn assert_scheduler_completes_all(
        scheduler: &dyn SchedulerHandle,
        output_rx: &mut mpsc::UnboundedReceiver<OutputSignal>,
        num_requests: usize,
        input_len: usize,
        max_output_tokens: usize,
        use_shared_tokens: bool,
    ) {
        let shared_tokens = if use_shared_tokens {
            Some(
                (0..input_len / 2)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        for _ in 0..num_requests {
            let input_tokens = if let Some(ref shared) = shared_tokens {
                let mut tokens = shared.clone();
                tokens.extend((0..input_len / 2).map(|_| rand::random::<u32>() % 50000));
                tokens
            } else {
                (0..input_len)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>()
            };

            scheduler.receive(DirectRequest {
                tokens: input_tokens,
                max_output_tokens,
                uuid: None,
                dp_rank: 0,
            });
        }

        let expected_tokens = num_requests * max_output_tokens;
        let mut received_tokens = 0;

        let timeout = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                biased;
                Some(_) = output_rx.recv() => {
                    received_tokens += 1;
                    if received_tokens >= expected_tokens {
                        break;
                    }
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }

        assert_eq!(
            received_tokens, expected_tokens,
            "Expected {expected_tokens} output signals, got {received_tokens}"
        );

        // Verify scheduler returns to idle
        tokio::time::sleep(Duration::from_millis(100)).await;
        let metrics = scheduler.metrics_receiver().borrow().clone();
        assert_eq!(
            metrics.active_decode_blocks, 0,
            "Scheduler should be idle after all requests complete, got {} active blocks",
            metrics.active_decode_blocks
        );
    }
}
