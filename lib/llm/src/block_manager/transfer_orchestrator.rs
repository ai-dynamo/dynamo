// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Priority class for transfer requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPriority {
    High,
    Low,
}

/// Sender side of a priority queue (high + low).
pub struct PrioritySender<T> {
    high: mpsc::Sender<T>,
    low: mpsc::Sender<T>,
}

impl<T> Clone for PrioritySender<T> {
    fn clone(&self) -> Self {
        Self {
            high: self.high.clone(),
            low: self.low.clone(),
        }
    }
}

impl<T> PrioritySender<T> {
    pub fn try_send(
        &self,
        priority: TransferPriority,
        value: T,
    ) -> Result<(), mpsc::error::TrySendError<T>> {
        match priority {
            TransferPriority::High => self.high.try_send(value),
            TransferPriority::Low => self.low.try_send(value),
        }
    }

    /// Async send that waits for capacity instead of dropping.
    pub async fn send(
        &self,
        priority: TransferPriority,
        value: T,
    ) -> Result<(), mpsc::error::SendError<T>> {
        match priority {
            TransferPriority::High => self.high.send(value).await,
            TransferPriority::Low => self.low.send(value).await,
        }
    }
}

/// Receiver side of a priority queue (high + low).
pub struct PriorityReceiver<T> {
    high: mpsc::Receiver<T>,
    low: mpsc::Receiver<T>,
}

/// Build a priority queue with independent capacities for high and low lanes.
pub fn priority_channel<T>(
    high_capacity: usize,
    low_capacity: usize,
) -> (PrioritySender<T>, PriorityReceiver<T>) {
    let (high_tx, high_rx) = mpsc::channel(high_capacity);
    let (low_tx, low_rx) = mpsc::channel(low_capacity);
    (
        PrioritySender {
            high: high_tx,
            low: low_tx,
        },
        PriorityReceiver {
            high: high_rx,
            low: low_rx,
        },
    )
}

/// Run a bounded-concurrency worker loop over a priority queue.
///
/// - Never blocks waiting for queue capacity.
/// - High-priority lane is preferred over low-priority lane.
/// - Concurrency is capped by `max_inflight`.
pub async fn run_priority_worker<T, F, Fut>(
    cancellation_token: CancellationToken,
    mut receiver: PriorityReceiver<T>,
    max_inflight: usize,
    mut worker: F,
) where
    T: Send + 'static,
    F: FnMut(T) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let mut join_set = JoinSet::new();

    loop {
        if join_set.len() >= max_inflight {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                done = join_set.join_next() => {
                    if let Some(Err(e)) = done {
                        tracing::error!("priority worker join error: {:?}", e);
                    }
                }
            }
            continue;
        }

        tokio::select! {
            _ = cancellation_token.cancelled() => break,
            done = join_set.join_next(), if !join_set.is_empty() => {
                if let Some(Err(e)) = done {
                    tracing::error!("priority worker join error: {:?}", e);
                }
            }
            req = async {
                tokio::select! {
                    biased;
                    hi = receiver.high.recv() => hi,
                    lo = receiver.low.recv() => lo,
                }
            } => {
                match req {
                    Some(req) => {
                        join_set.spawn(worker(req));
                    }
                    None => break,
                }
            }
        }
    }

    while let Some(done) = join_set.join_next().await {
        if let Err(e) = done {
            tracing::error!("priority worker join error: {:?}", e);
        }
    }
}
