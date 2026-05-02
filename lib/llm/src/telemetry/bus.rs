// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use tokio::sync::broadcast;

/// Typed, in-process fanout bus for low-overhead telemetry paths.
pub struct TelemetryBus<T> {
    sender: OnceLock<broadcast::Sender<T>>,
}

impl<T> TelemetryBus<T>
where
    T: Clone + Send + 'static,
{
    pub const fn new() -> Self {
        Self {
            sender: OnceLock::new(),
        }
    }

    pub fn init(&self, capacity: usize) {
        let (tx, _rx) = broadcast::channel::<T>(capacity.max(1));
        if self.sender.set(tx).is_err() {
            tracing::debug!(
                capacity,
                "telemetry bus already initialized; keeping existing sender"
            );
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<T> {
        self.sender
            .get()
            .expect("telemetry bus not initialized")
            .subscribe()
    }

    pub fn publish(&self, record: T) {
        if let Some(tx) = self.sender.get() {
            let _ = tx.send(record);
        }
    }
}

impl<T> Default for TelemetryBus<T>
where
    T: Clone + Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::TelemetryBus;

    #[tokio::test]
    async fn publishes_to_subscribers() {
        static BUS: TelemetryBus<u64> = TelemetryBus::new();
        BUS.init(4);
        let mut rx = BUS.subscribe();

        BUS.publish(7);

        assert_eq!(rx.recv().await.unwrap(), 7);
    }

    #[tokio::test]
    async fn publish_before_init_is_dropped() {
        static BUS: TelemetryBus<u64> = TelemetryBus::new();

        BUS.publish(7);
        BUS.init(4);
        let mut rx = BUS.subscribe();
        BUS.publish(8);

        assert_eq!(rx.recv().await.unwrap(), 8);
    }
}
