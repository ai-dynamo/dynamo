// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport trait and implementations.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;

#[async_trait]
pub trait RegistryTransport: Send + Sync {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>>;
    async fn publish(&self, data: &[u8]) -> Result<()>;
    fn name(&self) -> &'static str;
}

/// In-process transport for testing.
pub struct InProcessTransport {
    handler: Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>,
    publish_tx: mpsc::UnboundedSender<Vec<u8>>,
}

impl InProcessTransport {
    pub fn new<F>(handler: F) -> (Self, mpsc::UnboundedReceiver<Vec<u8>>)
    where
        F: Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                handler: Arc::new(handler),
                publish_tx: tx,
            },
            rx,
        )
    }

    pub fn pair() -> (Self, InProcessHub) {
        let hub = InProcessHub::new();
        let hub_clone = hub.clone();
        let (transport, rx) = Self::new(move |data| hub_clone.handle(data));

        let hub_for_publish = hub.clone();
        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(data) = rx.recv().await {
                hub_for_publish.handle(&data);
            }
        });

        (transport, hub)
    }
}

#[async_trait]
impl RegistryTransport for InProcessTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok((self.handler)(data))
    }

    async fn publish(&self, data: &[u8]) -> Result<()> {
        self.publish_tx.send(data.to_vec())?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "in_process"
    }
}

/// In-process hub for testing.
#[derive(Clone)]
pub struct InProcessHub {
    request_handler: Arc<Mutex<Option<Box<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>>>>,
}

impl InProcessHub {
    pub fn new() -> Self {
        Self {
            request_handler: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_handler<F>(&self, handler: F)
    where
        F: Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
    {
        *self.request_handler.lock() = Some(Box::new(handler));
    }

    pub fn handle(&self, data: &[u8]) -> Vec<u8> {
        if let Some(handler) = self.request_handler.lock().as_ref() {
            handler(data)
        } else {
            Vec::new()
        }
    }
}

impl Default for InProcessHub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_process_transport() {
        let (transport, _rx) = InProcessTransport::new(|data| {
            let mut response = data.to_vec();
            response.reverse();
            response
        });

        let result = transport.request(b"hello").await.unwrap();
        assert_eq!(result, b"olleh");
    }

    #[tokio::test]
    async fn test_in_process_publish() {
        let (transport, mut rx) = InProcessTransport::new(|_| Vec::new());

        transport.publish(b"test message").await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received, b"test message");
    }
}
