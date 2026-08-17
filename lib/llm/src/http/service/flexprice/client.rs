// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async client that emits LLM usage events to FlexPrice in the background.
//!
//! `enqueue` is non-blocking — the caller returns immediately and a background
//! worker task drains the queue independently, so billing never adds latency
//! to the request path. The queue is bounded; under sustained overload it
//! drops the oldest-pending event rather than applying backpressure to
//! request handlers (mirrors the Python `asyncio.Queue` + `put_nowait`
//! behavior it replaces).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::Serialize;
use tokio::sync::mpsc;

const EVENTS_PATH: &str = "/events";
const QUEUE_SIZE: usize = 1000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Serialize)]
struct UsageEvent {
    event_name: String,
    external_customer_id: String,
    properties: BTreeMap<String, String>,
    source: String,
    event_id: String,
    timestamp: String,
}

pub struct FlexPriceClient {
    tx: mpsc::Sender<UsageEvent>,
}

impl FlexPriceClient {
    /// Build the client and spawn its background drain worker.
    pub fn new(api_host: &str, api_key: &str) -> Arc<Self> {
        let events_url = format!("https://{api_host}{EVENTS_PATH}");
        let api_key = api_key.to_string();
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build FlexPrice HTTP client");

        let (tx, rx) = mpsc::channel::<UsageEvent>(QUEUE_SIZE);
        tokio::spawn(Self::worker(client, events_url, api_key, rx));

        Arc::new(Self { tx })
    }

    /// Non-blocking enqueue. Drops (with a warning) when the queue is full or
    /// the worker task is gone.
    pub fn enqueue(
        &self,
        event_name: String,
        external_customer_id: String,
        properties: BTreeMap<String, String>,
        source: String,
    ) {
        let event = UsageEvent {
            event_name,
            external_customer_id: external_customer_id.clone(),
            properties,
            source,
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        };

        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(
                    customer = %external_customer_id,
                    "FlexPrice event queue full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("FlexPrice event worker is not running; dropping event");
            }
        }
    }

    async fn worker(
        client: Client,
        events_url: String,
        api_key: String,
        mut rx: mpsc::Receiver<UsageEvent>,
    ) {
        while let Some(event) = rx.recv().await {
            let event_name = event.event_name.clone();
            let result = client
                .post(&events_url)
                .header("x-api-key", &api_key)
                .json(&event)
                .send()
                .await;

            match result {
                Ok(resp) if !resp.status().is_success() => {
                    tracing::warn!(
                        status = %resp.status(),
                        event_name = %event_name,
                        "FlexPrice API returned a non-success status"
                    );
                }
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!(error = %err, event_name = %event_name, "Failed to emit FlexPrice event");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_does_not_panic_without_a_reachable_endpoint() {
        // No real network call should ever block or panic the caller — the
        // worker will simply log a warning when the POST fails.
        let client = FlexPriceClient::new("localhost:1", "test-key");
        let mut properties = BTreeMap::new();
        properties.insert("model_id".to_string(), "test-model".to_string());
        client.enqueue(
            "test-event".to_string(),
            "org-1".to_string(),
            properties,
            "test-model".to_string(),
        );
        // Give the worker a chance to run without asserting on network outcome.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
