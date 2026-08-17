// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII usage-billing guard, in the same idiom as [`super::super::metrics::InflightGuard`]
//! and [`super::super::metrics::HttpQueueGuard`].
//!
//! Construct one per billed request (chat completions, completions,
//! embeddings). It is a cheap no-op shell when billing is disabled or the
//! caller's org id is unknown. Record usage as it becomes known — once for a
//! buffered JSON response, or per-chunk for a streaming response — then let
//! it drop. `Drop` enqueues the billing event exactly once, and only if usage
//! was ever recorded, so cancelled/errored requests emit nothing.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use dynamo_protocols::types::{CompletionUsage, EmbeddingUsage};

use super::client::FlexPriceClient;
use super::config::FlexPriceConfig;

pub struct UsageBillingGuard {
    client: Option<Arc<FlexPriceClient>>,
    org_uuid: String,
    event_name: String,
    source: String,
    model: String,
    streaming: bool,
    start: Instant,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    usage_recorded: bool,
}

impl UsageBillingGuard {
    /// Always safe to construct — becomes a no-op shell (`Drop` sends
    /// nothing) whenever `client` is `None` or `org_uuid` is `None`/empty.
    pub fn new(
        client: Option<Arc<FlexPriceClient>>,
        config: &FlexPriceConfig,
        org_uuid: Option<&str>,
        model: &str,
        streaming: bool,
    ) -> Self {
        let org_uuid = org_uuid.unwrap_or_default().to_string();
        let client = if org_uuid.is_empty() { None } else { client };
        Self {
            event_name: config.resolve_event_name(model),
            source: config.resolve_source_name(model),
            client,
            org_uuid,
            model: model.to_string(),
            streaming,
            start: Instant::now(),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            usage_recorded: false,
        }
    }

    /// Record usage from a chat/completions-shaped response. Safe to call
    /// more than once (e.g. per streamed chunk); fields accumulate.
    pub fn record_usage(&mut self, usage: &CompletionUsage) {
        self.input_tokens += usage.prompt_tokens as u64;
        self.output_tokens += usage.completion_tokens as u64;
        self.total_tokens += usage.total_tokens as u64;
        self.usage_recorded = true;
    }

    /// Convenience for the common `Option<&CompletionUsage>` call site.
    pub fn record_usage_opt(&mut self, usage: Option<&CompletionUsage>) {
        if let Some(usage) = usage {
            self.record_usage(usage);
        }
    }

    /// Record usage from an embeddings response (no completion tokens).
    pub fn record_embedding_usage(&mut self, usage: &EmbeddingUsage) {
        self.input_tokens += usage.prompt_tokens as u64;
        self.total_tokens += usage.total_tokens as u64;
        self.usage_recorded = true;
    }
}

impl Drop for UsageBillingGuard {
    fn drop(&mut self) {
        let Some(client) = self.client.take() else {
            return;
        };
        if !self.usage_recorded {
            return;
        }

        let mut properties = BTreeMap::new();
        properties.insert("model_id".to_string(), self.model.clone());
        properties.insert("input_tokens".to_string(), self.input_tokens.to_string());
        properties.insert("output_tokens".to_string(), self.output_tokens.to_string());
        properties.insert("total_tokens".to_string(), self.total_tokens.to_string());
        properties.insert(
            "time_taken".to_string(),
            format!("{:.4}", self.start.elapsed().as_secs_f64()),
        );
        properties.insert("streaming".to_string(), self.streaming.to_string());
        properties.insert("status".to_string(), "success".to_string());

        client.enqueue(
            self.event_name.clone(),
            self.org_uuid.clone(),
            properties,
            self.source.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, completion: u32, total: u32) -> CompletionUsage {
        CompletionUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    #[test]
    fn no_op_when_client_is_none() {
        let config = FlexPriceConfig::default();
        let mut guard = UsageBillingGuard::new(None, &config, Some("org-1"), "model", false);
        guard.record_usage(&usage(10, 5, 15));
        // Dropping must not panic even though there's no client to send to.
        drop(guard);
    }

    #[tokio::test]
    async fn no_op_when_org_uuid_missing() {
        let client = FlexPriceClient::new("localhost:1", "key");
        let config = FlexPriceConfig::default();
        let guard = UsageBillingGuard::new(Some(client), &config, None, "model", false);
        // usage never recorded and org id absent — Drop must no-op safely.
        drop(guard);
    }

    #[test]
    fn accumulates_usage_across_multiple_records() {
        let config = FlexPriceConfig::default();
        let mut guard = UsageBillingGuard::new(None, &config, Some("org-1"), "model", true);
        guard.record_usage(&usage(10, 5, 15));
        guard.record_usage(&usage(0, 3, 3));
        assert_eq!(guard.input_tokens, 10);
        assert_eq!(guard.output_tokens, 8);
        assert_eq!(guard.total_tokens, 18);
    }

    #[tokio::test]
    async fn drop_without_recorded_usage_is_a_no_op() {
        let client = FlexPriceClient::new("localhost:1", "key");
        let config = FlexPriceConfig::default();
        let guard = UsageBillingGuard::new(Some(client), &config, Some("org-1"), "model", false);
        // Cancelled/errored request: never called record_usage — must not panic.
        drop(guard);
    }
}
