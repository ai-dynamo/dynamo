// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A WorkerSet represents a group of workers deployed from the same configuration,
//! identified by their shared namespace. Each WorkerSet owns a complete pipeline
//! (engines, KV router, prefill router) built from its specific ModelDeploymentCard.
//!
//! During rolling updates, multiple WorkerSets coexist under the same Model, each
//! serving traffic proportional to its worker count.

use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use crate::{
    discovery::KvWorkerMonitor,
    kv_router::{KvRouter, PrefillRouter},
    model_card::ModelDeploymentCard,
    types::{
        generic::tensor::TensorStreamingEngine,
        openai::{
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine,
            images::OpenAIImagesStreamingEngine,
            videos::OpenAIVideosStreamingEngine,
        },
    },
};

/// A set of workers from the same namespace/configuration with their own pipeline.
pub struct WorkerSet {
    /// Full namespace (e.g., "default-myapp-abc12345")
    namespace: String,

    /// MDC checksum for this set's configuration
    mdcsum: String,

    /// The model deployment card used to build this set's pipeline
    card: ModelDeploymentCard,

    // Engines — each WorkerSet owns its own pipelines
    pub(crate) chat_engine: Option<OpenAIChatCompletionsStreamingEngine>,
    pub(crate) completions_engine: Option<OpenAICompletionsStreamingEngine>,
    pub(crate) embeddings_engine: Option<OpenAIEmbeddingsStreamingEngine>,
    pub(crate) images_engine: Option<OpenAIImagesStreamingEngine>,
    pub(crate) videos_engine: Option<OpenAIVideosStreamingEngine>,
    pub(crate) tensor_engine: Option<TensorStreamingEngine>,

    /// KV router for this set's workers (if KV mode)
    pub(crate) kv_router: Option<Arc<KvRouter>>,

    /// Prefill router for this set's prefill workers (if disaggregated)
    pub(crate) prefill_router: Option<Arc<PrefillRouter>>,

    /// Worker monitor for load-based rejection
    pub(crate) worker_monitor: Option<KvWorkerMonitor>,

    /// Number of active workers in this set (for weighted selection)
    worker_count: AtomicUsize,
}

impl WorkerSet {
    pub fn new(
        namespace: String,
        mdcsum: String,
        card: ModelDeploymentCard,
    ) -> Self {
        Self {
            namespace,
            mdcsum,
            card,
            chat_engine: None,
            completions_engine: None,
            embeddings_engine: None,
            images_engine: None,
            videos_engine: None,
            tensor_engine: None,
            kv_router: None,
            prefill_router: None,
            worker_monitor: None,
            worker_count: AtomicUsize::new(0),
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    pub fn card(&self) -> &ModelDeploymentCard {
        &self.card
    }

    pub fn has_chat_engine(&self) -> bool {
        self.chat_engine.is_some()
    }

    pub fn has_completions_engine(&self) -> bool {
        self.completions_engine.is_some()
    }

    pub fn has_embeddings_engine(&self) -> bool {
        self.embeddings_engine.is_some()
    }

    pub fn has_images_engine(&self) -> bool {
        self.images_engine.is_some()
    }

    pub fn has_videos_engine(&self) -> bool {
        self.videos_engine.is_some()
    }

    pub fn has_tensor_engine(&self) -> bool {
        self.tensor_engine.is_some()
    }

    /// Whether this set has any decode engine (chat or completions)
    pub fn has_decode_engine(&self) -> bool {
        self.has_chat_engine() || self.has_completions_engine()
    }

    /// Whether this set tracks a prefill model (no engine, just lifecycle)
    pub fn is_prefill_set(&self) -> bool {
        !self.has_decode_engine()
            && !self.has_embeddings_engine()
            && !self.has_images_engine()
            && !self.has_videos_engine()
            && !self.has_tensor_engine()
    }

    /// Build ParsingOptions from this WorkerSet's card configuration.
    pub fn parsing_options(&self) -> crate::protocols::openai::ParsingOptions {
        crate::protocols::openai::ParsingOptions::new(
            self.card.runtime_config.tool_call_parser.clone(),
            self.card.runtime_config.reasoning_parser.clone(),
        )
    }

    /// Number of active workers in this set.
    pub fn worker_count(&self) -> usize {
        self.worker_count.load(Ordering::Relaxed)
    }

    /// Increment worker count (worker joined this set).
    pub fn increment_workers(&self) -> usize {
        self.worker_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement worker count (worker left this set). Returns new count.
    pub fn decrement_workers(&self) -> usize {
        loop {
            let current = self.worker_count.load(Ordering::Relaxed);
            if current == 0 {
                return 0;
            }
            match self.worker_count.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current - 1,
                Err(_) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;

    fn make_worker_set(namespace: &str, mdcsum: &str) -> WorkerSet {
        WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        )
    }

    #[test]
    fn test_worker_set_basics() {
        let ws = make_worker_set("ns1", "abc123");
        assert_eq!(ws.namespace(), "ns1");
        assert_eq!(ws.mdcsum(), "abc123");
    }

    #[test]
    fn test_no_engines_by_default() {
        let ws = make_worker_set("ns1", "abc123");
        assert!(!ws.has_chat_engine());
        assert!(!ws.has_completions_engine());
        assert!(!ws.has_embeddings_engine());
        assert!(!ws.has_images_engine());
        assert!(!ws.has_tensor_engine());
        assert!(!ws.has_decode_engine());
        assert!(ws.is_prefill_set());
    }

    #[test]
    fn test_worker_count_increment() {
        let ws = make_worker_set("ns1", "abc123");
        assert_eq!(ws.worker_count(), 0);
        assert_eq!(ws.increment_workers(), 1);
        assert_eq!(ws.increment_workers(), 2);
        assert_eq!(ws.increment_workers(), 3);
        assert_eq!(ws.worker_count(), 3);
    }

    #[test]
    fn test_worker_count_decrement() {
        let ws = make_worker_set("ns1", "abc123");
        ws.increment_workers();
        ws.increment_workers();
        ws.increment_workers();
        assert_eq!(ws.decrement_workers(), 2);
        assert_eq!(ws.decrement_workers(), 1);
        assert_eq!(ws.decrement_workers(), 0);
        assert_eq!(ws.worker_count(), 0);
    }

    #[test]
    fn test_worker_count_decrement_at_zero() {
        let ws = make_worker_set("ns1", "abc123");
        assert_eq!(ws.decrement_workers(), 0);
        assert_eq!(ws.decrement_workers(), 0);
        assert_eq!(ws.worker_count(), 0);
    }
}
