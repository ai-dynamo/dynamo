// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A WorkerSet represents a group of workers deployed from the same configuration,
//! identified by their shared namespace. Each WorkerSet owns a complete pipeline
//! (engines, KV router, prefill router) built from its specific ModelDeploymentCard.

use std::sync::Arc;

use tokio::sync::watch;

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

    /// Watcher for available instance IDs (from the Client's discovery watch).
    /// None for in-process models (http/grpc) which don't have a discovery client.
    instance_count_rx: Option<watch::Receiver<Vec<u64>>>,
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
            instance_count_rx: None,
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

    /// Number of active workers in this set, derived from the Client's discovery watcher.
    /// Returns 1 for in-process models (no watcher) since they always have one local worker.
    pub fn worker_count(&self) -> usize {
        match &self.instance_count_rx {
            Some(rx) => rx.borrow().len(),
            None => 1,
        }
    }

    /// Store the instance watcher from the Client's discovery system.
    /// Must be called before the WorkerSet is wrapped in Arc.
    pub fn set_instance_watcher(&mut self, rx: watch::Receiver<Vec<u64>>) {
        self.instance_count_rx = Some(rx);
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

}
