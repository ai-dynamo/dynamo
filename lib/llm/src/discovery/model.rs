// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Model represents a named model (e.g., "llama-3-70b") that may be served by
//! one or more WorkerSets. Each WorkerSet corresponds to a namespace.
//!
//! Requests are routed to a WorkerSet selected by weighted random (proportional to worker count).

use std::sync::Arc;

use dashmap::DashMap;
use rand::Rng;

use super::worker_monitor::LoadThresholdConfig;
use super::worker_set::WorkerSet;
use super::{KvWorkerMonitor, ModelManagerError};
use crate::protocols::openai::ParsingOptions;

use crate::types::{
    generic::tensor::TensorStreamingEngine,
    openai::{
        chat_completions::OpenAIChatCompletionsStreamingEngine,
        completions::OpenAICompletionsStreamingEngine,
        embeddings::OpenAIEmbeddingsStreamingEngine,
        images::OpenAIImagesStreamingEngine,
        videos::OpenAIVideosStreamingEngine,
    },
};

/// A named model backed by one or more WorkerSets.
pub struct Model {
    name: String,
    worker_sets: DashMap<String, Arc<WorkerSet>>,
}

impl Model {
    pub fn new(name: String) -> Self {
        Self {
            name,
            worker_sets: DashMap::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_worker_set(&self, namespace: String, worker_set: Arc<WorkerSet>) {
        tracing::info!(
            model = %self.name,
            namespace = %namespace,
            "Adding worker set to model"
        );
        self.worker_sets.insert(namespace, worker_set);
    }

    pub fn remove_worker_set(&self, namespace: &str) -> Option<Arc<WorkerSet>> {
        let removed = self.worker_sets.remove(namespace).map(|(_, ws)| ws);
        if removed.is_some() {
            tracing::info!(
                model = %self.name,
                namespace = %namespace,
                remaining_sets = self.worker_sets.len(),
                "Removed worker set from model"
            );
        }
        removed
    }

    pub fn has_worker_set(&self, namespace: &str) -> bool {
        self.worker_sets.contains_key(namespace)
    }

    pub fn get_worker_set(&self, namespace: &str) -> Option<Arc<WorkerSet>> {
        self.worker_sets.get(namespace).map(|entry| entry.value().clone())
    }

    pub fn is_empty(&self) -> bool {
        self.worker_sets.is_empty()
    }

    pub fn worker_set_count(&self) -> usize {
        self.worker_sets.len()
    }

    /// Check if this model has any decode engine (chat or completions) across any WorkerSet.
    pub fn has_decode_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_decode_engine())
    }

    /// Check if this model tracks prefill (any WorkerSet is a prefill set).
    pub fn has_prefill(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().is_prefill_set())
    }

    /// Check if any WorkerSet has a chat engine.
    pub fn has_chat_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_chat_engine())
    }

    /// Check if any WorkerSet has a completions engine.
    pub fn has_completions_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_completions_engine())
    }

    /// Check if any WorkerSet has an embeddings engine.
    pub fn has_embeddings_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_embeddings_engine())
    }

    /// Check if any WorkerSet has a tensor engine.
    pub fn has_tensor_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_tensor_engine())
    }

    /// Check if any WorkerSet has an images engine.
    pub fn has_images_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_images_engine())
    }

    /// Check if any WorkerSet has a videos engine.
    pub fn has_videos_engine(&self) -> bool {
        self.worker_sets.iter().any(|entry| entry.value().has_videos_engine())
    }

    /// Get the MDC checksum for a specific namespace's WorkerSet, if it exists.
    /// Returns None if no WorkerSet exists for this namespace (new namespace is OK).
    pub fn checksum_for_namespace(&self, namespace: &str) -> Option<String> {
        self.worker_sets
            .get(namespace)
            .map(|entry| entry.value().mdcsum().to_string())
    }

    // -- Engine accessors: select a WorkerSet, return its engine --

    pub fn get_chat_engine(&self) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.chat_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_completions_engine(
        &self,
    ) -> Result<OpenAICompletionsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.completions_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_embeddings_engine(
        &self,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.embeddings_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_images_engine(&self) -> Result<OpenAIImagesStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.images_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_videos_engine(&self) -> Result<OpenAIVideosStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.videos_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_tensor_engine(&self) -> Result<TensorStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.tensor_engine.clone())
            .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    // -- Combined engine + parsing options (atomically from one WorkerSet) --

    pub fn get_chat_engine_with_parsing(
        &self,
    ) -> Result<(OpenAIChatCompletionsStreamingEngine, ParsingOptions), ModelManagerError> {
        self.select_worker_set_with(|ws| {
            ws.chat_engine.clone().map(|e| (e, ws.parsing_options()))
        })
        .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    pub fn get_completions_engine_with_parsing(
        &self,
    ) -> Result<(OpenAICompletionsStreamingEngine, ParsingOptions), ModelManagerError> {
        self.select_worker_set_with(|ws| {
            ws.completions_engine
                .clone()
                .map(|e| (e, ws.parsing_options()))
        })
        .ok_or_else(|| ModelManagerError::ModelNotFound(self.name.clone()))
    }

    // -- Worker monitoring (aggregated across WorkerSets) --

    /// Get load threshold config from the first WorkerSet that has a monitor.
    /// Optionally updates the config.
    pub fn load_threshold_config(
        &self,
        config: Option<&LoadThresholdConfig>,
    ) -> Option<LoadThresholdConfig> {
        for entry in self.worker_sets.iter() {
            if let Some(ref monitor) = entry.value().worker_monitor {
                if let Some(cfg) = config {
                    monitor.set_load_threshold_config(cfg);
                }
                return Some(monitor.load_threshold_config());
            }
        }
        None
    }

    /// Get a worker monitor from the first WorkerSet that has one.
    pub fn get_worker_monitor(&self) -> Option<KvWorkerMonitor> {
        self.worker_sets
            .iter()
            .find_map(|entry| entry.value().worker_monitor.clone())
    }

    /// Total worker count across all WorkerSets.
    pub fn total_workers(&self) -> usize {
        self.worker_sets
            .iter()
            .map(|entry| entry.value().worker_count())
            .sum()
    }

    // -- Internal selection --

    /// Select a WorkerSet and extract a value from it.
    ///
    /// When there's only one set (steady state), returns from that set directly.
    /// With multiple sets (rollout), uses weighted random selection proportional
    /// to worker count, filtering to sets that have the requested engine.
    ///
    /// The `extract` closure should return `Some(value)` if the WorkerSet has the
    /// desired engine, or `None` if it doesn't.
    fn select_worker_set_with<T, F>(&self, extract: F) -> Option<T>
    where
        F: Fn(&WorkerSet) -> Option<T>,
    {
        // Fast path: single set
        if self.worker_sets.len() == 1 {
            return self
                .worker_sets
                .iter()
                .next()
                .and_then(|entry| extract(entry.value()));
        }

        // Collect eligible sets with their worker counts
        let eligible: Vec<(T, usize)> = self
            .worker_sets
            .iter()
            .filter_map(|entry| {
                let ws = entry.value();
                extract(ws).map(|val| (val, ws.worker_count().max(1)))
            })
            .collect();

        if eligible.is_empty() {
            return None;
        }

        if eligible.len() == 1 {
            return eligible.into_iter().next().map(|(val, _)| val);
        }

        // Weighted random selection proportional to worker count
        let total_weight: usize = eligible.iter().map(|(_, w)| w).sum();
        let mut pick = rand::rng().random_range(0..total_weight);
        for (val, weight) in eligible {
            if pick < weight {
                return Some(val);
            }
            pick -= weight;
        }
        // Should not reach here, but fallback to None
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;

    fn make_worker_set(namespace: &str, mdcsum: &str) -> Arc<WorkerSet> {
        Arc::new(WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        ))
    }

    fn make_worker_set_with_count(namespace: &str, mdcsum: &str, count: usize) -> Arc<WorkerSet> {
        let ws = Arc::new(WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        ));
        for _ in 0..count {
            ws.increment_workers();
        }
        ws
    }

    #[test]
    fn test_model_new() {
        let model = Model::new("llama".to_string());
        assert_eq!(model.name(), "llama");
        assert!(model.is_empty());
        assert_eq!(model.worker_set_count(), 0);
    }

    #[test]
    fn test_add_remove_worker_set() {
        let model = Model::new("llama".to_string());
        let ws = make_worker_set("ns1", "abc");

        model.add_worker_set("ns1".to_string(), ws);
        assert!(!model.is_empty());
        assert_eq!(model.worker_set_count(), 1);
        assert!(model.has_worker_set("ns1"));
        assert!(!model.has_worker_set("ns2"));

        let removed = model.remove_worker_set("ns1");
        assert!(removed.is_some());
        assert!(model.is_empty());

        let removed_again = model.remove_worker_set("ns1");
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_get_worker_set() {
        let model = Model::new("llama".to_string());
        let ws = make_worker_set("ns1", "abc");
        model.add_worker_set("ns1".to_string(), ws);

        let retrieved = model.get_worker_set("ns1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().namespace(), "ns1");

        assert!(model.get_worker_set("ns2").is_none());
    }

    #[test]
    fn test_multiple_worker_sets() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "def"));

        assert_eq!(model.worker_set_count(), 2);
        assert!(model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));

        model.remove_worker_set("ns1");
        assert_eq!(model.worker_set_count(), 1);
        assert!(!model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));
    }

    #[test]
    fn test_checksum_for_namespace() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc123"));
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "def456"));

        assert_eq!(model.checksum_for_namespace("ns1"), Some("abc123".to_string()));
        assert_eq!(model.checksum_for_namespace("ns2"), Some("def456".to_string()));
        assert_eq!(model.checksum_for_namespace("ns3"), None);
    }

    #[test]
    fn test_no_engines_means_prefill() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        // WorkerSets with no engines are treated as prefill sets
        assert!(model.has_prefill());
        assert!(!model.has_decode_engine());
        assert!(!model.has_chat_engine());
        assert!(!model.has_completions_engine());
        assert!(!model.has_embeddings_engine());
        assert!(!model.has_tensor_engine());
        assert!(!model.has_images_engine());
    }

    #[test]
    fn test_get_engine_returns_error_without_engines() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        assert!(model.get_chat_engine().is_err());
        assert!(model.get_completions_engine().is_err());
        assert!(model.get_embeddings_engine().is_err());
        assert!(model.get_images_engine().is_err());
        assert!(model.get_tensor_engine().is_err());
    }

    #[test]
    fn test_total_workers() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set_with_count("ns1", "abc", 5));
        model.add_worker_set("ns2".to_string(), make_worker_set_with_count("ns2", "def", 3));

        assert_eq!(model.total_workers(), 8);
    }

    #[test]
    fn test_worker_count_updates_through_model() {
        let model = Model::new("llama".to_string());
        let ws = make_worker_set("ns1", "abc");
        model.add_worker_set("ns1".to_string(), ws);

        // Get the worker set and modify its count
        let ws_ref = model.get_worker_set("ns1").unwrap();
        ws_ref.increment_workers();
        ws_ref.increment_workers();

        assert_eq!(model.total_workers(), 2);

        ws_ref.decrement_workers();
        assert_eq!(model.total_workers(), 1);
    }

    #[test]
    fn test_select_worker_set_with_extracts_namespace() {
        // Test that select_worker_set_with works by going through the public API.
        // Since we can't create real engines in tests, we verify that selection
        // returns None/Err when no engines are configured, which exercises the
        // filtering and selection code paths.
        let model = Model::new("llama".to_string());

        // Empty model
        assert!(model.get_chat_engine().is_err());

        // Single set (fast path)
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        assert!(model.get_chat_engine().is_err()); // No engine → filtered out

        // Multiple sets (weighted path)
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "def"));
        assert!(model.get_chat_engine().is_err()); // Still no engines → all filtered out
    }
}
