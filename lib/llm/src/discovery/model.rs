// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Model represents a named model (e.g., "llama-3-70b") that may be served by
//! one or more WorkerSets. Each WorkerSet corresponds to a namespace.
//!
//! Requests are routed to a WorkerSet selected by weighted random (proportional to worker count).

use std::sync::{Arc, OnceLock};

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
        completions::OpenAICompletionsStreamingEngine, embeddings::OpenAIEmbeddingsStreamingEngine,
        images::OpenAIImagesStreamingEngine, videos::OpenAIVideosStreamingEngine,
    },
};

/// A named model backed by one or more WorkerSets.
pub struct Model {
    name: String,
    worker_sets: DashMap<String, Arc<WorkerSet>>,
    /// The canonical MDC checksum for this model. Set by the first WorkerSet registered;
    /// all subsequent WorkerSets must match. Naturally cleared when the Model is dropped
    /// (last WorkerSet removed), allowing a new version to register.
    canonical_checksum: OnceLock<String>,
}

impl Model {
    pub fn new(name: String) -> Self {
        Self {
            name,
            worker_sets: DashMap::new(),
            canonical_checksum: OnceLock::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a WorkerSet to this model. Returns `Err` if the WorkerSet's checksum
    /// doesn't match the model's canonical checksum (set by the first WorkerSet).
    pub fn add_worker_set(
        &self,
        namespace: String,
        worker_set: Arc<WorkerSet>,
    ) -> Result<(), ModelManagerError> {
        self.set_canonical_checksum(worker_set.mdcsum())?;
        tracing::info!(
            model = %self.name,
            namespace = %namespace,
            "Adding worker set to model"
        );
        self.worker_sets.insert(namespace, worker_set);
        Ok(())
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
        self.worker_sets
            .get(namespace)
            .map(|entry| entry.value().clone())
    }

    pub fn is_empty(&self) -> bool {
        self.worker_sets.is_empty()
    }

    pub fn worker_set_count(&self) -> usize {
        self.worker_sets.len()
    }

    /// Check if this model has any decode engine (chat or completions) across any WorkerSet.
    pub fn has_decode_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_decode_engine())
    }

    /// Check if this model tracks prefill (any WorkerSet is a prefill set).
    pub fn has_prefill(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().is_prefill_set())
    }

    /// Check if any WorkerSet has a chat engine.
    pub fn has_chat_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_chat_engine())
    }

    /// Check if any WorkerSet has a completions engine.
    pub fn has_completions_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_completions_engine())
    }

    /// Check if any WorkerSet has an embeddings engine.
    pub fn has_embeddings_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_embeddings_engine())
    }

    /// Check if any WorkerSet has a tensor engine.
    pub fn has_tensor_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_tensor_engine())
    }

    /// Check if any WorkerSet has an images engine.
    pub fn has_images_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_images_engine())
    }

    /// Check if any WorkerSet has a videos engine.
    pub fn has_videos_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_videos_engine())
    }

    /// Check if a candidate checksum is valid for this model.
    /// Returns `Some(true)` if it matches the canonical checksum, `Some(false)` if it
    /// doesn't match, or `None` if no canonical checksum has been set yet (no WorkerSets).
    pub fn is_valid_checksum(&self, candidate: &str) -> Option<bool> {
        let canonical = self.canonical_checksum.get()?;
        Some(canonical == candidate)
    }

    /// Set the canonical checksum for this model. The first caller wins (OnceLock).
    /// Returns `Err` if a different checksum was already set.
    fn set_canonical_checksum(&self, checksum: &str) -> Result<(), ModelManagerError> {
        // Try to set; if already set, verify it matches.
        match self.canonical_checksum.set(checksum.to_string()) {
            Ok(()) => Ok(()),
            Err(_) => {
                // OnceLock was already set — check if the value matches
                let canonical = self.canonical_checksum.get().unwrap();
                if canonical == checksum {
                    Ok(())
                } else {
                    Err(ModelManagerError::ChecksumMismatch {
                        model: self.name.clone(),
                        expected: canonical.clone(),
                        got: checksum.to_string(),
                    })
                }
            }
        }
    }

    // -- Engine accessors: select a WorkerSet, return its engine --

    pub fn get_chat_engine(
        &self,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
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
        self.select_worker_set_with(|ws| ws.chat_engine.clone().map(|e| (e, ws.parsing_options())))
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
    /// When `config` is Some, updates ALL monitors (each WorkerSet has its own).
    pub fn load_threshold_config(
        &self,
        config: Option<&LoadThresholdConfig>,
    ) -> Option<LoadThresholdConfig> {
        let mut result = None;
        for entry in self.worker_sets.iter() {
            if let Some(ref monitor) = entry.value().worker_monitor {
                if let Some(cfg) = config {
                    monitor.set_load_threshold_config(cfg);
                }
                if result.is_none() {
                    result = Some(monitor.load_threshold_config());
                }
            }
        }
        result
    }

    /// Get the worker monitor for a specific namespace's WorkerSet.
    pub fn get_worker_monitor_for_namespace(&self, namespace: &str) -> Option<KvWorkerMonitor> {
        self.worker_sets
            .get(namespace)
            .and_then(|entry| entry.value().worker_monitor.clone())
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
    /// With multiple sets, uses weighted random selection proportional
    /// to worker count, filtering to sets that have the requested engine.
    ///
    /// The `extract` closure should return `Some(value)` if the WorkerSet has the
    /// desired engine, or `None` if it doesn't.
    fn select_worker_set_with<T, F>(&self, extract: F) -> Option<T>
    where
        F: Fn(&WorkerSet) -> Option<T>,
    {
        // Fast path: single set (same zero-worker filtering as the multi-set path below)
        // TODO: When the single set has 0 workers, this returns None which maps to
        // ModelNotFound (404). Ideally should be 503 "no available workers" — see follow-up.
        if self.worker_sets.len() == 1 {
            return self.worker_sets.iter().next().and_then(|entry| {
                let ws = entry.value();
                if ws.worker_count() == 0 {
                    return None;
                }
                extract(ws)
            });
        }

        // Collect eligible sets with their worker counts, skipping sets with no workers.
        // In-process models (no discovery watcher) return count=1, so they always participate.
        // Discovery models with count=0 have no available workers and are skipped.
        let eligible: Vec<(T, usize)> = self
            .worker_sets
            .iter()
            .filter_map(|entry| {
                let ws = entry.value();
                let count = ws.worker_count();
                if count == 0 {
                    return None;
                }
                extract(ws).map(|val| (val, count))
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
    use tokio::sync::watch;

    fn make_worker_set(namespace: &str, mdcsum: &str) -> Arc<WorkerSet> {
        Arc::new(WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        ))
    }

    /// Create a WorkerSet backed by a watch channel so worker_count reflects the vec length.
    fn make_worker_set_with_count(
        namespace: &str,
        mdcsum: &str,
        worker_ids: Vec<u64>,
    ) -> (Arc<WorkerSet>, watch::Sender<Vec<u64>>) {
        let (tx, rx) = watch::channel(worker_ids);
        let mut ws = WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        );
        ws.set_instance_watcher(rx);
        (Arc::new(ws), tx)
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

        model.add_worker_set("ns1".to_string(), ws).unwrap();
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
        model.add_worker_set("ns1".to_string(), ws).unwrap();

        let retrieved = model.get_worker_set("ns1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().namespace(), "ns1");

        assert!(model.get_worker_set("ns2").is_none());
    }

    #[test]
    fn test_multiple_worker_sets_same_checksum() {
        let model = Model::new("llama".to_string());
        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();
        model
            .add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"))
            .unwrap();

        assert_eq!(model.worker_set_count(), 2);
        assert!(model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));

        model.remove_worker_set("ns1");
        assert_eq!(model.worker_set_count(), 1);
        assert!(!model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));
    }

    #[test]
    fn test_add_worker_set_rejects_checksum_mismatch() {
        let model = Model::new("llama".to_string());
        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();

        // Different checksum from a different namespace should be rejected
        let result = model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "def"));
        assert!(result.is_err());
        assert_eq!(model.worker_set_count(), 1); // ns2 was not added
    }

    #[test]
    fn test_is_valid_checksum() {
        let model = Model::new("llama".to_string());

        // No canonical set yet
        assert_eq!(model.is_valid_checksum("abc123"), None);

        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc123"))
            .unwrap();

        // Matches canonical
        assert_eq!(model.is_valid_checksum("abc123"), Some(true));
        // Does not match canonical
        assert_eq!(model.is_valid_checksum("wrong"), Some(false));
    }

    #[test]
    fn test_no_engines_means_prefill() {
        let model = Model::new("llama".to_string());
        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();

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
        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();

        assert!(model.get_chat_engine().is_err());
        assert!(model.get_completions_engine().is_err());
        assert!(model.get_embeddings_engine().is_err());
        assert!(model.get_images_engine().is_err());
        assert!(model.get_tensor_engine().is_err());
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
        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(model.get_chat_engine().is_err()); // No engine → filtered out

        // Multiple sets (weighted path)
        model
            .add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"))
            .unwrap();
        assert!(model.get_chat_engine().is_err()); // Still no engines → all filtered out
    }

    #[test]
    fn test_total_workers_no_watcher() {
        // In-process WorkerSets (no watcher) default to worker_count=1
        let model = Model::new("llama".to_string());
        assert_eq!(model.total_workers(), 0); // empty model

        model
            .add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"))
            .unwrap();
        assert_eq!(model.total_workers(), 1);

        model
            .add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"))
            .unwrap();
        assert_eq!(model.total_workers(), 2);
    }

    #[test]
    fn test_total_workers_with_watcher() {
        let model = Model::new("llama".to_string());

        let (ws1, _tx1) = make_worker_set_with_count("ns1", "abc", vec![1, 2, 3]);
        let (ws2, _tx2) = make_worker_set_with_count("ns2", "abc", vec![10, 20]);
        model.add_worker_set("ns1".to_string(), ws1).unwrap();
        model.add_worker_set("ns2".to_string(), ws2).unwrap();

        assert_eq!(model.total_workers(), 5); // 3 + 2
    }

    #[test]
    fn test_total_workers_updates_dynamically() {
        let model = Model::new("llama".to_string());

        let (ws1, tx1) = make_worker_set_with_count("ns1", "abc", vec![1, 2]);
        model.add_worker_set("ns1".to_string(), ws1).unwrap();
        assert_eq!(model.total_workers(), 2);

        // Workers leave
        tx1.send(vec![1]).unwrap();
        assert_eq!(model.total_workers(), 1);

        // All workers gone
        tx1.send(vec![]).unwrap();
        assert_eq!(model.total_workers(), 0);
    }

    #[test]
    fn test_zero_worker_single_set_filtered() {
        // Single WorkerSet with 0 workers should be filtered by select_worker_set_with.
        // We test via select_worker_set_with's internal behavior: even though the set
        // exists and is_prefill_set() returns true, engine accessors should fail because
        // the zero-worker filter runs before the extract closure.
        let model = Model::new("llama".to_string());

        let (ws, _tx) = make_worker_set_with_count("ns1", "abc", vec![]);
        model.add_worker_set("ns1".to_string(), ws).unwrap();

        // WorkerSet exists but has 0 workers → selection filtered out → Err
        assert!(model.get_chat_engine().is_err());
        assert!(model.get_completions_engine().is_err());
    }

    #[test]
    fn test_zero_worker_multi_set_filtered() {
        // With multiple sets, only those with workers > 0 participate in selection.
        let model = Model::new("llama".to_string());

        let (ws1, _tx1) = make_worker_set_with_count("ns1", "abc", vec![]);
        let (ws2, _tx2) = make_worker_set_with_count("ns2", "abc", vec![]);
        model.add_worker_set("ns1".to_string(), ws1).unwrap();
        model.add_worker_set("ns2".to_string(), ws2).unwrap();

        // Both have 0 workers → all filtered → Err
        assert!(model.get_chat_engine().is_err());
    }
}
