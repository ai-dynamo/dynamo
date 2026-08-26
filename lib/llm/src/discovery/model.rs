// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Model represents a named model (e.g., "llama-3-70b") that may be served by
//! one or more WorkerSets. Each WorkerSet corresponds to a namespace.
//!
//! Requests are routed to a WorkerSet selected by weighted random (proportional to worker count).

use std::sync::Arc;

use dashmap::DashMap;
use rand::Rng;
use serde::Serialize;

use super::ModelManagerError;
use super::watcher::effective_worker_type;
use super::worker_monitor::LoadThresholdConfig;
use super::worker_set::WorkerSet;
use crate::protocols::openai::ParsingOptions;

use crate::types::{
    RealtimeBidirectionalEngine,
    generic::tensor::TensorStreamingEngine,
    openai::{
        audios::OpenAIAudiosStreamingEngine,
        chat_completions::OpenAIChatCompletionsStreamingEngine,
        classify::OpenAIClassifyStreamingEngine, completions::OpenAICompletionsStreamingEngine,
        embeddings::OpenAIEmbeddingsStreamingEngine, generate::GenerateStreamingEngine,
        images::OpenAIImagesStreamingEngine, pooling::OpenAIPoolingStreamingEngine,
        videos::OpenAIVideosStreamingEngine,
    },
};

/// Emit a one-time deprecation warning when serving-readiness falls back to
/// the legacy path because a namespace still contains a legacy card (a
/// worker with no declared `worker_type`). Logged once per process to avoid
/// spamming the per-request readiness hot path. Remove with the compat shim.
fn warn_legacy_readiness_once(model: &str, namespace: &str) {
    static LEGACY_READINESS_WARNED: std::sync::Once = std::sync::Once::new();
    LEGACY_READINESS_WARNED.call_once(|| {
        tracing::warn!(
            model = model,
            namespace = namespace,
            "Serving-readiness in compatibility mode, please upgrade the workers to latest version. This compatibility shim will be removed in a future release."
        );
    });
}

/// Per-worker-type detail within a namespace: how many workers of this type are
/// live and what peer types it depends on (DNF — a list of alternative AND-sets).
#[derive(Debug, Clone, Serialize)]
pub struct WorkerTypeReadiness {
    pub workers: usize,
    pub needs: Vec<Vec<String>>,
}

/// Worker readiness for one namespace (one deployment) of a model.
#[derive(Debug, Clone, Serialize)]
pub struct NamespaceReadiness {
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Worker type (lowercase `worker_type`) → detail. Legacy workers with no
    /// declared `worker_type` are not keyed here; see `reason`.
    pub worker_types: std::collections::BTreeMap<String, WorkerTypeReadiness>,
    pub present: Vec<String>,
    pub missing_worker_types: Vec<String>,
}

/// Structured worker readiness for a model across all its namespaces — the
/// response body of `GET /v1/models/{model}/ready`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelReadiness {
    pub model: String,
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub namespaces: std::collections::BTreeMap<String, NamespaceReadiness>,
}

/// Readiness facts for one namespace, from [`Model::evaluate_namespace`].
/// Shared by the serving gate and the `/ready` endpoint so they can't diverge.
struct NamespaceReadinessEval {
    ready: bool,
    has_legacy: bool,
    legacy_live_workers: usize,
    present: std::collections::HashSet<crate::worker_type::WorkerType>,
    missing: std::collections::HashSet<crate::worker_type::WorkerType>,
    ambiguous: std::collections::HashSet<crate::worker_type::WorkerType>,
}

/// A named model backed by one or more WorkerSets.
pub struct Model {
    name: String,
    worker_sets: DashMap<String, Arc<WorkerSet>>,
}

/// A value extracted from one of a model's WorkerSets, paired with the namespace
/// of the set it came from.
///
/// The namespace must come out of the *same* selection that produced the value.
/// A model with WorkerSets in several namespaces picks one per request by
/// weighted random choice, so asking again afterwards can name a different
/// deployment than the one actually serving the request — which would silently
/// misattribute that request's metrics.
#[derive(Debug, Clone)]
pub struct Selected<T> {
    pub value: T,
    pub namespace: String,
}

impl<T> Selected<T> {
    /// Apply `f` to the value, keeping the namespace it was selected from.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Selected<U> {
        Selected {
            value: f(self.value),
            namespace: self.namespace,
        }
    }

    /// Discard the namespace. Use only where the caller records no metrics.
    pub fn into_value(self) -> T {
        self.value
    }
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

    /// Add a WorkerSet to this model.
    pub fn add_worker_set(&self, namespace: String, worker_set: Arc<WorkerSet>) {
        tracing::info!(
            model = %self.name,
            namespace = %namespace,
            "Adding worker set to model"
        );
        self.worker_sets.insert(namespace, worker_set);
    }

    /// Check whether a candidate checksum is compatible with an existing WorkerSet
    /// identified by `ws_key`.
    pub fn is_checksum_compatible(&self, ws_key: &str, candidate_checksum: &str) -> bool {
        match self.worker_sets.get(ws_key) {
            Some(existing_ws) => existing_ws.mdcsum() == candidate_checksum,
            None => true,
        }
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

    /// Snapshot all WorkerSets. Used by cross-role lifecycle coordination
    /// where storage keys include role/surface suffixes but deployment
    /// matching is based on `WorkerSet::namespace()`.
    pub(crate) fn worker_sets(&self) -> Vec<Arc<WorkerSet>> {
        self.worker_sets
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Build an immutable membership snapshot for request-plane publication.
    ///
    /// WorkerSets themselves are shared because their engines and routing lifecycle are
    /// long-lived. The membership map is copied so later discovery mutations cannot leak
    /// through an older published catalog.
    pub(crate) fn snapshot(&self) -> Self {
        let snapshot = Self::new(self.name.clone());
        for entry in &self.worker_sets {
            snapshot
                .worker_sets
                .insert(entry.key().clone(), entry.value().clone());
        }
        snapshot
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

    /// Check if any WorkerSet has a classify engine.
    pub fn has_classify_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_classify_engine())
    }

    /// Check if any WorkerSet has a pooling engine.
    pub fn has_pooling_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_pooling_engine())
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

    /// Check if any WorkerSet has an audios engine.
    pub fn has_audios_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_audios_engine())
    }

    /// Check if any WorkerSet has a realtime engine.
    pub fn has_realtime_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_realtime_engine())
    }

    /// Check if any WorkerSet has a generate engine.
    pub fn has_generate_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_generate_engine())
    }

    /// Check whether a Generate worker also advertises `capability`.
    pub fn has_generate_engine_for_capability(&self, capability: &str) -> bool {
        self.worker_sets.iter().any(|entry| {
            let worker_set = entry.value();
            worker_set.has_generate_engine() && worker_set.supports_runtime_capability(capability)
        })
    }

    // -- Model serving readiness --
    //
    // The set of WorkerSets in this Model that share the same `namespace`
    // string collectively serve traffic for one deployment. A worker's
    // `needs` is in DNF: a list of alternative AND-sets of required peer
    // worker types. The namespace is ready when, for every WorkerSet in it,
    // at least one alternative is fully covered by the worker types
    // currently present (workers with worker_count > 0).
    //

    /// Distinct namespaces represented by this model's WorkerSets, sorted.
    /// Each namespace identifies one deployment of the model.
    /// Whether any WorkerSet places this model in the `(namespace, worker_type)`
    /// deployment that the frontend's per-deployment gauges are keyed by.
    ///
    /// `worker_type` is compared against the same [`effective_worker_type`]
    /// rendering the watcher used when it emitted `ModelUpdate::Added`, so the
    /// add and remove sides cannot disagree on a legacy card whose `worker_type`
    /// is unset.
    pub fn has_deployment(&self, namespace: &str, worker_type: &str) -> bool {
        self.worker_sets.iter().any(|entry| {
            let ws = entry.value();
            ws.namespace() == namespace
                && effective_worker_type(ws.card().worker_type, ws.card().model_type).as_str()
                    == worker_type
        })
    }

    pub fn distinct_namespaces_sorted(&self) -> Vec<String> {
        let mut ns: Vec<String> = self
            .worker_sets
            .iter()
            .map(|entry| entry.value().namespace().to_string())
            .collect();
        ns.sort();
        ns.dedup();
        ns
    }

    /// Return `(worker_type, needs)` for this WorkerSet, or `None` if the
    /// card has no declared `worker_type` (a legacy card from an old,
    /// pre-`worker_type` worker). The readiness path treats `None` as the
    /// signal to fall back to the cross-version compat path — see
    /// `is_workers_ready`.
    fn ws_type_and_needs(
        ws: &WorkerSet,
    ) -> Option<(
        crate::worker_type::WorkerType,
        Vec<Vec<crate::worker_type::WorkerType>>,
    )> {
        let card = ws.card();
        card.worker_type.map(|wt| (wt, card.needs.clone()))
    }

    /// Whether the workers in the given namespace are ready to serve traffic.
    ///
    /// Iterates the WorkerSets sharing this namespace and checks that every
    /// WorkerSet's `needs` (DNF) has at least one alternative fully covered
    /// by the present worker types. Returns false for an unknown namespace or
    /// one with no WorkerSets.
    ///
    /// **Cross-version compat:** if the namespace still contains a legacy card
    /// (a worker with no declared `worker_type`, i.e. an old pre-`worker_type`
    /// binary), its disaggregated worker types cannot be reconstructed — old decode
    /// and old aggregated workers are indistinguishable on the wire. Rather than
    /// hide the model, we fall back to legacy behavior and report ready as long
    /// as some worker is live. Strict worker-type readiness gating resumes automatically once
    /// every worker in the namespace carries a `worker_type`.
    ///
    /// TODO(v1.5): Remove this branch with the legacy MDC topology shims after
    /// the v1.2 compatibility window expires.
    pub fn is_workers_ready(&self, namespace: &str) -> bool {
        let wsets: Vec<Arc<WorkerSet>> = self
            .worker_sets
            .iter()
            .filter(|entry| entry.value().namespace() == namespace)
            .map(|entry| entry.value().clone())
            .collect();
        self.evaluate_namespace(&wsets).ready
    }

    /// Evaluate readiness for one namespace's WorkerSets — the shared check
    /// behind both the serving gate and the `/ready` endpoint.
    ///
    /// Strict (non-legacy) semantics: ready when a worker is live, every
    /// registered worker type has a live worker, and every *live* WorkerSet's
    /// `needs` DNF is satisfied; anything absent goes in `missing`. A dead
    /// WorkerSet's needs are ignored (a live sibling of the same type carries
    /// the same `needs`). A legacy card (no `worker_type`) bypasses the strict
    /// check: ready iff any worker is live. Empty `wsets` is not ready.
    fn evaluate_namespace(&self, wsets: &[Arc<WorkerSet>]) -> NamespaceReadinessEval {
        let units = wsets
            .iter()
            .map(|ws| match Self::ws_type_and_needs(ws) {
                Some((worker_type, needs)) => super::readiness::ReadinessUnit {
                    worker_type: Some(worker_type),
                    live_count: ws.worker_count(),
                    needs,
                },
                None => super::readiness::ReadinessUnit {
                    worker_type: None,
                    live_count: ws.worker_count(),
                    needs: Vec::new(),
                },
            })
            .collect::<Vec<_>>();
        let eval = super::readiness::evaluate_readiness(&units);
        if eval.has_legacy {
            warn_legacy_readiness_once(&self.name, wsets[0].namespace());
        }
        NamespaceReadinessEval {
            ready: eval.ready,
            has_legacy: eval.has_legacy,
            legacy_live_workers: eval.legacy_live_workers,
            present: eval.present,
            missing: eval.missing,
            ambiguous: eval.ambiguous,
        }
    }

    /// Return the namespace identifier of the first ready set of workers (in
    /// sorted order), or `None` if none are ready.
    pub fn first_ready_workers(&self) -> Option<String> {
        self.distinct_namespaces_sorted()
            .into_iter()
            .find(|ns| self.is_workers_ready(ns))
    }

    /// Whether at least one namespace's worker set in this model is ready to
    /// serve traffic.
    pub fn has_ready_workers(&self) -> bool {
        self.first_ready_workers().is_some()
    }

    /// Structured per-namespace worker readiness for this model — the data
    /// behind the `GET /v1/models/{model}/ready` observability endpoint.
    ///
    /// Built on the same `Self::evaluate_namespace` facts the serving gate
    /// uses, so the reported `ready`/`missing` can never disagree with routing;
    /// this method only layers display data (per-type counts, reason strings).
    pub fn namespace_readiness(&self) -> ModelReadiness {
        let mut namespaces = std::collections::BTreeMap::new();

        for ns in self.distinct_namespaces_sorted() {
            let wsets: Vec<Arc<WorkerSet>> = self
                .worker_sets
                .iter()
                .filter(|entry| entry.value().namespace() == ns)
                .map(|entry| entry.value().clone())
                .collect();

            // Authoritative readiness facts (shared with the serving gate).
            let eval = self.evaluate_namespace(&wsets);

            // Display layer: per-type live-worker counts and declared `needs`.
            let mut worker_types: std::collections::BTreeMap<String, WorkerTypeReadiness> =
                std::collections::BTreeMap::new();
            for ws in &wsets {
                let card = ws.card();
                if let Some(wt) = card.worker_type {
                    let entry = worker_types
                        .entry(wt.as_str().to_string())
                        .or_insert_with(|| WorkerTypeReadiness {
                            workers: 0,
                            needs: card
                                .needs
                                .iter()
                                .map(|alt| alt.iter().map(|t| t.as_str().to_string()).collect())
                                .collect(),
                        });
                    entry.workers += ws.worker_count();
                }
            }

            let mut present_vec: Vec<String> = eval
                .present
                .iter()
                .map(|wt| wt.as_str().to_string())
                .collect();
            present_vec.sort();
            let mut missing_vec: Vec<String> = eval
                .missing
                .iter()
                .map(|wt| wt.as_str().to_string())
                .collect();
            missing_vec.sort();

            let reason = if eval.ready {
                if eval.has_legacy {
                    let legacy_live_workers = eval.legacy_live_workers;
                    Some(format!(
                        "legacy worker(s) present (no worker_type); readiness gating bypassed \
                         (ready while {legacy_live_workers} worker(s) live) — compat window only"
                    ))
                } else {
                    None
                }
            } else if eval.has_legacy {
                Some("legacy worker(s) present but no live worker".to_string())
            } else if !eval.ambiguous.is_empty() {
                let mut roles = eval
                    .ambiguous
                    .iter()
                    .map(|worker_type| worker_type.as_str())
                    .collect::<Vec<_>>();
                roles.sort_unstable();
                Some(format!("ambiguous worker types: {}", roles.join(", ")))
            } else {
                Some(format!("missing worker types: {}", missing_vec.join(", ")))
            };

            namespaces.insert(
                ns.clone(),
                NamespaceReadiness {
                    ready: eval.ready,
                    reason,
                    worker_types,
                    present: present_vec,
                    missing_worker_types: missing_vec,
                },
            );
        }

        let ready = namespaces.values().any(|n| n.ready);
        ModelReadiness {
            model: self.name.clone(),
            ready,
            reason: if ready {
                None
            } else {
                Some("no namespace has all required worker types live".to_string())
            },
            namespaces,
        }
    }

    /// Whether this model can serve at least one inference request right now.
    ///
    /// Differs from [`Self::is_displayable`] in that it does **not** fall back
    /// to prefill-only WorkerSets: requires a WorkerSet that has a serving
    /// engine attached and workers connected.
    /// Used by KServe gRPC `model_ready` / `server_ready` to avoid the race
    /// where a `ModelDeploymentCard` is registered before its WorkerSet has
    /// been wired up.
    ///
    /// Delegates to `Self::select_worker_set_with` so readiness reports
    /// exactly what request routing would accept — including the namespace
    /// completeness gate. Without that, a live decode-only WorkerSet with a
    /// chat engine but no prefill peer would report ready while every request
    /// was rejected.
    pub fn is_ready_to_serve(&self) -> bool {
        self.select_worker_set_with(|ws| ws.has_any_serving_engine().then_some(()))
            .is_some()
    }

    /// Whether this model should be visible in /v1/models.
    pub fn is_displayable(&self) -> bool {
        let any_set_has_engine = self
            .worker_sets
            .iter()
            .any(|entry| entry.value().has_any_serving_engine());

        self.worker_sets.iter().any(|entry| {
            let ws = entry.value();
            if ws.worker_count() == 0 {
                return false;
            }
            ws.has_any_serving_engine() || (!any_set_has_engine && ws.is_prefill_set())
        })
    }

    // -- Engine accessors: select a WorkerSet, return its engine --

    pub fn get_chat_engine(
        &self,
    ) -> Result<Selected<OpenAIChatCompletionsStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.chat_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_chat_engine()))
    }

    pub fn get_completions_engine(
        &self,
    ) -> Result<Selected<OpenAICompletionsStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.completions_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_completions_engine()))
    }

    pub fn get_embeddings_engine(
        &self,
    ) -> Result<Selected<OpenAIEmbeddingsStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.embeddings_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_embeddings_engine()))
    }

    pub fn get_classify_engine(
        &self,
    ) -> Result<Selected<OpenAIClassifyStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.classify_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_classify_engine()))
    }

    pub fn get_pooling_engine(
        &self,
    ) -> Result<Selected<OpenAIPoolingStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.pooling_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_pooling_engine()))
    }

    pub fn get_images_engine(
        &self,
    ) -> Result<Selected<OpenAIImagesStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.images_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_images_engine()))
    }

    pub fn get_videos_engine(
        &self,
    ) -> Result<Selected<OpenAIVideosStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.videos_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_videos_engine()))
    }

    pub fn get_audios_engine(
        &self,
    ) -> Result<Selected<OpenAIAudiosStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.audios_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_audios_engine()))
    }

    pub fn get_tensor_engine(&self) -> Result<Selected<TensorStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.tensor_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_tensor_engine()))
    }

    pub fn get_realtime_engine(
        &self,
    ) -> Result<Selected<RealtimeBidirectionalEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.realtime_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_realtime_engine()))
    }

    pub fn get_generate_engine(
        &self,
    ) -> Result<Selected<GenerateStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.generate_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_generate_engine()))
    }
    /// Get a Generate engine from a worker advertising `capability`.
    pub fn get_generate_engine_for_capability(
        &self,
        capability: &str,
    ) -> Result<Selected<GenerateStreamingEngine>, ModelManagerError> {
        self.select_worker_set_with(|worker_set| {
            worker_set
                .supports_runtime_capability(capability)
                .then(|| worker_set.generate_engine.clone())
                .flatten()
        })
        .ok_or_else(|| self.engine_error(self.has_generate_engine_for_capability(capability)))
    }

    // -- Combined engine + parsing options (atomically from one WorkerSet) --

    pub fn get_chat_engine_with_parsing(
        &self,
    ) -> Result<Selected<(OpenAIChatCompletionsStreamingEngine, ParsingOptions)>, ModelManagerError>
    {
        self.select_worker_set_with(|ws| ws.chat_engine.clone().map(|e| (e, ws.parsing_options())))
            .ok_or_else(|| self.engine_error(self.has_chat_engine()))
    }

    pub fn get_completions_engine_with_parsing(
        &self,
    ) -> Result<Selected<(OpenAICompletionsStreamingEngine, ParsingOptions)>, ModelManagerError>
    {
        self.select_worker_set_with(|ws| {
            ws.completions_engine
                .clone()
                .map(|e| (e, ws.parsing_options()))
        })
        .ok_or_else(|| self.engine_error(self.has_completions_engine()))
    }

    pub fn get_generate_engine_with_parsing(
        &self,
    ) -> Result<Selected<(GenerateStreamingEngine, ParsingOptions)>, ModelManagerError> {
        self.select_worker_set_with(|ws| {
            ws.generate_engine
                .clone()
                .map(|e| (e, ws.parsing_options()))
        })
        .ok_or_else(|| self.engine_error(self.has_generate_engine()))
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

    /// Total worker count across all WorkerSets.
    pub fn total_workers(&self) -> usize {
        self.worker_sets
            .iter()
            .map(|entry| entry.value().worker_count())
            .sum()
    }

    // -- Internal helpers --

    /// Return the appropriate error when no servable WorkerSet was found.
    /// If the engine exists but no WorkerSet can serve (zero workers, prefill not activated,
    /// etc.), return ModelUnavailable (maps to 503). Otherwise ModelNotFound (maps to 404).
    fn engine_error(&self, engine_exists: bool) -> ModelManagerError {
        if engine_exists {
            ModelManagerError::ModelUnavailable(self.name.clone())
        } else {
            ModelManagerError::ModelNotFound(self.name.clone())
        }
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
    ///
    fn select_worker_set_with<T, F>(&self, extract: F) -> Option<Selected<T>>
    where
        F: Fn(&WorkerSet) -> Option<T>,
    {
        // One snapshot drives both the readiness filter and candidate
        // eligibility, so a concurrent add/remove can't make us treat a
        // namespace as complete while routing to a set that lost a peer. It also
        // avoids re-entering the DashMap mid-iteration, which can deadlock.
        let snapshot: Vec<Arc<WorkerSet>> = self
            .worker_sets
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Namespaces whose worker set is complete, evaluated against the snapshot.
        let mut namespaces: Vec<&str> = snapshot.iter().map(|ws| ws.namespace()).collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        let ready_namespaces: std::collections::HashSet<&str> = namespaces
            .into_iter()
            .filter(|ns| {
                let in_ns: Vec<Arc<WorkerSet>> = snapshot
                    .iter()
                    .filter(|ws| ws.namespace() == *ns)
                    .cloned()
                    .collect();
                self.evaluate_namespace(&in_ns).ready
            })
            .collect();

        // Carry the borrowed WorkerSet alongside each candidate and only allocate the
        // winner's namespace. Building `Selected` per candidate would allocate a String
        // for every eligible set on the request hot path and discard all but one.
        let selected = |ws: &WorkerSet, value: T| Selected {
            value,
            namespace: ws.metrics_namespace().to_string(),
        };

        // Fast path: single set (same zero-worker filtering as the multi-set path below)
        if snapshot.len() == 1 {
            let ws = &snapshot[0];
            if ws.worker_count() == 0 || !ready_namespaces.contains(ws.namespace()) {
                return None;
            }
            return extract(ws).map(|val| selected(ws, val));
        }

        // Collect eligible sets with their worker counts, skipping sets with no workers or sets in
        // a namespace whose worker set is incomplete.
        // In-process models (no discovery watcher) return count=1, so they always participate.
        // Discovery models with count=0 have no available workers and are skipped.
        let eligible: Vec<(T, &Arc<WorkerSet>, usize)> = snapshot
            .iter()
            .filter_map(|ws| {
                let count = ws.worker_count();
                if count == 0 || !ready_namespaces.contains(ws.namespace()) {
                    return None;
                }
                extract(ws).map(|val| (val, ws, count))
            })
            .collect();

        if eligible.is_empty() {
            return None;
        }

        if eligible.len() == 1 {
            let (val, ws, _) = eligible.into_iter().next()?;
            return Some(selected(ws, val));
        }

        // Weighted random selection proportional to worker count
        let total_weight: usize = eligible.iter().map(|(_, _, w)| w).sum();
        let mut pick = rand::rng().random_range(0..total_weight);
        for (val, ws, weight) in eligible {
            if pick < weight {
                return Some(selected(ws, val));
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
    fn test_multiple_worker_sets_same_checksum() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"));

        assert_eq!(model.worker_set_count(), 2);
        assert!(model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));

        model.remove_worker_set("ns1");
        assert_eq!(model.worker_set_count(), 1);
        assert!(!model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));
    }

    #[test]
    fn test_multiple_worker_sets_different_checksums() {
        // Different namespaces are allowed to have different checksums
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "def"));

        assert_eq!(model.worker_set_count(), 2);
        assert!(model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));
    }

    #[test]
    fn test_is_checksum_compatible_no_existing_worker_set() {
        let model = Model::new("llama".to_string());
        // No WorkerSet exists yet — any checksum is compatible
        assert!(model.is_checksum_compatible("ns1", "abc"));
        assert!(model.is_checksum_compatible("ns1", "xyz"));
    }

    #[test]
    fn test_is_checksum_compatible_matching_checksum() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        // Same ws_key, same checksum → compatible
        assert!(model.is_checksum_compatible("ns1", "abc"));
    }

    #[test]
    fn test_is_checksum_compatible_mismatched_checksum() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        // Same ws_key, different checksum → incompatible
        assert!(!model.is_checksum_compatible("ns1", "def"));
    }

    #[test]
    fn test_is_checksum_compatible_different_ws_key() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        // Different ws_key — no existing WorkerSet for "ns2", so any checksum is fine
        assert!(model.is_checksum_compatible("ns2", "def"));
        assert!(model.is_checksum_compatible("ns2", "abc"));
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
        assert!(model.get_realtime_engine().is_err());
    }

    fn make_realtime_worker_set(namespace: &str) -> Arc<WorkerSet> {
        let mut ws = WorkerSet::new(
            namespace.to_string(),
            "abc".to_string(),
            ModelDeploymentCard::default(),
        );
        ws.realtime_engine = Some(Arc::new(crate::engines::EchoBidirectionalEngine));
        Arc::new(ws)
    }

    #[test]
    fn test_realtime_engine_round_trip() {
        let model = Model::new("realtime-mock".to_string());
        model.add_worker_set("ns1".to_string(), make_realtime_worker_set("ns1"));
        assert!(model.has_realtime_engine());
        assert!(model.get_realtime_engine().is_ok());
    }

    #[test]
    fn test_realtime_only_model_is_displayable() {
        let model = Model::new("realtime-mock".to_string());
        model.add_worker_set("ns1".to_string(), make_realtime_worker_set("ns1"));
        assert!(model.is_displayable());
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
        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"));
        assert!(model.get_chat_engine().is_err()); // Still no engines → all filtered out
    }

    #[test]
    fn test_total_workers_no_watcher() {
        // In-process WorkerSets (no watcher) default to worker_count=1
        let model = Model::new("llama".to_string());
        assert_eq!(model.total_workers(), 0); // empty model

        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        assert_eq!(model.total_workers(), 1);

        model.add_worker_set("ns2".to_string(), make_worker_set("ns2", "abc"));
        assert_eq!(model.total_workers(), 2);
    }

    #[test]
    fn test_total_workers_with_watcher() {
        let model = Model::new("llama".to_string());

        let (ws1, _tx1) = make_worker_set_with_count("ns1", "abc", vec![1, 2, 3]);
        let (ws2, _tx2) = make_worker_set_with_count("ns2", "abc", vec![10, 20]);
        model.add_worker_set("ns1".to_string(), ws1);
        model.add_worker_set("ns2".to_string(), ws2);

        assert_eq!(model.total_workers(), 5); // 3 + 2
    }

    #[test]
    fn test_total_workers_updates_dynamically() {
        let model = Model::new("llama".to_string());

        let (ws1, tx1) = make_worker_set_with_count("ns1", "abc", vec![1, 2]);
        model.add_worker_set("ns1".to_string(), ws1);
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
        model.add_worker_set("ns1".to_string(), ws);

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
        model.add_worker_set("ns1".to_string(), ws1);
        model.add_worker_set("ns2".to_string(), ws2);

        // Both have 0 workers → all filtered → Err
        assert!(model.get_chat_engine().is_err());
    }

    // -- Disaggregated prefill death tests --

    use crate::kv_router::PrefillRouter;

    /// Build a WorkerSet with a deactivated PrefillRouter simulating "was activated, now dead".
    /// worker_count defaults to 1 (no instance_count_rx -> in-process default).
    fn make_worker_set_with_dead_prefill(namespace: &str) -> Arc<WorkerSet> {
        let mut ws = WorkerSet::new(
            namespace.to_string(),
            "abc".to_string(),
            crate::model_card::ModelDeploymentCard::default(),
        );
        let pr = PrefillRouter::disabled(
            std::sync::Arc::new(crate::discovery::ModelManager::new()),
            dynamo_runtime::pipeline::RouterMode::RoundRobin,
            None,
        );
        pr.set_target(None);
        ws.prefill_router = Some(pr);
        Arc::new(ws)
    }

    /// Baseline: a WorkerSet without a PrefillRouter is always displayable
    /// (worker_count=1, is_prefill_set=true).
    #[test]
    fn test_is_displayable_true_basic() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));
        assert!(
            model.is_displayable(),
            "model with an unconstrained WorkerSet must be displayable"
        );
    }

    /// Prefill-router lifecycle is not a separate model-visibility policy. Registered worker
    /// topology gates the serving-ready model list and request selection.
    #[test]
    fn test_is_displayable_ignores_prefill_router_lifecycle() {
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set_with_dead_prefill("ns1"));

        assert!(
            model.is_displayable(),
            "prefill-router lifecycle must not override registered topology"
        );
    }

    // -- Encode-set visibility --
    //
    // Encode workers are reached through encoder routing, not the public
    // chat/completions surface. Tests below verify is_displayable correctly
    // hides Encode-only
    // deployments from /v1/models and continues to surface mixed
    // Aggregated+Encode deployments via the Aggregated set only.

    /// Helper: build an Encode-role WorkerSet wrapped in Arc. worker_count=1
    /// via a live watcher so is_displayable doesn't filter it at the
    /// worker_count == 0 guard. Dropping the sender is fine: tokio's
    /// `watch::Receiver::borrow()` returns the current value even after the
    /// sender closes, which is all `is_displayable` reads.
    fn make_encode_worker_set(namespace: &str, mdcsum: &str) -> Arc<WorkerSet> {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Encode);
        let (_tx, rx) = watch::channel(vec![1_u64]);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        ws.set_instance_watcher(rx);
        Arc::new(ws)
    }

    #[test]
    fn encode_only_model_is_not_displayable() {
        // An Encode worker has no serving engines (the watcher's role gate
        // skips pipeline construction) and is_prefill_set excludes Encode,
        // so an Encode-only model has no displayable WorkerSet and stays
        // hidden from /v1/models. The frontend's chat/completions surface
        // is not where users reach Encode workers; encoder routing is.
        let model = Model::new("llava".to_string());
        model.add_worker_set(
            "dynamo:encode".to_string(),
            make_encode_worker_set("dynamo", "mdc-e"),
        );

        assert!(
            !model.is_displayable(),
            "Encode-only model must be hidden from /v1/models -- Encode workers \
             aren't a public serving surface"
        );
    }

    #[test]
    fn aggregated_plus_encode_model_is_displayable_via_aggregated() {
        // E/Agg topology: an Aggregated worker plus an Encode peer. The
        // Aggregated set is engineless in this stub but is_prefill_set
        // remains true for it (legacy classification), so the model is
        // displayable via the Aggregated WorkerSet's fallback. The Encode
        // WorkerSet does NOT contribute to displayability -- it's filtered
        // by is_encode_set excluding it from is_prefill_set.
        let model = Model::new("llava".to_string());
        model.add_worker_set("dynamo".to_string(), make_worker_set("dynamo", "mdc-a"));
        model.add_worker_set(
            "dynamo:encode".to_string(),
            make_encode_worker_set("dynamo", "mdc-e"),
        );

        assert!(
            model.is_displayable(),
            "Aggregated+Encode model must be displayable via the Aggregated set"
        );

        // Removing the Aggregated set leaves only the Encode set -- model
        // must flip to hidden (the bug we're guarding against would have
        // kept it displayable via the Encode set being misclassified as
        // prefill).
        model.remove_worker_set("dynamo");
        assert!(
            !model.is_displayable(),
            "after Aggregated leaves, Encode-only model must be hidden"
        );
    }

    // -- Model serving readiness --
    //
    // These tests exercise the live-compute readiness methods on `Model`.
    // They construct WorkerSets with specific `worker_type` / `needs` values
    // on their cards and verify DNF readiness math, including the encode
    // worker's two-alternative needs and the rejection of cards with no
    // declared `worker_type`.

    use crate::worker_type::WorkerType;

    /// Build a WorkerSet with an explicit worker_type / needs and a live
    /// worker count (via a watch channel).
    fn ws_with_type(
        namespace: &str,
        mdcsum: &str,
        worker_type: WorkerType,
        needs: Vec<Vec<WorkerType>>,
        worker_ids: Vec<u64>,
    ) -> (Arc<WorkerSet>, watch::Sender<Vec<u64>>) {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(worker_type);
        card.needs = needs;
        let (tx, rx) = watch::channel(worker_ids);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        ws.set_instance_watcher(rx);
        (Arc::new(ws), tx)
    }

    #[test]
    fn readiness_empty_model_not_ready() {
        let model = Model::new("llama".to_string());
        assert!(!model.has_ready_workers());
        assert_eq!(model.first_ready_workers(), None);
        assert!(!model.is_workers_ready("dynamo"));
    }

    #[test]
    fn readiness_pd_pair_ready() {
        let model = Model::new("llama".to_string());
        let (prefill, _tx_p) = ws_with_type(
            "dynamo",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (decode, _tx_d) = ws_with_type(
            "dynamo",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("dynamo:prefill".to_string(), prefill);
        model.add_worker_set("dynamo".to_string(), decode);

        assert!(model.is_workers_ready("dynamo"));
        assert_eq!(model.first_ready_workers(), Some("dynamo".to_string()));
    }

    #[test]
    fn readiness_pd_missing_prefill_not_ready() {
        let model = Model::new("llama".to_string());
        let (decode, _tx) = ws_with_type(
            "dynamo",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("dynamo".to_string(), decode);

        assert!(!model.is_workers_ready("dynamo"));
    }

    #[test]
    fn readiness_epd_aggregated_plus_encode_ready() {
        // E-PD pattern: Aggregated worker (with --route-to-encoder, so it
        // needs Encode) + Encode worker (whose needs has two alternatives:
        // P+D pair OR a single Aggregated peer). The second alternative is
        // satisfied here because Aggregated is present.
        let model = Model::new("llava".to_string());
        let (agg, _tx_a) = ws_with_type(
            "dynamo",
            "mdc-a",
            WorkerType::Aggregated,
            vec![vec![WorkerType::Encode]],
            vec![1],
        );
        let (enc, _tx_e) = ws_with_type(
            "dynamo",
            "mdc-e",
            WorkerType::Encode,
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
            vec![2],
        );
        model.add_worker_set("dynamo:aggregated".to_string(), agg);
        model.add_worker_set("dynamo:encode".to_string(), enc);

        assert!(model.is_workers_ready("dynamo"));
    }

    #[test]
    fn readiness_epd_pd_pair_plus_encode_ready() {
        // E-P-D pattern: separate Prefill + Decode + Encode workers.
        // Encode's first alternative (Prefill+Decode) is satisfied.
        let model = Model::new("llava".to_string());
        let (prefill, _tx_p) = ws_with_type(
            "dynamo",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode, WorkerType::Encode]],
            vec![1],
        );
        let (decode, _tx_d) = ws_with_type(
            "dynamo",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        let (enc, _tx_e) = ws_with_type(
            "dynamo",
            "mdc-e",
            WorkerType::Encode,
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
            vec![3],
        );
        model.add_worker_set("dynamo:prefill".to_string(), prefill);
        model.add_worker_set("dynamo".to_string(), decode);
        model.add_worker_set("dynamo:encode".to_string(), enc);

        assert!(model.is_workers_ready("dynamo"));
    }

    #[test]
    fn readiness_encode_alone_not_ready() {
        // Encode alone: neither alternative in its needs DNF is satisfied.
        let model = Model::new("llava".to_string());
        let (enc, _tx) = ws_with_type(
            "dynamo",
            "mdc-e",
            WorkerType::Encode,
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
            vec![1],
        );
        model.add_worker_set("dynamo:encode".to_string(), enc);

        assert!(!model.is_workers_ready("dynamo"));
    }

    #[test]
    fn readiness_cross_namespace_isolation() {
        // Prefill in ns-old, Decode in ns-new: neither namespace is ready.
        let model = Model::new("llama".to_string());
        let (p, _tp) = ws_with_type(
            "ns-old",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (d, _td) = ws_with_type(
            "ns-new",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("ns-old:prefill".to_string(), p);
        model.add_worker_set("ns-new".to_string(), d);

        assert!(!model.is_workers_ready("ns-old"));
        assert!(!model.is_workers_ready("ns-new"));
        assert!(!model.has_ready_workers());
    }

    #[test]
    fn readiness_scale_down_flips_to_not_ready() {
        // Decode worker_count drops to 0 → readiness flips from ready to
        // not-ready with no clearing hook (the point of live-compute).
        let model = Model::new("llama".to_string());
        let (p, _tp) = ws_with_type(
            "dynamo",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (d, tx_d) = ws_with_type(
            "dynamo",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("dynamo:prefill".to_string(), p);
        model.add_worker_set("dynamo".to_string(), d);

        assert!(model.is_workers_ready("dynamo"));

        // Drop decode workers — no hook, just an update to the watch channel.
        tx_d.send(vec![]).unwrap();
        assert!(!model.is_workers_ready("dynamo"));

        // Rejoin a decode worker — readiness flips back to ready.
        tx_d.send(vec![2]).unwrap();
        assert!(model.is_workers_ready("dynamo"));
    }

    // -- Cross-version compat: legacy cards (no worker_type) --
    //
    // A new frontend may see cards from old (pre-`worker_type`) workers that
    // carry no `worker_type`. Their disaggregated worker types can't be
    // reconstructed (old decode and old aggregated workers are
    // wire-indistinguishable), so a namespace containing any legacy card falls
    // back to legacy behavior: ready iff some worker is live. Strict worker-type
    // readiness gating resumes once every worker carries a worker_type.

    #[test]
    fn readiness_legacy_card_with_live_worker_is_ready() {
        // COMPAT: old aggregated worker (no worker_type) upgraded-frontend case.
        let model = Model::new("llama".to_string());
        let (ws, _tx) = make_worker_set_with_count("dynamo", "mdc-default", vec![1]);
        model.add_worker_set("dynamo".to_string(), ws);

        assert!(
            model.is_workers_ready("dynamo"),
            "a legacy card with a live worker must be ready under the compat shim"
        );
    }

    #[test]
    fn readiness_legacy_card_with_no_live_worker_is_not_ready() {
        // Even under compat, an empty namespace can't serve.
        let model = Model::new("llama".to_string());
        let (ws, _tx) = make_worker_set_with_count("dynamo", "mdc-default", vec![]);
        model.add_worker_set("dynamo".to_string(), ws);

        assert!(
            !model.is_workers_ready("dynamo"),
            "a legacy card with no live worker must not be ready"
        );
    }

    #[test]
    fn readiness_legacy_disagg_pd_is_ready() {
        // COMPAT: old disaggregated deployment (prefill + decode, both legacy
        // and cardless of worker_type) seen by a new frontend. The worker types
        // can't be reconstructed, so the namespace is ready as long as workers
        // are live — matching legacy behavior.
        let model = Model::new("llama".to_string());
        let (p, _tp) = make_worker_set_with_count("dynamo", "mdc-p", vec![1]);
        let (d, _td) = make_worker_set_with_count("dynamo", "mdc-d", vec![2]);
        model.add_worker_set("dynamo:prefill".to_string(), p);
        model.add_worker_set("dynamo".to_string(), d);

        assert!(
            model.is_workers_ready("dynamo"),
            "a legacy disagg namespace with live workers must be ready under compat"
        );
    }

    #[test]
    fn readiness_mixed_legacy_and_typed_uses_legacy_path() {
        // A namespace containing even one legacy card drops to the compat path
        // (ready iff live), rather than strict-gating on the typed cards.
        let model = Model::new("llama".to_string());
        // Typed decode that, under strict rules, needs a Prefill peer (absent).
        let (decode, _td) = ws_with_type(
            "dynamo",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        // Legacy card alongside it.
        let (legacy, _tl) = make_worker_set_with_count("dynamo", "mdc-legacy", vec![3]);
        model.add_worker_set("dynamo".to_string(), decode);
        model.add_worker_set("dynamo:legacy".to_string(), legacy);

        // Strict rules would say not-ready (decode's Prefill need unmet), but
        // the legacy card forces the compat path → ready because workers live.
        assert!(
            model.is_workers_ready("dynamo"),
            "a namespace with a legacy card must use the compat (live) path"
        );
    }

    // -- namespace_readiness() (GET /v1/models/{model}/ready) --

    #[test]
    fn readiness_detail_pd_pair_ready() {
        let model = Model::new("llama".to_string());
        let (p, _tp) = ws_with_type(
            "ns1",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (d, _td) = ws_with_type(
            "ns1",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("ns1:prefill".to_string(), p);
        model.add_worker_set("ns1".to_string(), d);

        let topo = model.namespace_readiness();
        assert!(topo.ready);
        assert_eq!(topo.reason, None);
        let ns = &topo.namespaces["ns1"];
        assert!(ns.ready);
        assert_eq!(
            ns.present,
            vec!["decode".to_string(), "prefill".to_string()]
        );
        assert!(ns.missing_worker_types.is_empty());
        assert_eq!(ns.worker_types["decode"].workers, 1);
        assert_eq!(ns.worker_types["prefill"].workers, 1);
        assert_eq!(
            ns.worker_types["decode"].needs,
            vec![vec!["prefill".to_string()]]
        );
        // Endpoint must agree with the gate.
        assert_eq!(ns.ready, model.is_workers_ready("ns1"));
    }

    #[test]
    fn readiness_detail_decode_only_reports_missing_prefill() {
        let model = Model::new("llama".to_string());
        let (d, _td) = ws_with_type(
            "ns1",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("ns1".to_string(), d);

        let topo = model.namespace_readiness();
        assert!(!topo.ready);
        assert_eq!(
            topo.reason.as_deref(),
            Some("no namespace has all required worker types live")
        );
        let ns = &topo.namespaces["ns1"];
        assert!(!ns.ready);
        assert_eq!(ns.present, vec!["decode".to_string()]);
        assert_eq!(ns.missing_worker_types, vec!["prefill".to_string()]);
        assert_eq!(ns.reason.as_deref(), Some("missing worker types: prefill"));
    }

    #[test]
    fn readiness_detail_zero_worker_type_reported_missing() {
        // A typed worker type registered with no live workers must surface as missing,
        // not produce an empty "missing worker types: " reason.
        let model = Model::new("llama".to_string());
        let (p, _tp) = ws_with_type(
            "ns1",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![], // registered, but zero live workers
        );
        model.add_worker_set("ns1".to_string(), p);

        let topo = model.namespace_readiness();
        assert!(!topo.ready);
        let ns = &topo.namespaces["ns1"];
        assert!(!ns.ready);
        assert!(ns.present.is_empty());
        assert_eq!(ns.missing_worker_types, vec!["prefill".to_string()]);
        assert_eq!(ns.reason.as_deref(), Some("missing worker types: prefill"));
    }

    #[test]
    fn readiness_detail_zero_worker_type_not_missing_when_present_elsewhere() {
        // Two WorkerSets of the same worker type in one namespace: one live, one with
        // zero workers. The worker type is present via the live set, so it must NOT be
        // reported missing; the real gap (its unsatisfied peer) must surface.
        let model = Model::new("llama".to_string());
        let (d_live, _tl) = ws_with_type(
            "ns1",
            "mdc-d-live",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![1, 2],
        );
        let (d_dead, _td) = ws_with_type(
            "ns1",
            "mdc-d-dead",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![], // zero live workers, same worker type
        );
        model.add_worker_set("ns1".to_string(), d_live);
        model.add_worker_set("ns1:dead".to_string(), d_dead);

        let topo = model.namespace_readiness();
        assert!(!topo.ready);
        let ns = &topo.namespaces["ns1"];
        assert!(!ns.ready);
        assert_eq!(ns.present, vec!["decode".to_string()]);
        // decode is present (live set) → not missing; prefill is the real gap.
        assert_eq!(ns.missing_worker_types, vec!["prefill".to_string()]);
        assert_eq!(ns.reason.as_deref(), Some("missing worker types: prefill"));
    }

    #[test]
    fn readiness_detail_multi_namespace_one_ready_one_partial() {
        let model = Model::new("llama".to_string());
        // ns-old: complete P/D pair.
        let (p, _tp) = ws_with_type(
            "ns-old",
            "mdc-p",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (d, _td) = ws_with_type(
            "ns-old",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        // ns-new: decode only (partial).
        let (d2, _td2) = ws_with_type(
            "ns-new",
            "mdc-d2",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![3],
        );
        model.add_worker_set("ns-old:prefill".to_string(), p);
        model.add_worker_set("ns-old".to_string(), d);
        model.add_worker_set("ns-new".to_string(), d2);

        let topo = model.namespace_readiness();
        // Model is ready overall because at least one namespace is complete.
        assert!(topo.ready);
        assert!(topo.namespaces["ns-old"].ready);
        assert!(!topo.namespaces["ns-new"].ready);
        assert_eq!(
            topo.namespaces["ns-new"].missing_worker_types,
            vec!["prefill".to_string()]
        );
    }

    #[test]
    fn readiness_detail_legacy_namespace_notes_bypass() {
        let model = Model::new("llama".to_string());
        let (legacy, _tl) = make_worker_set_with_count("ns1", "mdc-legacy", vec![1]);
        model.add_worker_set("ns1".to_string(), legacy);

        let topo = model.namespace_readiness();
        assert!(topo.ready);
        let ns = &topo.namespaces["ns1"];
        assert!(ns.ready);
        // Legacy card has no worker_type → not keyed under worker_types; reason flags
        // the compat bypass.
        assert!(ns.worker_types.is_empty());
        assert!(ns.missing_worker_types.is_empty());
        assert!(
            ns.reason
                .as_deref()
                .unwrap_or("")
                .contains("readiness gating bypassed"),
            "legacy namespace reason should note the bypass, got {:?}",
            ns.reason
        );
    }

    #[test]
    fn readiness_aggregated_zero_workers_not_ready() {
        // An Aggregated WorkerSet with `worker_count() == 0` must NOT be
        // considered ready: its `needs` is empty, but with no live worker
        // the namespace can't serve traffic.
        let model = Model::new("llama".to_string());
        let (agg, _tx) = ws_with_type(
            "dynamo",
            "mdc-a",
            WorkerType::Aggregated,
            vec![],
            vec![], // zero workers
        );
        model.add_worker_set("dynamo".to_string(), agg);

        assert!(
            !model.is_workers_ready("dynamo"),
            "an Aggregated worker set with zero live workers must NOT be ready"
        );
    }

    // -- is_ready_to_serve tests --
    //
    // Regression coverage for the KServe gRPC `model_ready` race: a
    // ModelDeploymentCard is saved before the WorkerSet's engines are wired up,
    // and `is_ready_to_serve` must remain false until at least one serving
    // engine is attached to a WorkerSet that has workers.

    #[test]
    fn test_is_ready_to_serve_false_when_no_worker_sets() {
        let model = Model::new("llama".to_string());
        assert!(!model.is_ready_to_serve());
    }

    #[test]
    fn test_is_ready_to_serve_false_for_prefill_only_set() {
        // A WorkerSet without any serving engine attached (the lifecycle state
        // between ModelDeploymentCard save and engine attach) must not count
        // as ready, even though `is_displayable` treats prefill-only as visible.
        let model = Model::new("llama".to_string());
        model.add_worker_set("ns1".to_string(), make_worker_set("ns1", "abc"));

        assert!(
            model.is_displayable(),
            "displayable fallback covers prefill"
        );
        assert!(
            !model.is_ready_to_serve(),
            "prefill-only set must not be ready to serve inference"
        );
    }

    #[test]
    fn test_is_ready_to_serve_false_when_zero_workers_even_with_engine() {
        // Engine attached but the discovery watcher reports zero connected
        // workers. KServe must report not-ready until a worker is available.
        let model = Model::new("llama".to_string());
        let mut ws = WorkerSet::new(
            "ns1".to_string(),
            "abc".to_string(),
            crate::model_card::ModelDeploymentCard::default(),
        );
        // Keep the sender bound for the duration of the test so the watcher
        // doesn't close.
        let (_tx, rx) = watch::channel::<Vec<u64>>(vec![]);
        ws.set_instance_watcher(rx);
        ws.chat_engine = Some(make_test_chat_engine());
        model.add_worker_set("ns1".to_string(), Arc::new(ws));

        assert!(
            !model.is_ready_to_serve(),
            "engine attached but no workers connected -> not ready"
        );
    }

    #[test]
    fn test_is_ready_to_serve_true_with_chat_engine() {
        // In-process WorkerSet (no instance_count_rx → worker_count==1) with a
        // chat engine attached is ready to serve.
        let model = Model::new("llama".to_string());
        let mut ws = WorkerSet::new(
            "ns1".to_string(),
            "abc".to_string(),
            crate::model_card::ModelDeploymentCard::default(),
        );
        ws.chat_engine = Some(make_test_chat_engine());
        model.add_worker_set("ns1".to_string(), Arc::new(ws));

        assert!(model.is_ready_to_serve());
    }

    /// Build a chat completions engine backed by the in-tree echo engine.
    fn make_test_chat_engine()
    -> crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine {
        Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        ))
    }

    /// Build a WorkerSet with an explicit worker_type / needs, a live worker
    /// count, AND a chat engine attached (i.e. it can be selected to serve a
    /// chat request).
    fn ws_serving_role(
        namespace: &str,
        mdcsum: &str,
        worker_type: WorkerType,
        needs: Vec<Vec<WorkerType>>,
        worker_ids: Vec<u64>,
    ) -> (Arc<WorkerSet>, watch::Sender<Vec<u64>>) {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(worker_type);
        card.needs = needs;
        let (tx, rx) = watch::channel(worker_ids);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        ws.set_instance_watcher(rx);
        ws.chat_engine = Some(make_test_chat_engine());
        (Arc::new(ws), tx)
    }

    /// A single deployment with an incomplete worker set (decode-only, no
    /// prefill peer) must not be selected for serving, even though it has a
    /// live worker and a chat engine attached. Confirms the fast-path
    /// readiness filter in `select_worker_set_with`.
    #[test]
    fn select_skips_unready_namespace_single_set() {
        let model = Model::new("llama".to_string());
        let (decode, _tx) = ws_serving_role(
            "bad",
            "mdc-d",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![1],
        );
        model.add_worker_set("bad".to_string(), decode);

        assert!(!model.is_workers_ready("bad"));
        assert!(
            model.get_chat_engine().is_err(),
            "decode-only namespace (missing prefill) must not be selectable"
        );
        // KServe readiness must agree with routability: an incomplete namespace
        // is not ready to serve even though it has a live worker + chat engine.
        assert!(
            !model.is_ready_to_serve(),
            "incomplete namespace must not report ready to serve"
        );
    }

    /// With two deployments of one model — a ready P/D pair and an unready
    /// decode-only namespace — selection must only ever land on the ready one.
    /// The decode WorkerSets in BOTH namespaces carry a chat engine and a live
    /// worker, so before the readiness filter the unready namespace could be
    /// picked. Removing the ready namespace leaves only the unready one, which
    /// must then be unservable rather than silently accepting (and failing) the
    /// request.
    #[test]
    fn select_skips_unready_namespace_multi_set() {
        let model = Model::new("llama".to_string());

        // Ready deployment: live prefill + live decode (decode is the front door).
        let (good_prefill, _tx_gp) = ws_with_type(
            "good",
            "mdc-gp",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (good_decode, _tx_gd) = ws_serving_role(
            "good",
            "mdc-gd",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        model.add_worker_set("good:prefill".to_string(), good_prefill);
        model.add_worker_set("good".to_string(), good_decode);

        // Unready deployment: live decode only, no prefill peer.
        let (bad_decode, _tx_bd) = ws_serving_role(
            "bad",
            "mdc-bd",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![3],
        );
        model.add_worker_set("bad".to_string(), bad_decode);

        assert!(model.is_workers_ready("good"));
        assert!(!model.is_workers_ready("bad"));

        // The ready namespace keeps the model servable.
        assert!(
            model.get_chat_engine().is_ok(),
            "a ready namespace must keep the model servable"
        );
        assert!(
            model.is_ready_to_serve(),
            "a ready namespace must report ready to serve"
        );

        // Drop the ready namespace; only the unready decode-only namespace is
        // left. It must NOT be selectable despite having a live worker + engine.
        model.remove_worker_set("good:prefill");
        model.remove_worker_set("good");
        assert!(
            model.get_chat_engine().is_err(),
            "unready namespace must never be selected for serving"
        );
        assert!(
            !model.is_ready_to_serve(),
            "only an incomplete namespace remains: not ready to serve"
        );
    }

    // -- Selected<T> namespace-attribution tests --
    //
    // The namespace on `Selected` is what every per-request metric is labelled
    // with, so it must name the WorkerSet the engine was actually taken from.
    // `select_worker_set_with` has three distinct return paths (single-set fast
    // path, `eligible.len() == 1`, weighted random) and each is covered here.

    /// A legacy-card WorkerSet (no worker_type) with a chat engine: always
    /// ready, worker_count == 1, and selectable.
    fn ws_chat(namespace: &str, mdcsum: &str) -> Arc<WorkerSet> {
        let mut ws = WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        );
        ws.chat_engine = Some(make_test_chat_engine());
        Arc::new(ws)
    }

    #[test]
    fn selected_namespace_matches_set_on_fast_path() {
        let model = Model::new("llama".to_string());
        let alpha = ws_chat("alpha", "abc");
        model.add_worker_set("alpha".to_string(), alpha.clone());

        let selected = model.get_chat_engine().expect("engine available");
        assert_eq!(selected.namespace, "alpha");
        assert!(
            Arc::ptr_eq(&selected.value, alpha.chat_engine.as_ref().unwrap()),
            "engine must come from the namespace it was attributed to"
        );
    }

    #[test]
    fn selected_namespace_matches_set_when_only_one_is_eligible() {
        // Two namespaces with live workers, but only one carries a chat engine.
        // Exercises the `eligible.len() == 1` path: the namespace must be the
        // one that HAD the engine, not the first set in the map.
        let model = Model::new("llama".to_string());
        model.add_worker_set("alpha".to_string(), make_worker_set("alpha", "abc"));
        let beta = ws_chat("beta", "abc");
        model.add_worker_set("beta".to_string(), beta.clone());

        for _ in 0..64 {
            let selected = model.get_chat_engine().expect("engine available");
            assert_eq!(
                selected.namespace, "beta",
                "only 'beta' has a chat engine; attribution must not name 'alpha'"
            );
            assert!(Arc::ptr_eq(
                &selected.value,
                beta.chat_engine.as_ref().unwrap()
            ));
        }
    }

    #[test]
    fn selected_namespace_matches_set_on_weighted_path() {
        // Three eligible namespaces -> weighted random selection. Whichever is
        // picked, the reported namespace must identify the set the engine came
        // from, and over many draws every namespace must be reachable (i.e. the
        // namespace is not pinned to one set).
        let model = Model::new("llama".to_string());
        let sets: Vec<Arc<WorkerSet>> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|ns| {
                let ws = ws_chat(ns, "abc");
                model.add_worker_set(ns.to_string(), ws.clone());
                ws
            })
            .collect();

        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let selected = model.get_chat_engine().expect("engine available");
            let owner = sets
                .iter()
                .find(|ws| Arc::ptr_eq(&selected.value, ws.chat_engine.as_ref().unwrap()))
                .expect("engine must belong to one of the registered sets");
            assert_eq!(
                selected.namespace,
                owner.namespace(),
                "reported namespace must match the set the engine came from"
            );
            seen.insert(selected.namespace);
        }
        assert_eq!(
            seen.len(),
            3,
            "weighted selection should reach every eligible namespace; saw {seen:?}"
        );
    }

    #[test]
    fn selected_namespace_is_correct_with_unequal_worker_counts() {
        // Weighted-by-worker-count path with very different weights: the heavy
        // namespace dominates, but attribution must still be per-draw exact.
        let model = Model::new("llama".to_string());
        let (light, _tx_l) = ws_serving_role(
            "light",
            "mdc-l",
            WorkerType::Aggregated,
            vec![],
            (0..1).collect(),
        );
        let (heavy, _tx_h) = ws_serving_role(
            "heavy",
            "mdc-h",
            WorkerType::Aggregated,
            vec![],
            (0..50).collect(),
        );
        model.add_worker_set("light".to_string(), light.clone());
        model.add_worker_set("heavy".to_string(), heavy.clone());

        for _ in 0..500 {
            let selected = model.get_chat_engine().expect("engine available");
            let expected = if Arc::ptr_eq(&selected.value, heavy.chat_engine.as_ref().unwrap()) {
                "heavy"
            } else {
                assert!(Arc::ptr_eq(
                    &selected.value,
                    light.chat_engine.as_ref().unwrap()
                ));
                "light"
            };
            assert_eq!(selected.namespace, expected);
        }
    }

    #[test]
    fn selected_namespace_is_carried_through_parsing_accessor() {
        // `get_chat_engine_with_parsing` extracts a tuple; the namespace must
        // still be the tuple's own set.
        let model = Model::new("llama".to_string());
        let beta = ws_chat("beta", "abc");
        model.add_worker_set("alpha".to_string(), make_worker_set("alpha", "abc"));
        model.add_worker_set("beta".to_string(), beta.clone());

        let selected = model
            .get_chat_engine_with_parsing()
            .expect("engine available");
        assert_eq!(selected.namespace, "beta");
        assert!(Arc::ptr_eq(
            &selected.value.0,
            beta.chat_engine.as_ref().unwrap()
        ));
    }

    #[test]
    fn selected_map_preserves_namespace() {
        let selected = Selected {
            value: 7u32,
            namespace: "ns-a".to_string(),
        };
        let mapped = selected.map(|v| v * 2);
        assert_eq!(mapped.value, 14);
        assert_eq!(mapped.namespace, "ns-a");
    }

    #[test]
    fn selected_namespace_never_leaks_an_unready_namespace() {
        // "bad" is decode-only (unready) but has a live worker and a chat
        // engine; "good" is a complete P/D pair. Every selection must both come
        // from and be labelled "good".
        let model = Model::new("llama".to_string());
        let (good_prefill, _tx_gp) = ws_with_type(
            "good",
            "mdc-gp",
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
            vec![1],
        );
        let (good_decode, _tx_gd) = ws_serving_role(
            "good",
            "mdc-gd",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![2],
        );
        let (bad_decode, _tx_bd) = ws_serving_role(
            "bad",
            "mdc-bd",
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
            vec![3],
        );
        model.add_worker_set("good:prefill".to_string(), good_prefill);
        model.add_worker_set("good".to_string(), good_decode.clone());
        model.add_worker_set("bad".to_string(), bad_decode.clone());

        for _ in 0..200 {
            let selected = model.get_chat_engine().expect("good namespace serves");
            assert_eq!(selected.namespace, "good");
            assert!(
                Arc::ptr_eq(&selected.value, good_decode.chat_engine.as_ref().unwrap()),
                "must not hand back the unready namespace's engine"
            );
        }
    }
}
