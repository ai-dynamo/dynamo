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
use super::worker_monitor::LoadThresholdConfig;
use super::worker_set::WorkerSet;
use crate::protocols::openai::ParsingOptions;

use crate::types::{
    RealtimeBidirectionalEngine,
    generic::tensor::TensorStreamingEngine,
    inference::generate::GenerateStreamingEngine,
    openai::{
        audios::OpenAIAudiosStreamingEngine,
        chat_completions::OpenAIChatCompletionsStreamingEngine,
        completions::OpenAICompletionsStreamingEngine, embeddings::OpenAIEmbeddingsStreamingEngine,
        images::OpenAIImagesStreamingEngine, videos::OpenAIVideosStreamingEngine,
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
}

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

    /// Check if any WorkerSet has an engine-native generate engine.
    pub fn has_generate_engine(&self) -> bool {
        self.worker_sets
            .iter()
            .any(|entry| entry.value().has_generate_engine())
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
    /// every worker in the namespace carries a `worker_type`. Remove this branch
    /// when the compat shim is retired.
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
        let mut present: std::collections::HashSet<crate::worker_type::WorkerType> =
            std::collections::HashSet::new();
        let mut missing: std::collections::HashSet<crate::worker_type::WorkerType> =
            std::collections::HashSet::new();
        let mut has_legacy = false;
        let mut legacy_live_workers = 0usize;
        let mut has_live_worker = false;

        // First pass: which worker types have a live worker (+ legacy detection).
        for ws in wsets {
            let count = ws.worker_count();
            if count > 0 {
                has_live_worker = true;
            }
            match Self::ws_type_and_needs(ws) {
                Some((wt, _needs)) => {
                    if count > 0 {
                        present.insert(wt);
                    }
                }
                // No declared worker_type → legacy card.
                None => {
                    has_legacy = true;
                    legacy_live_workers += count;
                }
            }
        }

        // COMPAT branch: a legacy card disables strict gating; the disaggregated
        // worker types can't be reconstructed, so ready iff any worker is live.
        if has_legacy {
            warn_legacy_readiness_once(&self.name, wsets[0].namespace());
            return NamespaceReadinessEval {
                ready: has_live_worker,
                has_legacy,
                legacy_live_workers,
                present,
                missing,
            };
        }

        // Strict path: a registered worker type with no live worker anywhere is
        // missing; a *live* WorkerSet whose `needs` DNF is unsatisfied flags its
        // absent peers.
        for ws in wsets {
            let Some((wt, needs)) = Self::ws_type_and_needs(ws) else {
                continue;
            };
            if !present.contains(&wt) {
                missing.insert(wt);
            }
            if ws.worker_count() == 0 || needs.is_empty() {
                continue;
            }
            let satisfied = needs
                .iter()
                .any(|alt| alt.iter().all(|t| present.contains(t)));
            if !satisfied {
                for alt in &needs {
                    for t in alt {
                        if !present.contains(t) {
                            missing.insert(*t);
                        }
                    }
                }
            }
        }

        NamespaceReadinessEval {
            ready: has_live_worker && missing.is_empty(),
            has_legacy,
            legacy_live_workers,
            present,
            missing,
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
    /// Built on the same [`Self::evaluate_namespace`] facts the serving gate
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
    /// Delegates to [`Self::select_worker_set_with`] so readiness reports
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
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.chat_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_chat_engine()))
    }

    pub fn get_completions_engine(
        &self,
    ) -> Result<OpenAICompletionsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.completions_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_completions_engine()))
    }

    pub fn get_embeddings_engine(
        &self,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.embeddings_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_embeddings_engine()))
    }

    pub fn get_images_engine(&self) -> Result<OpenAIImagesStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.images_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_images_engine()))
    }

    pub fn get_videos_engine(&self) -> Result<OpenAIVideosStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.videos_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_videos_engine()))
    }

    pub fn get_audios_engine(&self) -> Result<OpenAIAudiosStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.audios_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_audios_engine()))
    }

    pub fn get_tensor_engine(&self) -> Result<TensorStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.tensor_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_tensor_engine()))
    }

    pub fn get_realtime_engine(&self) -> Result<RealtimeBidirectionalEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| ws.realtime_engine.clone())
            .ok_or_else(|| self.engine_error(self.has_realtime_engine()))
    }

    pub fn get_generate_engine(&self) -> Result<GenerateStreamingEngine, ModelManagerError> {
        self.select_worker_set_with(|ws| {
            ws.has_available_generate_engine()
                .then(|| ws.generate_engine.clone())
                .flatten()
        })
        .ok_or_else(|| self.engine_error(self.has_generate_engine()))
    }

    // -- Combined engine + parsing options (atomically from one WorkerSet) --

    pub fn get_chat_engine_with_parsing(
        &self,
    ) -> Result<(OpenAIChatCompletionsStreamingEngine, ParsingOptions), ModelManagerError> {
        self.select_worker_set_with(|ws| ws.chat_engine.clone().map(|e| (e, ws.parsing_options())))
            .ok_or_else(|| self.engine_error(self.has_chat_engine()))
    }

    pub fn get_completions_engine_with_parsing(
        &self,
    ) -> Result<(OpenAICompletionsStreamingEngine, ParsingOptions), ModelManagerError> {
        self.select_worker_set_with(|ws| {
            ws.completions_engine
                .clone()
                .map(|e| (e, ws.parsing_options()))
        })
        .ok_or_else(|| self.engine_error(self.has_completions_engine()))
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
            if let Some(ref monitor) = entry.value().engine_generate_worker_monitor {
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

    pub(crate) fn refresh_prefill_routers_for_namespace(
        &self,
        namespace: &str,
        engine_generate: bool,
        endpoint: dynamo_runtime::component::Endpoint,
    ) -> bool {
        let mut refreshed = false;
        for entry in self.worker_sets.iter() {
            let worker_set = entry.value();
            if worker_set.namespace() != namespace {
                continue;
            }
            let router = if engine_generate {
                worker_set.engine_generate_prefill_router.as_ref()
            } else {
                worker_set.prefill_router.as_ref()
            };
            if let Some(router) = router {
                refreshed |= router.refresh_activation_endpoint(endpoint.clone());
            }
        }
        refreshed
    }

    pub(crate) fn deactivate_prefill_routers_for_namespace(&self, namespace: &str) {
        for entry in self.worker_sets.iter() {
            let worker_set = entry.value();
            if worker_set.namespace() != namespace {
                continue;
            }
            if let Some(router) = &worker_set.prefill_router {
                router.deactivate();
            }
            if let Some(router) = &worker_set.engine_generate_prefill_router {
                router.deactivate();
            }
        }
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
    fn select_worker_set_with<T, F>(&self, extract: F) -> Option<T>
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

        // Fast path: single set (same zero-worker filtering as the multi-set path below)
        if snapshot.len() == 1 {
            let ws = &snapshot[0];
            if ws.worker_count() == 0 || !ready_namespaces.contains(ws.namespace()) {
                return None;
            }
            return extract(ws);
        }

        // Collect eligible sets with their worker counts, skipping sets with no workers or sets in
        // a namespace whose worker set is incomplete.
        // In-process models (no discovery watcher) return count=1, so they always participate.
        // Discovery models with count=0 have no available workers and are skipped.
        let eligible: Vec<(T, usize)> = snapshot
            .iter()
            .filter_map(|ws| {
                let count = ws.worker_count();
                if count == 0 || !ready_namespaces.contains(ws.namespace()) {
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
        pr.mark_active_for_test();
        pr.deactivate();
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
}
