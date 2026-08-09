// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local Dynamo fleet.
//!
//! The select/reserve split is what makes cross-provider routing possible at
//! all: `select` prices a worker *without* booking load, so we can compare that
//! price against a frontier model and walk away if the frontier wins. An
//! unclaimed selection simply expires (120s by default).
//!
//! The reservation lifecycle is not optional. A booked reservation that never
//! sees `prefill_complete` and `release` leaves the router's load accounting
//! permanently wrong, so [`Reservation`] logs loudly if it is dropped without
//! being settled.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::services::selection::{
    PromptRequest, ReservationRequest, SelectRequest, SelectionService, WorkerRequest,
};
use roundhouse_core::routing::{Candidate, Target};

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("no worker available for model `{0}`")]
    NoWorker(String),
    #[error("selection service rejected the request: {0}")]
    Rejected(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// What the router needs in order to price a turn.
///
/// Note there are no token ids: the selection service accepts precomputed
/// sequence hashes, so the incremental buffer in `roundhouse-core` is all the
/// input required. For a 100k-token context that is the difference between
/// shipping a 400 KB array and a few kilobytes of hashes.
#[derive(Debug, Clone)]
pub struct FleetQuery {
    pub model_name: String,
    pub routing_group: String,
    /// Per-block hashes. Required alongside `sequence_hashes` when no token ids
    /// are sent; the service rebuilds its own view of the chain from them.
    pub block_hashes: Vec<u64>,
    /// Rolling sequence hashes, the form prefix matching happens on.
    pub sequence_hashes: Vec<u64>,
    pub isl_tokens: usize,
    pub expected_output_tokens: Option<u32>,
    /// Propagated so the router can apply its own session affinity underneath
    /// ours.
    pub session_id: Option<String>,
}

impl FleetQuery {
    /// Build a query-only select request under a caller-minted id.
    ///
    /// The service only caches booking inputs when `select` carries a
    /// `selection_id`, and reservation ids must be globally unique -- a
    /// collision across processes would let one session book against another's
    /// captured prompt -- so this is a fresh UUID per quote.
    fn to_select_request(&self, selection_id: String) -> SelectRequest {
        SelectRequest {
            model_name: self.model_name.clone(),
            routing_group: self.routing_group.clone(),
            selection_id: Some(selection_id),
            prompt: PromptRequest {
                token_ids: None,
                mm_routing_info: None,
                block_mm_infos: None,
                // `as i64` is a bit reinterpretation; the service converts back
                // with `as u64`, so values round-trip exactly even above
                // `i64::MAX`.
                block_hashes: Some(self.block_hashes.iter().map(|hash| *hash as i64).collect()),
                sequence_hashes: Some(
                    self.sequence_hashes
                        .iter()
                        .map(|hash| *hash as i64)
                        .collect(),
                ),
                isl_tokens: Some(self.isl_tokens),
                lora_name: None,
                cache_namespace: None,
                is_eagle: None,
            },
            router_config_override: None,
            expected_output_tokens: self.expected_output_tokens,
            priority_jump: None,
            strict_priority: None,
            session_id: self.session_id.clone(),
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: Default::default(),
        }
    }
}

/// A priced, unbooked local option.
#[derive(Debug, Clone)]
pub struct LocalQuote {
    pub selection_id: String,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub endpoint: String,
    pub model_name: String,
    /// The scheduler's cache-credit-weighted prefill cost. This is the
    /// authoritative local number, and the axis frontier candidates are
    /// projected onto.
    pub effective_prefill_tokens: usize,
    /// Raw matched tokens across cache tiers. Observability only — deriving
    /// decisions from this rather than `effective_prefill_tokens` would ignore
    /// the scheduler's own weighting.
    pub longest_matched_tokens: u32,
    pub isl_tokens: usize,
    pub load: Option<f64>,
}

impl LocalQuote {
    pub fn target(&self) -> Target {
        Target::Local {
            worker_id: self.worker_id,
            dp_rank: self.dp_rank,
            model: self.model_name.clone(),
        }
    }

    /// Project onto the common comparison axis.
    ///
    /// Local execution is priced at zero dollars: its cost is capacity, already
    /// captured by `expected_prefill_tokens` and `load`. Mixing an amortized
    /// GPU cost in here would double-count.
    pub fn to_candidate(&self, quality_prior: f64, ttft_ms: f64) -> Candidate {
        Candidate {
            target: self.target(),
            expected_prefill_tokens: self.effective_prefill_tokens as f64,
            matched_prefix_tokens: self.longest_matched_tokens as u64,
            expected_ttft_ms: ttft_ms,
            expected_cost_usd: 0.0,
            quality_prior,
            load: self.load,
        }
    }
}

/// A booked reservation. Settle it with [`Reservation::release`].
pub struct Reservation {
    selection_id: String,
    fleet: Arc<dyn LocalFleet>,
    settled: bool,
}

impl Reservation {
    pub fn selection_id(&self) -> &str {
        &self.selection_id
    }

    pub async fn prefill_complete(&self) -> Result<(), FleetError> {
        self.fleet.prefill_complete(&self.selection_id).await
    }

    pub async fn output_block(&self) -> Result<(), FleetError> {
        self.fleet.output_block(&self.selection_id).await
    }

    pub async fn release(mut self) -> Result<(), FleetError> {
        self.settled = true;
        self.fleet.release(&self.selection_id).await
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.settled {
            // Can't await here, so this cannot self-heal. A leak permanently
            // inflates the router's view of this worker's load, which then
            // silently distorts every later decision -- worth shouting about.
            tracing::error!(
                selection_id = %self.selection_id,
                "reservation dropped without release; router load accounting will drift"
            );
        }
    }
}

#[async_trait]
pub trait LocalFleet: Send + Sync + 'static {
    /// Price the best local worker without booking load.
    ///
    /// `None` means the fleet has nothing schedulable, which is a routing
    /// input rather than an error: the frontier path may still be viable.
    async fn price(&self, query: &FleetQuery) -> Result<Option<LocalQuote>, FleetError>;

    /// Book a quote previously returned by [`LocalFleet::price`].
    async fn reserve(self: Arc<Self>, quote: &LocalQuote) -> Result<Reservation, FleetError>;

    async fn prefill_complete(&self, selection_id: &str) -> Result<(), FleetError>;
    async fn output_block(&self, selection_id: &str) -> Result<(), FleetError>;
    async fn release(&self, selection_id: &str) -> Result<(), FleetError>;
}

/// Worker catalog entry.
///
/// Every selector replica needs the same catalog before it serves traffic —
/// replica sync never creates workers — so this is registered per instance.
#[derive(Debug, Clone)]
pub struct WorkerRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub routing_group: String,
    pub endpoint: String,
    pub block_size: u32,
    /// Per-DP-rank ZMQ endpoints the indexer subscribes to for KV events.
    pub kv_events_endpoints: HashMap<u32, String>,
}

/// [`LocalFleet`] backed by an in-process Dynamo `SelectionService`.
pub struct EmbeddedFleet {
    service: Arc<SelectionService>,
    routing_group: String,
}

impl EmbeddedFleet {
    pub fn new(service: Arc<SelectionService>) -> Self {
        Self {
            service,
            routing_group: "default".to_string(),
        }
    }

    pub fn with_routing_group(mut self, routing_group: impl Into<String>) -> Self {
        self.routing_group = routing_group.into();
        self
    }

    pub fn service(&self) -> &SelectionService {
        &self.service
    }

    /// Register a worker with the embedded catalog.
    pub async fn register_worker(&self, worker: WorkerRegistration) -> Result<(), FleetError> {
        let request = WorkerRequest {
            worker_id: worker.worker_id,
            model_name: worker.model_name,
            routing_group: worker.routing_group,
            endpoint: Some(worker.endpoint),
            kv_events_endpoint: None,
            kv_events_endpoints: worker.kv_events_endpoints,
            replay_endpoint: None,
            block_size: Some(worker.block_size),
            data_parallel_start_rank: None,
            data_parallel_size: None,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
            stable_routing_id: None,
            is_eagle: None,
            taints: Default::default(),
            topology_domains: Default::default(),
            kv_transfer_domain: None,
            kv_transfer_enforcement: None,
            kv_transfer_preferred_weight: None,
        };
        self.service
            .upsert_worker(request)
            .await
            .map_err(|error| FleetError::Rejected(error.to_string()))?;
        Ok(())
    }

    /// Current pressure on a worker, or `None` if it has no load entry yet.
    ///
    /// Uses potential prefill tokens rather than request count: a single
    /// long-context request loads a worker far more than several short ones,
    /// and prefill tokens are the same unit the rest of the decision is in.
    async fn load_for(&self, model_name: &str, worker_id: u64) -> Option<f64> {
        self.service
            .loads(Some(model_name), Some(&self.routing_group))
            .into_iter()
            .flat_map(|model| model.loads)
            .find(|load| load.worker_id == worker_id)
            .map(|load| load.potential_prefill_tokens as f64)
    }
}

#[async_trait]
impl LocalFleet for EmbeddedFleet {
    async fn price(&self, query: &FleetQuery) -> Result<Option<LocalQuote>, FleetError> {
        // Query-only. Nothing is booked, so abandoning this quote in favour of
        // a frontier target costs only a pending-cache entry that expires.
        let selection_id = format!("rh_{}", uuid::Uuid::new_v4().simple());
        let response = match self
            .service
            .select(query.to_select_request(selection_id))
            .await
        {
            Ok(response) => response,
            // Only "nothing schedulable" and "not warmed up yet" are routing
            // inputs. Everything else is a real fault and must surface --
            // collapsing all errors to `None` would silently route every turn
            // to the frontier the moment the local plane misbehaved.
            Err(error) if matches!(error.kind(), "not_found" | "not_ready") => {
                tracing::debug!(
                    kind = error.kind(),
                    %error,
                    model = %query.model_name,
                    "local fleet has nothing schedulable"
                );
                return Ok(None);
            }
            Err(error) => {
                return Err(FleetError::Rejected(format!(
                    "select failed ({}): {error}",
                    error.kind()
                )));
            }
        };

        let load = self.load_for(&query.model_name, response.worker_id).await;
        Ok(Some(LocalQuote {
            selection_id: response
                .selection_id
                .ok_or_else(|| FleetError::Rejected("select returned no selection_id".into()))?,
            worker_id: response.worker_id,
            dp_rank: response.dp_rank,
            endpoint: response.endpoint,
            model_name: response.model_name,
            effective_prefill_tokens: response.effective_prefill_tokens,
            longest_matched_tokens: response.overlap.longest_matched,
            isl_tokens: response.isl_tokens.unwrap_or(query.isl_tokens),
            load,
        }))
    }

    async fn reserve(self: Arc<Self>, quote: &LocalQuote) -> Result<Reservation, FleetError> {
        // Minimal replay form: the service still holds the inputs `select`
        // captured under this id, so the prompt is not resent. That cache is
        // replica-local, which is free for us because one process does both
        // halves.
        let request = ReservationRequest {
            selection_id: quote.selection_id.clone(),
            model_name: quote.model_name.clone(),
            routing_group: self.routing_group.clone(),
            // Leaving `worker_id` unset selects the replay form. Supplying it
            // would switch to the explicit form, which ignores the cached
            // selection and re-derives the booking from the prompt.
            worker_id: None,
            dp_rank: None,
            prompt: PromptRequest {
                token_ids: None,
                mm_routing_info: None,
                block_mm_infos: None,
                block_hashes: None,
                sequence_hashes: None,
                isl_tokens: None,
                lora_name: None,
                cache_namespace: None,
                is_eagle: None,
            },
            router_config_override: None,
            expected_output_tokens: None,
            effective_prefill_tokens: None,
            track_prefill_tokens: None,
        };
        self.service
            .create_reservation(request)
            .await
            .map_err(|error| FleetError::Rejected(error.to_string()))?;

        Ok(Reservation {
            selection_id: quote.selection_id.clone(),
            fleet: self,
            settled: false,
        })
    }

    async fn prefill_complete(&self, selection_id: &str) -> Result<(), FleetError> {
        self.service
            .prefill_complete(selection_id)
            .await
            .map_err(|error| FleetError::Rejected(error.to_string()))
    }

    async fn output_block(&self, selection_id: &str) -> Result<(), FleetError> {
        // Sync on purpose in the upstream API: this fires per decode block and
        // must not await on the hot path.
        self.service
            .add_output_block(selection_id, None)
            .map_err(|error| FleetError::Rejected(error.to_string()))
    }

    async fn release(&self, selection_id: &str) -> Result<(), FleetError> {
        self.service
            .free_reservation(selection_id)
            .await
            .map_err(|error| FleetError::Rejected(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_carries_hashes_rather_than_tokens() {
        let query = FleetQuery {
            model_name: "llama".into(),
            routing_group: "default".into(),
            block_hashes: vec![10, 20, 30],
            sequence_hashes: vec![1, 2, u64::MAX],
            isl_tokens: 4096,
            expected_output_tokens: Some(256),
            session_id: Some("sess_x".into()),
        };
        let request = query.to_select_request("sel_1".into());

        assert_eq!(request.selection_id.as_deref(), Some("sel_1"));
        assert!(
            request.prompt.token_ids.is_none(),
            "the token array must never be sent; hashes are sufficient"
        );
        assert_eq!(request.prompt.isl_tokens, Some(4096));

        // u64 -> i64 -> u64 must be lossless, including above i64::MAX.
        let sent = request.prompt.sequence_hashes.unwrap();
        let restored: Vec<u64> = sent.iter().map(|hash| *hash as u64).collect();
        assert_eq!(restored, vec![1, 2, u64::MAX]);
    }

    #[test]
    fn a_quote_projects_onto_the_shared_cost_axis() {
        let quote = LocalQuote {
            selection_id: "s1".into(),
            worker_id: 7,
            dp_rank: 0,
            endpoint: "http://w7:8000".into(),
            model_name: "llama".into(),
            effective_prefill_tokens: 512,
            longest_matched_tokens: 3_584,
            isl_tokens: 4_096,
            load: Some(0.25),
        };

        let candidate = quote.to_candidate(0.6, 90.0);
        assert_eq!(candidate.expected_prefill_tokens, 512.0);
        assert_eq!(
            candidate.expected_cost_usd, 0.0,
            "local capacity is not priced in dollars"
        );
        assert_eq!(candidate.load, Some(0.25));
        assert!(candidate.cache_hit_ratio(4_096) > 0.87);
        assert_eq!(
            candidate.target,
            Target::Local {
                worker_id: 7,
                dp_rank: 0,
                model: "llama".into()
            }
        );
    }
}
