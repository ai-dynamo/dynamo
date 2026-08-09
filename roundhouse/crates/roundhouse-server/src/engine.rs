// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turn execution.

use std::sync::Arc;

use async_trait::async_trait;
use roundhouse_core::context::{ContextAssembler, Tokenizer};
use roundhouse_core::event::Usage;
use roundhouse_core::ids::{ResponseId, SessionId, TurnId};
use roundhouse_core::item::Item;
use roundhouse_core::now_ms;
use roundhouse_core::routing::{
    CacheLedger, Candidate, Decision, DecisionRecord, RoutingContext, RoutingError, RoutingPolicy,
    Target,
};
use roundhouse_core::session::{Session, SessionError, TurnAdmission};
use roundhouse_core::store::SessionStore;
use roundhouse_fleet::{
    FleetError, FleetQuery, FrontierChunk, FrontierClient, FrontierError, FrontierQuote,
    LocalFleet, StaticFrontierCatalog,
};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Routing(#[from] RoutingError),
    #[error(transparent)]
    Fleet(#[from] FleetError),
    #[error(transparent)]
    Frontier(#[from] FrontierError),
    #[error("chosen target `{0:?}` had no matching quote")]
    UnresolvableTarget(Target),
}

/// What a local worker produced.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalExecution {
    pub text: String,
    pub output_tokens: u64,
}

/// Dispatches a prompt to a chosen local worker.
///
/// Separate from [`LocalFleet`], which only decides *where* a turn should go.
/// Splitting them keeps the routing decision testable without a worker to talk
/// to, and lets the real implementation be swapped for a mock in tests.
#[async_trait]
pub trait LocalExecutor: Send + Sync + 'static {
    async fn execute(
        &self,
        endpoint: &str,
        prompt: &str,
        expected_output_tokens: Option<u32>,
    ) -> Result<LocalExecution, FleetError>;
}

/// Deterministic [`LocalExecutor`] for tests and offline runs.
pub struct EchoLocalExecutor {
    reply: String,
}

impl EchoLocalExecutor {
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

#[async_trait]
impl LocalExecutor for EchoLocalExecutor {
    async fn execute(
        &self,
        _endpoint: &str,
        _prompt: &str,
        _expected_output_tokens: Option<u32>,
    ) -> Result<LocalExecution, FleetError> {
        Ok(LocalExecution {
            text: self.reply.clone(),
            output_tokens: self.reply.len() as u64,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub node_id: String,
    pub lease_ttl_ms: u64,
    pub block_size: u32,
    pub local_model: String,
    pub routing_group: String,
    /// Capability of the local model relative to the frontier catalog.
    pub local_quality_prior: f64,
    /// Latency floor attributed to a local worker before prefill.
    pub local_base_ttft_ms: f64,
    pub expected_output_tokens: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            node_id: "node-0".to_string(),
            lease_ttl_ms: 10_000,
            block_size: 16,
            local_model: "local".to_string(),
            routing_group: "default".to_string(),
            local_quality_prior: 0.6,
            local_base_ttft_ms: 60.0,
            expected_output_tokens: 256,
        }
    }
}

/// Outcome of one turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    pub response_id: ResponseId,
    pub text: String,
    /// `None` when the turn was deduplicated and no routing happened.
    pub decision: Option<Decision>,
    pub usage: Usage,
    /// Sequence number after the turn; the cursor a client resumes from.
    pub last_seq: u64,
    pub deduplicated: bool,
}

pub struct Engine<S: SessionStore, T: Tokenizer + Clone> {
    store: Arc<S>,
    tokenizer: T,
    fleet: Option<Arc<dyn LocalFleet>>,
    local_executor: Arc<dyn LocalExecutor>,
    frontier_catalog: StaticFrontierCatalog,
    frontier_client: Arc<dyn FrontierClient>,
    policy: Arc<dyn RoutingPolicy>,
    config: EngineConfig,
}

impl<S: SessionStore, T: Tokenizer + Clone> Engine<S, T> {
    pub fn new(
        store: Arc<S>,
        tokenizer: T,
        local_executor: Arc<dyn LocalExecutor>,
        frontier_catalog: StaticFrontierCatalog,
        frontier_client: Arc<dyn FrontierClient>,
        policy: Arc<dyn RoutingPolicy>,
        config: EngineConfig,
    ) -> Self {
        Self {
            store,
            tokenizer,
            fleet: None,
            local_executor,
            frontier_catalog,
            frontier_client,
            policy,
            config,
        }
    }

    pub fn with_fleet(mut self, fleet: Arc<dyn LocalFleet>) -> Self {
        self.fleet = Some(fleet);
        self
    }

    pub async fn create_session(&self, session_id: &SessionId) -> Result<bool, EngineError> {
        Ok(self
            .store
            .create_session(session_id, self.policy.name())
            .await
            .map_err(SessionError::from)?)
    }

    /// A ledger seeded with the frontier catalog's cache models and pricing.
    ///
    /// Seeded before replay so historical dispatches are interpreted under the
    /// right TTL rather than defaulting to "no cache".
    fn seeded_ledger(&self) -> CacheLedger {
        let mut ledger = CacheLedger::new();
        self.frontier_catalog.apply_to_ledger(&mut ledger);
        ledger
    }

    /// Run one turn to completion.
    pub async fn run_turn(
        &self,
        session_id: &SessionId,
        turn_id: TurnId,
        input: Vec<Item>,
    ) -> Result<TurnResult, EngineError> {
        let mut session = Session::open(
            Arc::clone(&self.store),
            session_id.clone(),
            &self.config.node_id,
            self.config.lease_ttl_ms,
            self.seeded_ledger(),
        )
        .await?;

        let admission = session.begin_turn(turn_id, input).await?;
        let response_id = admission.response_id().clone();
        if let TurnAdmission::Deduplicated(_) = admission {
            // A retry of a completed turn. Replay rather than regenerate: the
            // client already paid for this answer.
            let text = replay_output(&session, &response_id).await?;
            let last_seq = session.last_seq();
            // Release on this path too. Returning while still holding the lease
            // would lock the session out until the TTL lapsed, turning a
            // harmless client retry into a stall.
            session.release().await?;
            return Ok(TurnResult {
                response_id,
                text,
                decision: None,
                usage: Usage::default(),
                last_seq,
                deduplicated: true,
            });
        }

        // Rebuild the prompt from the committed log, so what we price is
        // exactly what a successor would reconstruct.
        let assembler = ContextAssembler::rehydrate(
            self.tokenizer.clone(),
            self.config.block_size,
            session.state().items.clone(),
        );
        let isl_tokens = assembler.buffer().isl_tokens();
        let turn_index = session.turn_index().saturating_sub(1);

        // --- price every option -------------------------------------------
        let local_quote = match &self.fleet {
            Some(fleet) => {
                fleet
                    .price(&FleetQuery {
                        model_name: self.config.local_model.clone(),
                        routing_group: self.config.routing_group.clone(),
                        block_hashes: assembler
                            .buffer()
                            .block_hashes()
                            .iter()
                            .map(|hash| hash.0)
                            .collect(),
                        sequence_hashes: assembler.buffer().sequence_hashes().to_vec(),
                        isl_tokens,
                        expected_output_tokens: Some(self.config.expected_output_tokens),
                        session_id: Some(session_id.to_string()),
                    })
                    .await?
            }
            None => None,
        };

        let mut candidates: Vec<Candidate> = Vec::new();
        if let Some(quote) = &local_quote {
            candidates.push(quote.to_candidate(
                self.config.local_quality_prior,
                self.config.local_base_ttft_ms,
            ));
        }
        candidates.extend(self.frontier_catalog.quote(
            session.ledger(),
            now_ms(),
            isl_tokens as u64,
            self.config.expected_output_tokens as u64,
        ));

        // --- choose --------------------------------------------------------
        let decision = self
            .policy
            .choose(&RoutingContext {
                session_id,
                turn_index,
                isl_tokens,
                candidates: &candidates,
                ledger: session.ledger(),
            })
            .await?;

        let chosen = candidates
            .iter()
            .find(|candidate| candidate.target == decision.target)
            .cloned()
            .ok_or_else(|| EngineError::UnresolvableTarget(decision.target.clone()))?;

        // Recorded before execution: a decision that led to a failure is still
        // part of the audit trail.
        session
            .record_routing(
                &response_id,
                DecisionRecord {
                    chosen: decision.target.clone(),
                    rationale: decision.rationale.clone(),
                    policy: self.policy.name().to_string(),
                    isl_tokens: isl_tokens as u64,
                    expected_prefill_tokens: chosen.expected_prefill_tokens,
                    expected_cost_usd: chosen.expected_cost_usd,
                    considered: candidates.clone(),
                },
            )
            .await?;

        // --- execute -------------------------------------------------------
        let prompt = render_prompt(assembler.items());
        let (text, usage) = match &decision.target {
            Target::Local { .. } => {
                let quote = local_quote
                    .as_ref()
                    .ok_or_else(|| EngineError::UnresolvableTarget(decision.target.clone()))?;
                let fleet = self
                    .fleet
                    .as_ref()
                    .ok_or_else(|| EngineError::UnresolvableTarget(decision.target.clone()))?;

                // Book only now that local has actually won. Had the frontier
                // won, the pending selection would simply expire unclaimed.
                let reservation = Arc::clone(fleet).reserve(quote).await?;
                let outcome = self
                    .local_executor
                    .execute(
                        &quote.endpoint,
                        &prompt,
                        Some(self.config.expected_output_tokens),
                    )
                    .await;

                // Settle regardless of outcome. A leaked reservation would
                // permanently distort this worker's load.
                reservation.prefill_complete().await?;
                let settled = reservation.release().await;
                let outcome = outcome?;
                settled?;

                let cached = isl_tokens.saturating_sub(quote.effective_prefill_tokens);
                (
                    outcome.text,
                    Usage {
                        input_tokens: isl_tokens as u64,
                        cached_input_tokens: cached as u64,
                        output_tokens: outcome.output_tokens,
                    },
                )
            }
            Target::Frontier { .. } => {
                let chunks = self
                    .frontier_client
                    .execute(&FrontierQuote {
                        target: decision.target.clone(),
                        prompt,
                        // Stable for the life of the session: providers use it
                        // to steer requests to the same cache node, so varying
                        // it would defeat the hit we just routed on.
                        prompt_cache_key: session_id.to_string(),
                        expected_output_tokens: Some(self.config.expected_output_tokens),
                    })
                    .await?;
                fold_frontier_chunks(chunks)
            }
        };

        session.complete(&response_id, &text, usage.clone()).await?;
        let last_seq = session.last_seq();
        session.release().await?;

        Ok(TurnResult {
            response_id,
            text,
            decision: Some(decision),
            usage,
            last_seq,
            deduplicated: false,
        })
    }
}

/// Concatenate a provider's chunks into a response plus usage.
fn fold_frontier_chunks(chunks: Vec<FrontierChunk>) -> (String, Usage) {
    let mut text = String::new();
    let mut usage = Usage::default();
    for chunk in chunks {
        match chunk {
            FrontierChunk::OutputText(part) => text.push_str(&part),
            FrontierChunk::Done {
                input_tokens,
                cached_input_tokens,
                output_tokens,
            } => {
                usage = Usage {
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                };
            }
        }
    }
    (text, usage)
}

/// Deterministic prompt rendering.
///
/// Determinism is load-bearing rather than cosmetic: a rendering that varied
/// between turns would change the token prefix and invalidate every cached
/// block after the first divergence.
fn render_prompt(items: &[Item]) -> String {
    items
        .iter()
        .map(Item::render)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Recover a completed response's text from the log.
async fn replay_output<S: SessionStore>(
    session: &Session<S>,
    response_id: &ResponseId,
) -> Result<String, SessionError> {
    Ok(session
        .state()
        .items
        .iter()
        .filter(|item| item.response_id.as_ref() == Some(response_id))
        .map(Item::render)
        .collect::<Vec<_>>()
        .join(""))
}
