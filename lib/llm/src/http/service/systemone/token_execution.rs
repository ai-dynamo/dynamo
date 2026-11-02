// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::backend_error::lifecycle_error;
use super::*;
use crate::discovery::{DecisionContext, TokenExecutor};
use futures::{StreamExt, stream};

#[path = "token_execution/backend.rs"]
mod backend;
pub(super) use backend::scoring_for;

pub(super) struct TokenPolicy {
    format_name: &'static str,
    output_tokens: u32,
    sibling_concurrency: usize,
}

/// Transport behavior belongs to the executor, not the passive model-card profile.
#[async_trait::async_trait]
pub(super) trait TokenScoring: Send + Sync {
    fn policy(&self) -> TokenPolicy;
    fn build_request(
        &self,
        prompt_ids: Vec<u32>,
        label_ids: &[u32],
        cache_salt: &str,
    ) -> Result<serde_json::Value, crate::protocols::systemone::SystemOneError>;
    async fn read_frames(
        &self,
        started: StartedBranch,
        rank: u32,
        size: u32,
        parent: Arc<dyn AsyncEngineContext>,
    ) -> Result<BranchResult, DispatchError>;
    async fn read_branch(
        &self,
        started: StartedBranch,
        rank: u32,
        size: u32,
        parent: Arc<dyn AsyncEngineContext>,
    ) -> Result<BranchResult, DispatchError> {
        let result = self.read_frames(started, rank, size, parent.clone()).await;
        if parent.is_killed() {
            return Err(DispatchError::cancelled());
        }
        result
    }
}

pub(super) struct PreparedTokenExecution {
    executor: TokenExecutor,
    branches: Vec<PreparedBranch>,
}

impl PreparedTokenExecution {
    pub(super) fn new(
        executor: TokenExecutor,
        branches: Vec<PreparedBranch>,
    ) -> Result<Self, DispatchError> {
        if branches.is_empty() {
            return Err(DispatchError::malformed(
                "prepared scoring execution has no branches",
            ));
        }
        Ok(Self { executor, branches })
    }
    pub(super) fn branches(&self) -> &[PreparedBranch] {
        &self.branches
    }
}

struct TokenDispatch<'a> {
    selection: &'a DecisionContext,
    engine: &'a crate::types::openai::generate::GenerateStreamingEngine,
    scoring: &'static dyn TokenScoring,
    request_id: &'a str,
    parent_context: &'a Arc<dyn AsyncEngineContext>,
    metrics: &'a Metrics,
    metric_model: &'a str,
}

pub(super) fn prepare_branches(
    request: &CanonicalRequest,
    selection: &DecisionContext,
    executor: &TokenExecutor,
    max_input_tokens: usize,
    cache_salt: &str,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
    let preprocessor = executor.preprocessor();
    let context_length = selection.card.effective_context_length() as usize;
    let mut total_input_tokens = 0_usize;
    let mut remaining_tokenization_work = tokenization_work_limit(
        max_input_tokens,
        selection.decision_capabilities().max_candidates,
    )
    .ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "System One tokenization work configuration exceeds its supported limit".to_string(),
        )
    })?;
    let mut remaining_encoding_bytes = MAX_TOKENIZATION_TEXT_BYTES;
    let mut branches = Vec::with_capacity(request.questions.len());
    for question in &request.questions {
        let index = question.ordinal;
        let question_id = question.id.as_deref().unwrap_or("unnamed");
        let rendered = render_question_prompt(&request.input, question)
            .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;
        let remaining_tokens = max_input_tokens.saturating_sub(total_input_tokens);
        let max_prompt_tokens = if context_length == 0 {
            remaining_tokens
        } else {
            remaining_tokens.min(context_length.saturating_sub(1))
        }
        .min(encoding_prompt_limit(
            remaining_tokenization_work,
            rendered.labels.len(),
        ));
        let (prompt_ids, label_ids, encoding_bytes) = preprocessor
            .render_systemone_question(
                &selection.canonical_model,
                &rendered.content,
                Some(&request.chat_template_kwargs),
                &rendered.labels,
                (max_prompt_tokens, remaining_encoding_bytes),
            )
            .map_err(|cause| {
                if let Some(oversized) =
                    cause.downcast_ref::<crate::preprocessor::SystemOnePromptTooLong>()
                {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("question {question_id:?}: {oversized}"),
                    );
                }
                if let Some(oversized) =
                    cause.downcast_ref::<crate::preprocessor::SystemOneEncodingTooLarge>()
                {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("question {question_id:?}: {oversized}"),
                    );
                }
                if let Some(incompatible) = cause
                    .downcast_ref::<crate::preprocessor::SystemOneCandidateLabelsIncompatible>()
                {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("question {question_id:?}: {incompatible}"),
                    );
                }
                let reason = if let Some(incompatible) =
                    cause.downcast_ref::<crate::preprocessor::SystemOneModelIncompatible>()
                {
                    incompatible.0
                } else if super::super::error::find_canonical_error_in_chain(cause.as_ref())
                    .is_some_and(|error| {
                        error.class().normalized() == dynamo_runtime::error::ErrorClass::InvalidRequest
                    })
                {
                    "chat template or tokenizer failed"
                } else {
                    tracing::error!("System One prompt tokenization failed");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "internal server error".to_string(),
                    );
                };
                tracing::warn!(reason, "System One prompt rendering failed");
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "question {question_id:?}: tokenizer or chat template could not render this request"
                    ),
                )
            })?;
        total_input_tokens =
            add_input_tokens(total_input_tokens, prompt_ids.len(), max_input_tokens)
                .map_err(|message| (StatusCode::UNPROCESSABLE_ENTITY, message))?;
        remaining_tokenization_work -= prompt_ids.len() * (rendered.labels.len() + 2);
        remaining_encoding_bytes -= encoding_bytes;
        let prompt_len = prompt_ids.len();
        let wire = PreparedScoringRequest::new(prompt_ids, label_ids.clone(), cache_salt);
        branches.push(PreparedBranch {
            scoring: ScoringBranch {
                index,
                temperature: request.temperature,
                prompt_len,
                label_ids,
            },
            wire,
        });
    }
    Ok(branches)
}

pub(super) async fn dispatch(
    selection: &DecisionContext,
    prepared: PreparedTokenExecution,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    metrics: Arc<Metrics>,
    metric_model: String,
    request: Arc<CanonicalRequest>,
) -> Response {
    let dialect = request.dialect;
    let PreparedTokenExecution {
        executor,
        mut branches,
    } = prepared;
    let dispatch = TokenDispatch {
        selection,
        engine: executor.engine(),
        scoring: scoring_for(executor.backend()),
        request_id: &request_id,
        parent_context: &parent_context,
        metrics: &metrics,
        metric_model: &metric_model,
    };
    let first = branches.remove(0);
    let first_started = Instant::now();
    let first_result = match dispatch.run_branch(first, None).await {
        Ok(result) => result,
        Err(cause) => return dispatch_error(&request_id, &parent_context, cause, dialect),
    };
    metrics.observe_systemone_phase(
        &metric_model,
        SystemOnePhase::FirstBranch,
        first_started.elapsed().as_secs_f64(),
    );
    let placement = first_result.placement;
    let mut results = Vec::with_capacity(branches.len() + 1);
    results.push(first_result);

    let fanout_started = Instant::now();
    let mut siblings = stream::iter(
        branches
            .into_iter()
            .map(|branch| dispatch.run_branch(branch, Some(placement))),
    )
    .buffer_unordered(dispatch.scoring.policy().sibling_concurrency);
    let mut failure = None;
    while let Some(result) = siblings.next().await {
        if failure.is_some() {
            continue;
        }
        match result {
            Ok(result) if result.placement == placement => results.push(result),
            Ok(result) => {
                parent_context.kill();
                failure = Some(DispatchError::unavailable(format!(
                    "branch {} was routed to {:?}, expected {:?}",
                    result.index, result.placement, placement
                )));
            }
            Err(cause) => {
                parent_context.kill();
                failure = Some(cause);
            }
        }
    }
    drop(siblings);
    if let Some(cause) = failure {
        return dispatch_error(&request_id, &parent_context, cause, dialect);
    }

    metrics.observe_systemone_phase(
        &metric_model,
        SystemOnePhase::Fanout,
        fanout_started.elapsed().as_secs_f64(),
    );

    results.sort_by_key(|result| result.index);
    let mut outcomes = Vec::with_capacity(results.len());
    let mut usage = Usage {
        cached_tokens: Some(0),
        ..Default::default()
    };
    for result in results {
        usage.input_tokens += result.input_tokens as u64;
        usage.output_tokens += result.output_tokens as u64;
        usage.cached_tokens = usage
            .cached_tokens
            .zip(result.cached_tokens)
            .map(|(total, next)| total + next as u64);
        outcomes.push(result.answer);
    }
    finish_response(
        &request,
        &selection.canonical_model,
        &outcomes,
        &usage,
        &request_id,
    )
}

impl TokenDispatch<'_> {
    async fn run_branch(
        &self,
        prepared: PreparedBranch,
        pinned: Option<Placement>,
    ) -> Result<BranchResult, DispatchError> {
        let Self {
            selection,
            engine,
            request_id,
            parent_context,
            metrics,
            ..
        } = self;
        selection
            .validate_dispatch(pinned.map(|placement| placement.worker_id))
            .map_err(|_| {
                DispatchError::unavailable("selected worker revision is no longer available")
            })?;
        let PreparedBranch {
            scoring: prepared,
            wire,
        } = prepared;
        let branch_id = format!("{request_id}-{}", prepared.index);
        let Bound {
            request: mut preprocessed,
            exchange,
        } = wire.compile(
            self.scoring,
            &selection.canonical_model,
            selection.card.kv_cache_block_size,
            &branch_id,
        )?;
        let tracker =
            preprocessed.tracker.as_ref().cloned().ok_or_else(|| {
                DispatchError::malformed("scoring preprocessing omitted its tracker")
            })?;
        if let Some(placement) = pinned {
            pin_to_placement(&mut preprocessed, placement);
        }
        let mut context: Context<PreprocessedRequest> =
            Context::with_id_and_metadata(preprocessed, branch_id, Default::default());
        let receipt = selection.attach_dispatch_guard(&mut context);
        let mut stream =
            decision_lifecycle::start_linked(parent_context.as_ref(), context, engine.as_ref())
                .await
                .map_err(lifecycle_error)?;
        // A lazy engine may establish its transport only while polling the first frame.
        // Do not trust even an otherwise valid score stream unless the final hop checked it.
        let first = run_until_killed(parent_context.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        if !receipt.is_validated() {
            stream.context().kill();
            return Err(DispatchError::unavailable(
                "scoring dispatch guard was not validated",
            ));
        }
        let Some(first) = first else {
            return Err(DispatchError::invalid_executor(
                "empty decision scoring stream",
            ));
        };
        let stream_context = stream.context();
        let stream = dynamo_runtime::pipeline::ResponseStream::new(
            Box::pin(stream::iter([first]).chain(stream)),
            stream_context,
        );
        let result = self
            .scoring
            .read_branch(
                StartedBranch {
                    prepared,
                    tracker,
                    stream,
                    exchange,
                },
                selection.card.runtime_config.data_parallel_start_rank,
                selection.card.runtime_config.data_parallel_size,
                (*parent_context).clone(),
            )
            .await?;
        if let Some(cached_tokens) = result.cached_tokens {
            metrics.observe_systemone_branch(self.metric_model, cached_tokens);
        }
        Ok(result)
    }
}

pub(super) struct PreparedScoringRequest {
    prompt_ids: Vec<u32>,
    label_ids: Vec<u32>,
    cache_salt: String,
}

pub(super) fn measured_cached_tokens(
    response: &serde_json::Value,
    input_tokens: usize,
) -> Result<Option<usize>, DispatchError> {
    let Some(count) = response
        .get("meta_info")
        .and_then(|meta| meta.get("cached_tokens"))
    else {
        return Ok(None);
    };
    let count = count
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value <= input_tokens)
        .ok_or_else(|| DispatchError::invalid_executor("invalid cache-read token count"))?;
    Ok(Some(count))
}

impl PreparedScoringRequest {
    #[cfg(test)]
    pub(super) fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    pub(super) fn new(prompt_ids: Vec<u32>, label_ids: Vec<u32>, cache_salt: &str) -> Self {
        Self {
            prompt_ids,
            label_ids,
            cache_salt: cache_salt.to_owned(),
        }
    }
    pub(super) fn compile(
        self,
        scoring: &dyn TokenScoring,
        canonical_model: &str,
        kv_cache_block_size: u32,
        branch_id: &str,
    ) -> Result<crate::generate_format::Bound, DispatchError> {
        let prompt_len = self.prompt_ids.len();
        let native = scoring
            .build_request(self.prompt_ids, &self.label_ids, &self.cache_salt)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        let mut bound = super::bind_native(
            scoring.policy().format_name,
            canonical_model,
            kv_cache_block_size,
            &native,
            branch_id,
        )
        .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))?;
        bound
            .request
            .routing
            .get_or_insert_default()
            .expected_output_tokens = Some(scoring.policy().output_tokens);
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(prompt_len, None);
        bound.request.tracker = Some(tracker);
        Ok(bound)
    }
}

pub(super) fn scoring_placement(
    tracker: &RequestTracker,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
) -> Result<Placement, DispatchError> {
    tracker
        .get_worker_info()
        .and_then(|worker| {
            worker
                .decode_worker_id
                .or(worker.prefill_worker_id)
                .map(|worker_id| {
                    resolve_dp_rank(
                        worker.decode_dp_rank.or(worker.prefill_dp_rank),
                        data_parallel_start_rank,
                        data_parallel_size,
                    )
                    .map(|dp_rank| Placement { worker_id, dp_rank })
                })
        })
        .flatten()
        .ok_or_else(|| {
            DispatchError::unavailable("Dynamo did not report an unambiguous scoring worker rank")
        })
}
