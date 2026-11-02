// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use super::{backend_error::lifecycle_error, token_scorer};
use crate::local_model::runtime_config::TokenBackend;
use futures::{StreamExt, stream};

struct TokenDispatch<'a> {
    selection: &'a SystemOneExecutionSelection,
    engine: &'a crate::types::openai::generate::GenerateStreamingEngine,
    backend: TokenBackend,
    request_id: &'a str,
    parent_context: &'a Arc<dyn AsyncEngineContext>,
    metrics: &'a Metrics,
    metric_model: &'a str,
}

pub(super) fn prepare_branches(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
    let cache_salt = fresh_cache_salt();
    let preprocessor = &selection.preprocessor;
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
        let wire = token_scorer::build(TokenBackend::Sglang, prompt_ids, &label_ids, &cache_salt)
            .map_err(|cause| {
            tracing::error!(error = %cause, "failed to build System One native request");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error".to_string(),
            )
        })?;
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
    selection: SystemOneExecutionSelection,
    mut branches: Vec<PreparedBranch>,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    _admission_permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
    request: Arc<CanonicalRequest>,
) -> Response {
    let dialect = request.dialect;
    if branches.is_empty()
        || branches
            .iter()
            .any(|branch| branch.wire.backend() != TokenBackend::Sglang)
    {
        return dispatch_error(
            &request_id,
            &parent_context,
            DispatchError::malformed("prepared scoring transport disagrees with its executor"),
            dialect,
        );
    }
    let dispatch = TokenDispatch {
        selection: &selection,
        engine: &selection.engine,
        backend: TokenBackend::Sglang,
        request_id: &request_id,
        parent_context: &parent_context,
        metrics: &metrics,
        metric_model: &selection.canonical_model,
    };
    let first = branches.remove(0);
    let first_started = Instant::now();
    let first_result = match dispatch.run_branch(first, None).await {
        Ok(result) => result,
        Err(cause) => return dispatch_error(&request_id, &parent_context, cause, dialect),
    };
    metrics.observe_systemone_phase(
        &selection.canonical_model,
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
    .buffer_unordered(token_scorer::max_concurrent_siblings(dispatch.backend));
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
        &selection.canonical_model,
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
        let context: Context<PreprocessedRequest> =
            Context::with_id_and_metadata(preprocessed, branch_id, Default::default());
        let stream =
            decision_lifecycle::start_linked(parent_context.as_ref(), context, engine.as_ref())
                .await
                .map_err(lifecycle_error)?;
        let result = token_scorer::finish_branch(
            self.backend,
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
