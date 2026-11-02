// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn prepare_branches(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
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
        let (prompt_ids, label_ids, encoding_bytes) = selection
            .preprocessor
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
                        error.class() == dynamo_runtime::error::ErrorClass::InvalidRequest
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
        let native =
            build_native_score_request(&prompt_ids, &label_ids, "validation").map_err(|cause| {
                tracing::error!(error = %cause, "failed to build System One native request");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
            })?;
        branches.push(PreparedBranch {
            index,
            temperature: request.temperature,
            prompt_ids,
            label_ids,
            native: Some(native),
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
    let cache_salt = fresh_cache_salt();
    let first = branches.remove(0);
    let first_started = Instant::now();
    let first_result = match run_branch(
        selection.clone(),
        first,
        request_id.clone(),
        cache_salt.clone(),
        None,
        parent_context.clone(),
        metrics.clone(),
    )
    .await
    {
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
    let mut siblings = stream::iter(branches.into_iter().map(|branch| {
        run_branch(
            selection.clone(),
            branch,
            request_id.clone(),
            cache_salt.clone(),
            Some(placement),
            parent_context.clone(),
            metrics.clone(),
        )
    }))
    .buffer_unordered(MAX_CONCURRENT_SIBLINGS);
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

async fn run_branch(
    selection: SystemOneExecutionSelection,
    mut prepared: PreparedBranch,
    request_id: String,
    cache_salt: String,
    pinned: Option<Placement>,
    parent_context: Arc<dyn AsyncEngineContext>,
    metrics: Arc<Metrics>,
) -> Result<BranchResult, DispatchError> {
    selection
        .validate_dispatch(pinned.map(|placement| placement.worker_id))
        .map_err(|_| {
            DispatchError::unavailable("selected worker revision is no longer available")
        })?;
    let mut native = prepared
        .native
        .take()
        .ok_or_else(|| DispatchError::malformed("native request was already consumed"))?;
    native["cache_salt"] = serde_json::Value::String(cache_salt);
    let branch_id = format!("{request_id}-{}", prepared.index);
    let Bound {
        request: mut preprocessed,
        exchange,
    } = bind_native(
        &selection.canonical_model,
        selection.card.kv_cache_block_size,
        &native,
        &branch_id,
    )
    .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))?;
    preprocessed
        .routing
        .get_or_insert_default()
        .expected_output_tokens = Some(0);
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(prepared.prompt_ids.len(), None);
    if let Some(placement) = pinned {
        pin_to_placement(&mut preprocessed, placement);
    }
    preprocessed.tracker = Some(tracker.clone());
    let context: Context<PreprocessedRequest> =
        Context::with_id_and_metadata(preprocessed, branch_id, Default::default());
    let branch_context = context.context();
    parent_context.link_child(branch_context.clone());
    if parent_context.is_killed() {
        branch_context.kill();
        return Err(DispatchError::cancelled());
    }
    let stream = run_until_killed(parent_context.as_ref(), selection.engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    let stream_context = stream.context();
    parent_context.link_child(stream_context.clone());
    if parent_context.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    let result = finish_branch(
        StartedBranch {
            prepared,
            tracker,
            stream,
            exchange,
        },
        selection.card.runtime_config.data_parallel_start_rank,
        selection.card.runtime_config.data_parallel_size,
        parent_context,
    )
    .await?;
    if let Some(cached_tokens) = result.cached_tokens {
        metrics.observe_systemone_branch(&selection.canonical_model, cached_tokens);
    }
    Ok(result)
}

fn stream_error(cause: dynamo_runtime::error::DynamoError) -> DispatchError {
    if cause.class() == dynamo_runtime::error::ErrorClass::BackendProtocol {
        DispatchError::malformed("invalid native scoring stream")
    } else {
        backend_error::classify(&cause)
    }
}

fn accept_frame(
    frame: serde_json::Value,
    terminal: &mut Option<serde_json::Value>,
) -> Result<(), DispatchError> {
    let finish = frame
        .get("meta_info")
        .and_then(|meta| meta.get("finish_reason"));
    if let Some(cause) = finish.and_then(abort::classify) {
        return Err(cause);
    }
    if finish.is_some_and(|finish| !finish.is_null()) && terminal.replace(frame).is_some() {
        return Err(DispatchError::malformed(
            "multiple terminal SGLang scoring frames",
        ));
    }
    Ok(())
}

pub(super) async fn finish_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let result = read_branch(
        started,
        data_parallel_start_rank,
        data_parallel_size,
        parent_context.clone(),
    )
    .await;
    if parent_context.is_killed() {
        return Err(DispatchError::cancelled());
    }
    result
}

async fn read_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let StartedBranch {
        prepared,
        tracker,
        mut stream,
        mut exchange,
    } = started;
    let mut terminal = None;
    let mut frames = Vec::new();
    loop {
        let delta = run_until_killed(parent_context.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(delta) = delta else { break };
        let prefill = delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION);
        if let Some(output) = delta.into_data().map_err(stream_error)? {
            exchange
                .chunk(output, prefill, &mut |frame| frames.push(frame.clone()))
                .map_err(stream_error)?;
        }
        for frame in frames.drain(..) {
            accept_frame(frame, &mut terminal)?;
        }
    }
    if let Some(frame) = exchange.finish().map_err(stream_error)? {
        accept_frame(frame, &mut terminal)?;
    }
    let terminal = terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal SGLang scoring frame"))?;
    let placement = tracker
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
        })?;
    let scores = parse_candidate_scores(&terminal, &prepared.label_ids)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    let input_tokens = prepared.prompt_ids.len();
    let cached_tokens = measured_cached_tokens(&terminal, input_tokens)?;
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens,
        cached_tokens,
        placement,
    })
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
        .ok_or_else(|| DispatchError::malformed("invalid cache-read token count"))?;
    Ok(Some(count))
}
