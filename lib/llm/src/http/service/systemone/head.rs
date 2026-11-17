// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{discovery::native_head::NativeHeadSelection, protocols::systemone::head::HeadRequest};

pub(super) async fn handle(
    state: Arc<service_v2::State>,
    request_id: String,
    request: CanonicalRequest,
    selection: NativeHeadSelection,
    deadline: tokio::time::Instant,
) -> Response {
    let dialect = request.dialect;
    let capabilities = dynamo_decisions::Capabilities {
        max_questions: 64,
        max_candidates: 26,
        supports_predicate: true,
        supports_choice: true,
        supports_score: true,
        measured_cache_reads: true,
        prompt_format_version: 1,
    };
    if let Err(cause) = request.validate_capabilities(&capabilities) {
        return decision_error(cause);
    }
    let request = Arc::new(request);
    let preparing = request.clone();
    let model = selection.canonical_model.clone();
    let max_input_tokens = state.systemone_max_input_tokens();
    let mut salt = [0_u8; 16];
    rand::rng().fill(&mut salt);
    let cache_salt = URL_SAFE_NO_PAD.encode(salt);
    let metric_model = state
        .manager()
        .metric_model_for(&selection.canonical_model)
        .to_owned();
    run_execution(
        &state,
        request_id,
        request,
        metric_model,
        deadline,
        move || {
            HeadRequest::new(&preparing, &model, cache_salt, max_input_tokens)
                .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))
        },
        move |wire, permit, execution| async move {
            let _permit = permit;
            let started = Instant::now();
            let result = run(
                &selection,
                wire,
                &execution.request_id,
                execution.parent.clone(),
                &execution.request,
                max_input_tokens,
            )
            .await;
            execution.metrics.observe_systemone_phase(
                &execution.metric_model,
                SystemOnePhase::FirstBranch,
                started.elapsed().as_secs_f64(),
            );
            match result {
                Ok((answers, usage)) => {
                    execution.metrics.observe_systemone_work(
                        &execution.metric_model,
                        execution.request.questions.len(),
                        usage.input_tokens as usize,
                        0,
                    );
                    execution.metrics.observe_systemone_branch(
                        &execution.metric_model,
                        usage.cached_tokens.unwrap_or_default() as usize,
                    );
                    match project_response(
                        &execution.request,
                        &selection.canonical_model,
                        &answers,
                        &usage,
                    ) {
                        Ok(value) => Json(value).into_response(),
                        Err(cause) => decision_error(cause),
                    }
                }
                Err(cause) => {
                    dispatch_error(&execution.request_id, &execution.parent, cause, dialect)
                }
            }
        },
    )
    .await
}

fn invalid(detail: impl Into<String>) -> DispatchError {
    DispatchError {
        status: StatusCode::BAD_GATEWAY,
        public_message: "malformed native decision response",
        detail: detail.into(),
        retry_after: false,
        is_capacity_exhausted: false,
    }
}

async fn run(
    selection: &NativeHeadSelection,
    wire: HeadRequest,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    request: &CanonicalRequest,
    max_input_tokens: usize,
) -> Result<(Vec<QuestionOutcome>, Usage), DispatchError> {
    let context = Context::with_id_and_metadata(
        wire,
        format!("{request_id}-head"),
        [("dynamo.internal.decision".to_owned(), "head".to_owned())].into(),
    );
    let child = context.context();
    parent.link_child(child.clone());
    if parent.is_killed() {
        child.kill();
        return Err(DispatchError::cancelled());
    }
    let mut stream = run_until_killed(parent.as_ref(), selection.engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    parent.link_child(stream.context());
    let mut terminal = None;
    loop {
        let frame = run_until_killed(parent.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        if let Some(output) = frame
            .into_data()
            .map_err(|cause| backend_error::classify(&cause))?
            && terminal.replace(output).is_some()
        {
            return Err(invalid("multiple native decision terminal frames"));
        }
    }
    if parent.is_killed() {
        return Err(DispatchError::cancelled());
    }
    let result = terminal
        .ok_or_else(|| invalid("native decision terminal frame is missing"))?
        .validate(request, max_input_tokens)
        .map_err(|cause| invalid(cause.to_string()))?;
    if selection.card.effective_context_length() > 0
        && result.1.input_tokens > u64::from(selection.card.effective_context_length())
    {
        return Err(invalid(
            "native response exceeds the effective context limit",
        ));
    }
    Ok(result)
}
