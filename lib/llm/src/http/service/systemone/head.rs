// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::DecisionExecutor,
    protocols::decision_head::{DecisionHeadRequest, DecisionHeadResponse},
    protocols::decision_read::DecisionReadRequest,
};

pub(super) enum PreparedExecution {
    Token(Vec<PreparedBranch>),
    Head(DecisionHeadRequest),
    NativeRead(DecisionReadRequest),
}

pub(super) fn prepare(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<PreparedExecution, (StatusCode, String)> {
    if selection.profile.is_token_scoring() {
        return prepare_branches(request, selection, max_input_tokens)
            .map(PreparedExecution::Token);
    }
    if request.temperature != 1.0 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "native decision profiles require temperature one".into(),
        ));
    }
    let mut salt = [0_u8; 16];
    rand::rng().fill(&mut salt);
    let mut canonical = request.clone();
    canonical.model = selection.canonical_model.clone();
    let salt = URL_SAFE_NO_PAD.encode(salt);
    let prepared = if selection.profile.is_native_read() {
        PreparedExecution::NativeRead(DecisionReadRequest::new(
            &canonical,
            selection.profile,
            salt,
            max_input_tokens,
        ))
    } else {
        PreparedExecution::Head(DecisionHeadRequest::new(
            &canonical,
            selection.profile,
            salt,
            max_input_tokens,
        ))
    };
    let encoded = match &prepared {
        PreparedExecution::Head(wire) => serde_json::to_vec(wire),
        PreparedExecution::NativeRead(wire) => serde_json::to_vec(wire),
        PreparedExecution::Token(_) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid native decision preparation".into(),
            ));
        }
    }
    .map_err(|_| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid native decision request".into(),
        )
    })?;
    if encoded.len() > MAX_TOKENIZATION_TEXT_BYTES {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "native decision encoding byte budget exceeded".into(),
        ));
    }
    Ok(prepared)
}

pub(super) fn observe_work(model: &str, prepared: &PreparedExecution, metrics: &Metrics) {
    if let PreparedExecution::Token(branches) = prepared {
        metrics.observe_systemone_work(
            model,
            branches.len(),
            branches.iter().map(|b| b.prompt_ids.len()).sum(),
            branches.iter().map(|b| b.label_ids.len()).sum(),
        );
    }
}

pub(super) async fn dispatch(
    selection: SystemOneExecutionSelection,
    prepared: PreparedExecution,
    request_id: String,
    parent: Arc<dyn AsyncEngineContext>,
    permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
    request: Arc<CanonicalRequest>,
) -> Response {
    match prepared {
        PreparedExecution::Token(branches) => {
            super::dispatch(
                selection, branches, request_id, parent, permit, metrics, request,
            )
            .await
        }
        PreparedExecution::Head(wire) => {
            let _permit = permit;
            let result = run(&selection, wire, &request_id, parent.clone(), &request).await;
            project_native(result, &selection, &request_id, &parent, &metrics, &request)
        }
        PreparedExecution::NativeRead(wire) => {
            let _permit = permit;
            let result =
                super::native_read::run(&selection, wire, &request_id, parent.clone(), &request)
                    .await;
            project_native(result, &selection, &request_id, &parent, &metrics, &request)
        }
    }
}

fn project_native(
    result: Result<(Vec<QuestionOutcome>, Usage), DispatchError>,
    selection: &SystemOneExecutionSelection,
    request_id: &str,
    parent: &Arc<dyn AsyncEngineContext>,
    metrics: &Metrics,
    request: &CanonicalRequest,
) -> Response {
    match result {
        Ok((outcomes, usage)) => {
            metrics.observe_systemone_work(
                &selection.canonical_model,
                request.questions.len(),
                usage.input_tokens as usize,
                request.questions.iter().map(|q| q.candidates.len()).sum(),
            );
            let mut response =
                match project_response(request, &selection.canonical_model, &outcomes, &usage) {
                    Ok(body) => Json(body).into_response(),
                    Err(cause) => decision_error(cause),
                };
            attach_request_ids(&mut response, request_id);
            response.headers_mut().insert(
                HeaderName::from_static("x-dynamo-systemone-version"),
                HeaderValue::from_static("1"),
            );
            response
        }
        Err(cause) => dispatch_error(request_id, parent, cause, request.dialect),
    }
}

async fn run(
    selection: &SystemOneExecutionSelection,
    wire: DecisionHeadRequest,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    request: &CanonicalRequest,
) -> Result<(Vec<QuestionOutcome>, Usage), DispatchError> {
    selection
        .validate_dispatch(None)
        .map_err(|_| DispatchError::unavailable("selected head revision is unavailable"))?;
    let DecisionExecutor::Head { engine } = &selection.executor else {
        return Err(DispatchError::malformed("invalid head executor"));
    };
    let max_input_tokens = wire.decision_head.max_input_tokens;
    let context =
        Context::with_id_and_metadata(wire, format!("{request_id}-head"), Default::default());
    parent.link_child(context.context());
    if parent.is_killed() {
        context.context().kill();
        return Err(DispatchError::cancelled());
    }
    let mut stream = run_until_killed(parent.as_ref(), engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error(cause.as_ref()))?;
    let stream_context = stream.context();
    parent.link_child(stream_context.clone());
    if parent.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    let mut terminal: Option<DecisionHeadResponse> = None;
    loop {
        let frame = run_until_killed(parent.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        if let Some(response) = frame.into_data().map_err(|cause| backend_error(&cause))?
            && terminal.replace(response).is_some()
        {
            return Err(DispatchError::malformed("multiple terminal head responses"));
        }
    }
    if parent.is_killed() {
        return Err(DispatchError::cancelled());
    }
    terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal head response"))?
        .validate(request, selection.profile, max_input_tokens)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))
}

pub(super) fn backend_error(cause: &(dyn std::error::Error + 'static)) -> DispatchError {
    let mut error = DispatchError::backend(cause);
    if let Some(canonical) = super::super::error::find_canonical_error_in_chain(cause)
        && canonical.class().normalized() == dynamo_runtime::error::ErrorClass::InvalidRequest
    {
        error.status = StatusCode::UNPROCESSABLE_ENTITY;
    }
    error
}
