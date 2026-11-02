// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::{DecisionContext, DecisionEngine},
    protocols::decision_head::DecisionHeadRequest,
    protocols::decision_read::DecisionReadRequest,
};

pub(super) struct PreparedExecution {
    context: DecisionContext,
    payload: PreparedPayload,
}

enum PreparedPayload {
    Token(super::scoring::PreparedTokenExecution),
    Head(PreparedNativeExecution<DecisionHeadRequest>),
    Read(PreparedNativeExecution<DecisionReadRequest>),
}

struct PreparedNativeExecution<R: super::native::NativeRequest> {
    engine: super::native::NativeEngine<R>,
    wire: R,
}

impl PreparedExecution {
    #[cfg(test)]
    pub(super) fn token(
        selection: SystemOneExecutionSelection,
        branches: Vec<PreparedBranch>,
    ) -> Option<Self> {
        let (context, engine) = selection.into_parts();
        match engine {
            DecisionEngine::Token(executor) => Some(Self {
                context,
                payload: PreparedPayload::Token(
                    super::scoring::PreparedTokenExecution::new(executor, branches).ok()?,
                ),
            }),
            DecisionEngine::Head(_) | DecisionEngine::Read(_) => None,
        }
    }
}

pub(super) fn prepare(
    request: &CanonicalRequest,
    selection: SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<PreparedExecution, (StatusCode, String)> {
    let cache_salt = fresh_cache_salt();
    let (context, engine) = selection.into_parts();
    let payload = match engine {
        DecisionEngine::Token(executor) => {
            let branches = super::scoring::prepare_branches(
                request,
                &context,
                &executor,
                max_input_tokens,
                &cache_salt,
            )?;
            PreparedPayload::Token(
                super::scoring::PreparedTokenExecution::new(executor, branches).map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "invalid prepared scoring execution".into(),
                    )
                })?,
            )
        }
        DecisionEngine::Head(engine) => PreparedPayload::Head(PreparedNativeExecution {
            engine,
            wire: prepare_native(request, &context, max_input_tokens, &cache_salt)?,
        }),
        DecisionEngine::Read(engine) => PreparedPayload::Read(PreparedNativeExecution {
            engine,
            wire: prepare_native(request, &context, max_input_tokens, &cache_salt)?,
        }),
    };
    Ok(PreparedExecution { context, payload })
}

fn prepare_native<R: super::native::NativeRequest + serde::Serialize>(
    request: &CanonicalRequest,
    selection: &DecisionContext,
    max_input_tokens: usize,
    cache_salt: &str,
) -> Result<R, (StatusCode, String)> {
    if request.temperature != 1.0 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "native decision profiles require temperature one".into(),
        ));
    }
    if !request.chat_template_kwargs.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "native decision profiles do not support chat template overrides".into(),
        ));
    }
    debug_assert!(max_input_tokens <= MAX_DECISION_INPUT_TOKENS);
    let max_tokenization_work = tokenization_work_limit(
        max_input_tokens,
        selection.decision_capabilities().max_candidates,
    )
    .ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "invalid tokenization budget".into(),
        )
    })?
    .min(MAX_DECISION_INPUT_TOKENS);
    let wire = R::wrap(
        crate::protocols::decision_native::NativeDecisionRequestV1::new(
            request,
            &selection.canonical_model,
            selection.profile(),
            cache_salt.to_owned(),
            max_input_tokens,
            max_tokenization_work,
        ),
    );
    let encoded = serde_json::to_vec(&wire).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "invalid native decision request".into(),
        )
    })?;
    if encoded.len() > MAX_TOKENIZATION_TEXT_BYTES {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "native decision encoding byte budget exceeded".into(),
        ));
    }
    Ok(wire)
}

pub(super) fn observe_work(model: &str, prepared: &PreparedExecution, metrics: &Metrics) {
    let body = match &prepared.payload {
        PreparedPayload::Token(prepared) => {
            let branches = prepared.branches();
            metrics.observe_systemone_work(
                model,
                branches.len(),
                branches.iter().map(|b| b.scoring.label_ids.len()).sum(),
            );
            metrics.observe_systemone_input_work(
                model,
                branches.iter().map(|b| b.scoring.prompt_len).sum(),
            );
            return;
        }
        PreparedPayload::Head(prepared) => &prepared.wire.decision_head,
        PreparedPayload::Read(prepared) => &prepared.wire.decision_read,
    };
    metrics.observe_systemone_work(
        model,
        body.questions.len(),
        body.questions.iter().map(|q| q.candidates.len()).sum(),
    );
}

pub(super) async fn dispatch(
    prepared: PreparedExecution,
    request_id: String,
    parent: Arc<dyn AsyncEngineContext>,
    permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
    metric_model: String,
    request: Arc<CanonicalRequest>,
) -> Response {
    let _permit = permit;
    let PreparedExecution { context, payload } = prepared;
    match payload {
        PreparedPayload::Token(prepared) => {
            super::scoring::dispatch(
                &context,
                prepared,
                request_id,
                parent,
                metrics,
                metric_model,
                request,
            )
            .await
        }
        PreparedPayload::Head(prepared) => {
            dispatch_native(&context, prepared, &request_id, parent, &request).await
        }
        PreparedPayload::Read(prepared) => {
            dispatch_native(&context, prepared, &request_id, parent, &request).await
        }
    }
}

async fn dispatch_native<R: super::native::NativeRequest>(
    selection: &DecisionContext,
    prepared: PreparedNativeExecution<R>,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    request: &CanonicalRequest,
) -> Response {
    let result = super::native::run(
        selection,
        &prepared.engine,
        prepared.wire,
        request_id,
        parent.clone(),
        request,
    )
    .await;
    match result {
        Ok((outcomes, usage)) => finish_response(
            request,
            &selection.canonical_model,
            &outcomes,
            &usage,
            request_id,
        ),
        Err(cause) => dispatch_error(request_id, &parent, cause, request.dialect),
    }
}
