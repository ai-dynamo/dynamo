// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::DecisionContext;
use crate::protocols::{
    decision_head::{DecisionHeadRequest, HeadOutcome},
    decision_native::{NativeOutcome, NativeResponse},
    decision_read::{DecisionReadRequest, ReadOutcome},
};
use dynamo_runtime::{
    engine::{AsyncEngine, Data},
    pipeline::{Error, SingleIn},
};

pub(super) trait NativeRequest: Data {
    type Outcome: NativeOutcome + Data;
    const SUFFIX: &'static str;
    fn wrap(body: crate::protocols::decision_native::NativeDecisionRequestV1) -> Self;
    fn body(&self) -> &crate::protocols::decision_native::NativeDecisionRequestV1;
}

impl NativeRequest for DecisionHeadRequest {
    type Outcome = HeadOutcome;
    const SUFFIX: &'static str = "head";
    fn wrap(body: crate::protocols::decision_native::NativeDecisionRequestV1) -> Self {
        Self {
            decision_head: body,
        }
    }
    fn body(&self) -> &crate::protocols::decision_native::NativeDecisionRequestV1 {
        &self.decision_head
    }
}

impl NativeRequest for DecisionReadRequest {
    type Outcome = ReadOutcome;
    const SUFFIX: &'static str = "read";
    fn wrap(body: crate::protocols::decision_native::NativeDecisionRequestV1) -> Self {
        Self {
            decision_read: body,
        }
    }
    fn body(&self) -> &crate::protocols::decision_native::NativeDecisionRequestV1 {
        &self.decision_read
    }
}

pub(super) type NativeEngine<R> = Arc<
    dyn AsyncEngine<
            SingleIn<R>,
            ManyOut<Annotated<NativeResponse<<R as NativeRequest>::Outcome>>>,
            Error,
        >,
>;

pub(super) async fn run<R: NativeRequest>(
    selection: &DecisionContext,
    engine: &NativeEngine<R>,
    wire: R,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    request: &CanonicalRequest,
) -> Result<(Vec<QuestionOutcome>, Usage), DispatchError> {
    selection.validate_dispatch(None).map_err(|_| {
        DispatchError::unavailable("selected native decision revision is unavailable")
    })?;
    let max_input_tokens = wire.body().max_input_tokens;
    let mut context = Context::with_id_and_metadata(
        wire,
        format!("{request_id}-{}", R::SUFFIX),
        [(
            crate::protocols::systemone::DECISION_DISPATCH_METADATA_KEY.to_owned(),
            R::SUFFIX.to_owned(),
        )]
        .into(),
    );
    selection.attach_dispatch_guard(&mut context);
    let mut stream = decision_lifecycle::start_linked(parent.as_ref(), context, engine.as_ref())
        .await
        .map_err(backend_error::lifecycle_error)?;
    let mut terminal = None;
    while let Some(response) = decision_lifecycle::next_frame(parent.as_ref(), &mut stream)
        .await
        .map_err(backend_error::lifecycle_error)?
    {
        if terminal.replace(response).is_some() {
            return Err(DispatchError::invalid_executor(
                "multiple terminal native decision responses",
            ));
        }
    }
    if parent.is_killed() {
        return Err(DispatchError::cancelled());
    }
    terminal
        .ok_or_else(|| {
            DispatchError::invalid_executor("missing terminal native decision response")
        })?
        .validate(request, selection.profile(), max_input_tokens)
        .map_err(|cause| DispatchError::invalid_executor(cause.to_string()))
}
