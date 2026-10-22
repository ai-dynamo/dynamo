// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::DecisionExecutor,
    protocols::decision_read::{DecisionReadRequest, DecisionReadResponse},
};

pub(super) async fn run(
    selection: &SystemOneExecutionSelection,
    wire: DecisionReadRequest,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    request: &CanonicalRequest,
) -> Result<(Vec<QuestionOutcome>, Usage), DispatchError> {
    selection.validate_dispatch(None).map_err(|_| {
        DispatchError::unavailable("selected native decision revision is unavailable")
    })?;
    let DecisionExecutor::NativeRead { engine } = &selection.executor else {
        return Err(DispatchError::malformed("invalid native read executor"));
    };
    let max_input_tokens = wire.decision_read.max_input_tokens;
    let context =
        Context::with_id_and_metadata(wire, format!("{request_id}-read"), Default::default());
    parent.link_child(context.context());
    if parent.is_killed() {
        context.context().kill();
        return Err(DispatchError::cancelled());
    }
    let mut stream = run_until_killed(parent.as_ref(), engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| super::head::backend_error(cause.as_ref()))?;
    let stream_context = stream.context();
    parent.link_child(stream_context.clone());
    if parent.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    let mut terminal: Option<DecisionReadResponse> = None;
    loop {
        let frame = run_until_killed(parent.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        if let Some(response) = frame
            .into_data()
            .map_err(|cause| super::head::backend_error(&cause))?
        {
            if terminal.replace(response).is_some() {
                return Err(DispatchError::malformed(
                    "multiple terminal native read responses",
                ));
            }
        }
    }
    if parent.is_killed() {
        return Err(DispatchError::cancelled());
    }
    terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal native read response"))?
        .validate(request, selection.profile, max_input_tokens)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))
}
