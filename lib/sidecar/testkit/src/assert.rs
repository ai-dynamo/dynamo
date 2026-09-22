// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, DynamoError, ErrorType, FinishReason};

use crate::fixtures::Outputs;

pub fn terminal(outputs: Outputs, tokens: &[u32], prompt_tokens: u32, reason: FinishReason) {
    let outputs: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    assert_eq!(
        outputs
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens,
    );
    assert_eq!(
        outputs.iter().filter(|o| o.finish_reason.is_some()).count(),
        1
    );
    let terminal = outputs.last().expect("terminal output");
    assert_eq!(terminal.finish_reason, Some(reason));
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (
            prompt_tokens,
            tokens.len() as u32,
            prompt_tokens + tokens.len() as u32
        ),
    );
}

pub fn failure(mut outputs: Outputs, tokens: &[u32], kind: BackendError) -> DynamoError {
    let error = outputs
        .pop()
        .expect("error output")
        .expect_err("truncation must fail");
    assert_eq!(error.error_type(), ErrorType::Backend(kind));
    let preceding: Vec<_> = outputs.into_iter().collect::<Result<_, _>>().unwrap();
    assert!(preceding.iter().all(|o| o.finish_reason.is_none()));
    assert_eq!(
        preceding
            .iter()
            .flat_map(|o| &o.token_ids)
            .copied()
            .collect::<Vec<_>>(),
        tokens,
    );
    error
}
