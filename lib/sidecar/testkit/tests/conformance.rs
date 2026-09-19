// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{BackendError, FinishReason, GenerateContext, LLMEngine};
use dynamo_sidecar_testkit::assert::{failure, terminal};
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{
    Controller, Event, OpenAction, RequestPlan, StreamAction, StreamFault, StreamPoint,
};
use dynamo_sidecar_testkit::fixtures::{Outputs, collect, request};
use futures::{StreamExt, poll};

mod support;

use support::{FixtureConfig, SidecarFixture, vllm};

fn after_tokens(count: usize, action: StreamAction) -> RequestPlan {
    RequestPlan {
        stream: Some(StreamFault {
            at: StreamPoint::TokenResponse(count),
            action,
            pause: true,
        }),
        ..Default::default()
    }
}

async fn drained(fixture: &impl SidecarFixture) {
    bounded("Mocker request release", async {
        while fixture.active_request_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
}

async fn healthy(
    fixture: &impl SidecarFixture,
    engine: &impl LLMEngine,
    control: &Controller<impl dynamo_sidecar_testkit::control::Protocol>,
) {
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    let outputs = collect(
        engine,
        request("mocker-model", vec![11, 22, 33], 3),
        GenerateContext::new(ctx, None),
    )
    .await;
    terminal(outputs, &handle.tokens(), 3, FinishReason::Length);
    assert_eq!(handle.tokens().len(), 3);
    bounded("healthy remote drop", handle.wait(Event::Dropped)).await;
    fixture.scheduler_idle().await;
}

#[path = "conformance/cancellation.rs"]
mod cancellation;
#[path = "conformance/errors.rs"]
mod errors;
#[path = "conformance/lifecycle.rs"]
mod lifecycle;
#[path = "conformance/streaming.rs"]
mod streaming;
