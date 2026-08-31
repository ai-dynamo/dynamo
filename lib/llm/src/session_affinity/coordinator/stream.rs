// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    pipeline::{ManyOut, ResponseStream},
};
use futures::Stream;

use super::{AffinityLease, AffinityTarget};
use crate::{
    protocols::common::{FinishReason, llm_backend::LLMEngineOutput},
    session_affinity::LlmResponse,
};

pub(super) fn track(
    stream: ManyOut<LlmResponse>,
    lease: AffinityLease,
    rebind: Option<(AffinityTarget, AffinityTarget)>,
) -> ManyOut<LlmResponse> {
    let context = stream.context();
    ResponseStream::new(
        Box::pin(AffinityTrackedStream {
            stream,
            lease: Some(lease),
            rebind,
            failed: false,
        }),
        context,
    )
}

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
    rebind: Option<(AffinityTarget, AffinityTarget)>,
    failed: bool,
}

fn response_failed(item: &LlmResponse) -> bool {
    item.is_error()
        || item
            .data
            .as_ref()
            .and_then(|data: &LLMEngineOutput| data.finish_reason.as_ref())
            .is_some_and(|reason| {
                matches!(reason, FinishReason::Error(_) | FinishReason::Cancelled)
            })
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if self.rebind.is_some() {
                    self.failed |= response_failed(&item);
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                let context = self.stream.context();
                let rebind = self.rebind.take();
                if let Some(mut lease) = self.lease.take() {
                    if !self.failed
                        && !context.is_stopped()
                        && !context.is_killed()
                        && let Some((expected, target)) = rebind
                    {
                        lease.rebind(expected, target);
                    }
                    // Release explicitly so a successful rebind publishes its new version.
                    lease.release();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
