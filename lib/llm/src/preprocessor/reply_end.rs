// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A stream that tells the request classifier how a chat reply ended.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dynamo_protocols::types::FinishReason;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// `stream`, unchanged. When it ends, `tracker` records whether any choice finished with
/// `finish_reason: tool_calls`.
///
/// Nothing is recorded for a reply whose end the client did not get: a stream dropped
/// early, one that carried an error, or one that ended without a `finish_reason`.
///
/// Place it after tool-call parsing: the parsers are what turn `stop` into `tool_calls`,
/// and they can do so after the stream below them has ended.
pub(super) fn record_reply_end<S>(stream: S, tracker: Arc<RequestTracker>) -> RecordReplyEnd<S>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin,
{
    RecordReplyEnd {
        stream,
        tracker: Some(tracker),
        tool_call: None,
    }
}

/// The stream of [`record_reply_end`].
pub(super) struct RecordReplyEnd<S> {
    stream: S,
    // Taken when the end is recorded or an error is seen, so it is recorded at most once.
    tracker: Option<Arc<RequestTracker>>,
    // `None` until a choice finishes.
    tool_call: Option<bool>,
}

impl<S> Stream for RecordReplyEnd<S>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let this = &mut *self;
        let poll = this.stream.poll_next_unpin(cx);
        match &poll {
            Poll::Ready(Some(response)) if response.is_error() => this.tracker = None,
            Poll::Ready(Some(response)) => {
                let finished = response
                    .data
                    .iter()
                    .flat_map(|data| &data.inner.choices)
                    .filter_map(|choice| choice.finish_reason);
                for finish_reason in finished {
                    let tool_call = finish_reason == FinishReason::ToolCalls;
                    this.tool_call = Some(tool_call || this.tool_call == Some(true));
                }
            }
            Poll::Ready(None) => {
                if let (Some(tracker), Some(tool_call)) = (this.tracker.take(), this.tool_call) {
                    tracker.record_reply_end(tool_call);
                }
            }
            Poll::Pending => {}
        }
        poll
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}
