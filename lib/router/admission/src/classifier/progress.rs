// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const REPLY_END_UNKNOWN: u8 = 0;
const REPLY_END_OTHER: u8 = 1;
const REPLY_END_TOOL_CALL: u8 = 2;

#[derive(Debug)]
struct Observed {
    context_tokens: AtomicUsize,
    reply_end: AtomicU8,
}

/// Lock-free access to what the host has observed for one request: its latest logical
/// context and how its reply ended.
#[derive(Debug, Clone)]
pub struct RequestProgress {
    observed: Arc<Observed>,
}

/// Write capability paired with [`RequestProgress`].
///
/// Context updates are monotonic so concurrent or delayed observations cannot move a
/// request's logical context backwards.
#[derive(Debug, Clone)]
pub struct RequestProgressUpdater {
    observed: Arc<Observed>,
}

impl RequestProgress {
    pub fn new(initial_context_tokens: usize) -> (Self, RequestProgressUpdater) {
        let observed = Arc::new(Observed {
            context_tokens: AtomicUsize::new(initial_context_tokens),
            reply_end: AtomicU8::new(REPLY_END_UNKNOWN),
        });
        (
            Self {
                observed: Arc::clone(&observed),
            },
            RequestProgressUpdater { observed },
        )
    }

    #[inline]
    pub fn context_tokens(&self) -> usize {
        self.observed.context_tokens.load(Ordering::Relaxed)
    }

    /// Whether the reply ended in a tool call: `finish_reason: tool_calls` on any choice of
    /// the chat response stream. A non-streaming response is aggregated from that stream
    /// afterwards and can, rarely, end differently.
    ///
    /// `None` until the frontend has relayed the end of the reply, which is shortly after
    /// the request's terminal event, so a plugin keeps this handle past that event to read
    /// it. It stays `None` when the frontend did not see how the reply ended: an error, a
    /// client disconnect, `/v1/completions`, a Python chat processor, or a router running
    /// outside the frontend process.
    #[inline]
    pub fn reply_tool_call(&self) -> Option<bool> {
        match self.observed.reply_end.load(Ordering::Relaxed) {
            REPLY_END_TOOL_CALL => Some(true),
            REPLY_END_OTHER => Some(false),
            _ => None,
        }
    }
}

impl RequestProgressUpdater {
    #[inline]
    pub fn update_context_tokens(&self, context_tokens: usize) {
        self.observed
            .context_tokens
            .fetch_max(context_tokens, Ordering::Relaxed);
    }

    /// Record how the reply ended. The host calls this once per request, after the
    /// frontend has relayed the end of the reply.
    #[inline]
    pub fn record_reply_end(&self, tool_call: bool) {
        let reply_end = if tool_call {
            REPLY_END_TOOL_CALL
        } else {
            REPLY_END_OTHER
        };
        self.observed.reply_end.store(reply_end, Ordering::Relaxed);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_progress_is_monotonic() {
        let (progress, updater) = RequestProgress::new(42);

        updater.update_context_tokens(55);
        updater.update_context_tokens(50);

        assert_eq!(progress.context_tokens(), 55);
    }
}
