// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

#[derive(Debug)]
struct Observed {
    context_tokens: AtomicUsize,
    /// 0 until a choice of the reply finishes, then 1, or 2 once one finished with a tool call.
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
/// Updates are monotonic so concurrent or delayed observations cannot move a
/// request's logical context backwards.
#[derive(Debug, Clone)]
pub struct RequestProgressUpdater {
    observed: Arc<Observed>,
}

impl RequestProgress {
    pub(crate) fn new(initial_context_tokens: usize) -> (Self, RequestProgressUpdater) {
        let observed = Arc::new(Observed {
            context_tokens: AtomicUsize::new(initial_context_tokens),
            reply_end: AtomicU8::new(0),
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
    /// the chat response stream.
    ///
    /// `None` until a choice finishes, which can be shortly after the request's terminal
    /// event, so a plugin keeps this handle past that event to read it. With several choices,
    /// `Some(false)` becomes `Some(true)` if a later one ends in a tool call. It stays `None`
    /// when the reply did not finish, for `/v1/completions`, with a Python chat processor,
    /// and when the router runs outside the frontend process.
    ///
    /// It describes the streamed reply, not its delivery: the client can be gone, or the
    /// endpoint can still fail the request, after the value is set. A non-streaming response
    /// is aggregated from that stream afterwards and can, rarely, end differently.
    #[inline]
    pub fn reply_tool_call(&self) -> Option<bool> {
        match self.observed.reply_end.load(Ordering::Relaxed) {
            0 => None,
            reply_end => Some(reply_end == 2),
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

    /// Record how one choice of the reply ended. A tool call on any choice wins.
    #[inline]
    pub fn record_reply_end(&self, tool_call: bool) {
        self.observed
            .reply_end
            .fetch_max(1 + u8::from(tool_call), Ordering::Relaxed);
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
