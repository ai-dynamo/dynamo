// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session state and upstream transport, independent of the Dynamo frontend,
//! worker bootstrap, and inference engines. Model preprocessing stays upstream.

mod audio;
mod client;
mod events;
mod session;
pub mod speech;

#[cfg(test)]
mod tests;

pub use client::Client;
pub use session::Session;

use std::num::NonZeroUsize;
use std::time::Duration;

// Reserve enough space to report a terminal failure when rejecting a data event.
const MIN_OUTPUT_BUFFER_BYTES: usize = 1024;

/// The upstream wire contract, selected explicitly rather than from model names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// vLLM's transcription WebSocket: 16 kHz PCM and transcription.* events.
    VllmTranscription,
    /// vLLM-Omni's OpenAI session API, with independent, ordered audio turns.
    Omni,
}

/// Bounds apply per session/turn, including when clients or upstreams stall.
#[derive(Clone, Debug)]
pub struct Limits {
    pub concurrent_turns: NonZeroUsize,
    pub queued_turns: usize,
    pub event_capacity: NonZeroUsize,
    pub max_audio_chunk_bytes: NonZeroUsize,
    pub max_turn_audio_bytes: NonZeroUsize,
    pub max_buffered_output_bytes: NonZeroUsize,
    pub connect_timeout: Duration,
    pub response_timeout: Duration,
    pub close_timeout: Duration,
}

impl Limits {
    /// Defaults preserve the Python handler's turn concurrency and 60-second
    /// audio limit. The output byte budget is per turn, in addition to the event
    /// count limit. Limits are configurable by the embedding application.
    pub fn for_protocol(protocol: Protocol) -> Self {
        Self {
            concurrent_turns: NonZeroUsize::new(match protocol {
                Protocol::VllmTranscription => 1,
                Protocol::Omni => 8,
            })
            .unwrap(),
            queued_turns: usize::from(protocol == Protocol::VllmTranscription),
            event_capacity: NonZeroUsize::new(64).unwrap(),
            max_audio_chunk_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
            max_turn_audio_bytes: NonZeroUsize::new(audio::OPENAI_PCM_RATE * 2 * 60).unwrap(),
            max_buffered_output_bytes: NonZeroUsize::new(8 * 1024 * 1024).unwrap(),
            connect_timeout: Duration::from_secs(30),
            response_timeout: Duration::from_secs(120),
            close_timeout: Duration::from_secs(5),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        let now = std::time::Instant::now();
        if self
            .concurrent_turns
            .get()
            .checked_add(self.queued_turns)
            .is_none()
            || self.concurrent_turns.get() > tokio::sync::Semaphore::MAX_PERMITS
            || self.event_capacity.get() > tokio::sync::Semaphore::MAX_PERMITS
            || self.max_buffered_output_bytes.get() < MIN_OUTPUT_BUFFER_BYTES
            || self.max_buffered_output_bytes.get() > u32::MAX as usize
            || self.max_buffered_output_bytes.get() > tokio::sync::Semaphore::MAX_PERMITS
            || self.connect_timeout.is_zero()
            || self.response_timeout.is_zero()
            || self.close_timeout.is_zero()
            || [
                self.connect_timeout,
                self.response_timeout,
                self.close_timeout,
            ]
            .iter()
            .any(|&timeout| now.checked_add(timeout).is_none())
        {
            return Err(Error::Invalid("invalid session limits".into()));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("upstream protocol error: {0}")]
    Protocol(String),
    #[error("upstream request timed out")]
    Timeout,
    #[error("session closed")]
    Closed,
    #[error("upstream WebSocket: {0}")]
    WebSocket(#[source] Box<tokio_tungstenite::tungstenite::Error>),
    #[error("upstream HTTP: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("audio conversion: {0}")]
    Audio(String),
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(error: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::WebSocket(Box::new(error))
    }
}
