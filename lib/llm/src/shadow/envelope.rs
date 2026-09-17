// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The record a shadow tap publishes.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::protocols::TokenIdType;
use crate::protocols::common::preprocessor::PreprocessedRequest;

pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

/// Which pipeline the tap was linked into. The tap sits below the point where
/// the public APIs become one type, so this is as much as it can know: the
/// responses and messages APIs are converted to chat requests above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowOrigin {
    Chat,
    Completions,
    /// A Python chat processor tokenized the request.
    Preprocessed,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowEnvelope {
    pub schema_version: u32,
    pub tap: Arc<str>,
    /// Per tap, per frontend process, assigned before the record is queued.
    /// A missing value is a record the tap dropped because its queue was full.
    /// Concurrent requests can be queued slightly out of `seq` order, so a
    /// consumer counts missing values over a range and does not expect each
    /// record to be the successor of the last. The event-plane sequence is
    /// assigned at publish time and shows transport loss only.
    pub seq: u64,
    /// The id the primary frontend and workers log. A shadow that replays the
    /// request submits it under this id so the two sets of logs join.
    pub request_id: String,
    pub origin: ShadowOrigin,
    /// The filters in effect. Fields they name are absent from `request`.
    pub filters: Arc<[String]>,
    /// When the request reached the tap. The event-plane timestamp is publish
    /// time, which for a joined record is the end of the response stream.
    pub arrival_unix_ns: u64,
    pub request: Arc<PreprocessedRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ShadowResponse>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowOutcome {
    /// The stream ended and no chunk carried an error.
    Complete,
    /// The response stream was dropped before it ended, as on a client
    /// disconnect.
    Cancelled,
    Error,
}

/// What the primary deployment answered, as one consolidated object. One type
/// for both pipelines, which carry `BackendOutput` and `LLMEngineOutput`.
///
/// Inside the pipeline a response is always a stream of chunks, whether or not
/// the client asked for streaming. The tap never forwards chunks to a shadow:
/// it accumulates them and publishes this object one time, when the stream
/// ends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowResponse {
    pub outcome: ShadowOutcome,
    /// One entry per choice that produced a chunk, ordered by index. A request
    /// with `n > 1` interleaves the chunks of its choices in one stream, so
    /// tokens and finish reasons are kept apart by choice.
    pub choices: Vec<ShadowChoice>,
    /// Sum over all choices.
    pub output_tokens: u64,
    /// Offsets from arrival, on a monotonic clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_offset_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_token_offset_ns: Option<u64>,
    pub end_offset_ns: u64,
    /// One offset per chunk that carried tokens. Present when the tap sets
    /// `response.chunk_timing: true`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunk_offsets_ns: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowChoice {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub output_tokens: u64,
    /// Empty when the tap sets `response.tokens: false`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<TokenIdType>,
}
