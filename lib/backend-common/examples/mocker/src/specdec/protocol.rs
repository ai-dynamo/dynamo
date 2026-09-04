// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::{EndpointId, WorkerWithDpRank, validate_endpoint_id};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::PROTOCOL;

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_FRAME_BYTES: usize = 64 * 1024;
pub const MAX_PROMPT_TOKENS: usize = 16 * 1024;
pub const MAX_OUTPUT_TOKENS: u32 = 4 * 1024;
pub const MAX_PROPOSAL_TOKENS: usize = 64;

const DIGEST_HEX_LEN: usize = 64;
const MAX_ERROR_MESSAGE_BYTES: usize = 160;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub protocol_version: u16,
    pub request_id: Uuid,
    pub sequence: u64,
    #[serde(flatten)]
    pub message: Message,
}

impl Envelope {
    pub fn new(request_id: Uuid, sequence: u64, message: Message) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            sequence,
            message,
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, ProtocolViolation> {
        self.validate()?;
        let frame = serde_json::to_vec(self)
            .map_err(|_| ProtocolViolation::new("message is not JSON serializable"))?;
        if frame.len() > MAX_FRAME_BYTES {
            return Err(ProtocolViolation::new("frame exceeds protocol limit"));
        }
        Ok(frame)
    }

    pub fn decode(frame: &[u8]) -> Result<Self, ProtocolViolation> {
        if frame.len() > MAX_FRAME_BYTES {
            return Err(ProtocolViolation::new("frame exceeds protocol limit"));
        }
        let envelope: Self = serde_json::from_slice(frame)
            .map_err(|_| ProtocolViolation::new("frame is not a valid protocol message"))?;
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), ProtocolViolation> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(ProtocolViolation::new("unsupported protocol version"));
        }
        self.message.validate()?;
        if let Message::Complete(complete) = &self.message
            && complete.final_sequence != self.sequence
        {
            return Err(ProtocolViolation::new(
                "complete final sequence does not match envelope",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "payload", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Message {
    Hello(Hello),
    HelloAck(HelloAck),
    Start(Start),
    StartAck(StartAck),
    Heartbeat(Heartbeat),
    HeartbeatAck(HeartbeatAck),
    Cleanup(Cleanup),
    CleanupAck(CleanupAck),
    Proposal(Proposal),
    Complete(Complete),
    Error(ErrorPayload),
}

impl Message {
    fn validate(&self) -> Result<(), ProtocolViolation> {
        match self {
            Self::Hello(message) => message.expected.validate(),
            Self::HelloAck(message) => message.identity.validate(),
            Self::Start(message) => message.validate(),
            Self::Proposal(message) => message.validate(),
            Self::Complete(message) => validate_digest(&message.proposal_digest),
            Self::Error(message) => message.validate(),
            Self::StartAck(_)
            | Self::Heartbeat(_)
            | Self::HeartbeatAck(_)
            | Self::Cleanup(_)
            | Self::CleanupAck(_) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    pub expected: DraftIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HelloAck {
    pub identity: DraftIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Start {
    pub prompt_token_ids: Vec<u32>,
    pub max_output_tokens: u32,
}

impl Start {
    fn validate(&self) -> Result<(), ProtocolViolation> {
        if self.prompt_token_ids.is_empty() {
            return Err(ProtocolViolation::new(
                "prompt token list must not be empty",
            ));
        }
        if self.prompt_token_ids.len() > MAX_PROMPT_TOKENS {
            return Err(ProtocolViolation::new(
                "prompt token count exceeds protocol limit",
            ));
        }
        if !(1..=MAX_OUTPUT_TOKENS).contains(&self.max_output_tokens) {
            return Err(ProtocolViolation::new(
                "output token count exceeds protocol limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct StartAck {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct Heartbeat {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct HeartbeatAck {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct Cleanup {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct CleanupAck {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub token_ids: Vec<u32>,
}

impl Proposal {
    fn validate(&self) -> Result<(), ProtocolViolation> {
        if self.token_ids.is_empty() {
            return Err(ProtocolViolation::new(
                "proposal must contain at least one token",
            ));
        }
        if self.token_ids.len() > MAX_PROPOSAL_TOKENS {
            return Err(ProtocolViolation::new(
                "proposal token count exceeds protocol limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Complete {
    pub final_sequence: u64,
    pub proposal_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct DraftIdentity {
    pub endpoint: EndpointId,
    pub worker: WorkerWithDpRank,
    pub draft_incarnation_id: u64,
    pub protocol: String,
    pub address: String,
    pub orphan_cleanup_timeout_ms: u32,
}

impl DraftIdentity {
    pub fn validate(&self) -> Result<(), ProtocolViolation> {
        validate_endpoint_id(&self.endpoint)
            .map_err(|_| ProtocolViolation::new("draft endpoint identity is invalid"))?;
        if self.protocol != PROTOCOL {
            return Err(ProtocolViolation::new("draft protocol identity mismatch"));
        }
        if self.draft_incarnation_id == 0 {
            return Err(ProtocolViolation::new("draft incarnation must be positive"));
        }
        if self.address.is_empty()
            || self.address.len() > 512
            || self.address.chars().any(char::is_control)
        {
            return Err(ProtocolViolation::new("draft address is invalid"));
        }
        if !(1..=300_000).contains(&self.orphan_cleanup_timeout_ms) {
            return Err(ProtocolViolation::new("draft cleanup bound is invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureState {
    NotStarted,
    Accepted,
    Ambiguous,
    Cancelled,
    ProtocolInvalid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidMessage,
    IdentityMismatch,
    QueueFull,
    DuplicateRequest,
    UnknownRequest,
    Cancelled,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ErrorPayload {
    pub code: ErrorCode,
    pub state: FailureState,
    pub message: String,
}

impl ErrorPayload {
    pub fn new(code: ErrorCode, state: FailureState) -> Self {
        let message = match code {
            ErrorCode::InvalidMessage => "protocol message rejected",
            ErrorCode::IdentityMismatch => "draft identity rejected",
            ErrorCode::QueueFull => "draft queue is full",
            ErrorCode::DuplicateRequest => "request is already active",
            ErrorCode::UnknownRequest => "request is not active",
            ErrorCode::Cancelled => "request was cancelled",
            ErrorCode::Internal => "draft transport failed",
        };
        Self {
            code,
            state,
            message: message.to_string(),
        }
    }

    fn validate(&self) -> Result<(), ProtocolViolation> {
        if self.message.is_empty()
            || self.message.len() > MAX_ERROR_MESSAGE_BYTES
            || self.message.chars().any(char::is_control)
        {
            return Err(ProtocolViolation::new("protocol error text is invalid"));
        }
        if self.message != Self::new(self.code, self.state).message {
            return Err(ProtocolViolation::new(
                "protocol error text is not canonical",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolViolation {
    message: &'static str,
}

impl ProtocolViolation {
    fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl fmt::Display for ProtocolViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for ProtocolViolation {}

#[derive(Debug, Clone)]
pub struct SequenceValidator {
    next: u64,
}

impl SequenceValidator {
    pub fn starting_at(first: u64) -> Self {
        Self { next: first }
    }

    pub fn observe(&mut self, sequence: u64) -> Result<(), ProtocolViolation> {
        if sequence != self.next {
            return Err(ProtocolViolation::new(
                "message sequence is duplicated or out of order",
            ));
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| ProtocolViolation::new("message sequence overflow"))?;
        Ok(())
    }

    pub fn next(&self) -> u64 {
        self.next
    }
}

pub fn proposal_digest(token_ids: &[u32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for token in token_ids {
        hasher.update(&token.to_be_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn validate_digest(value: &str) -> Result<(), ProtocolViolation> {
    if value.len() != DIGEST_HEX_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(ProtocolViolation::new("proposal digest is invalid"));
    }
    Ok(())
}
