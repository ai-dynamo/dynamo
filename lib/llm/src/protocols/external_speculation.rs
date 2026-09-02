// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned worker protocol for externally hosted speculative decoding.

use anyhow::ensure;
use rand::TryRngCore;
use serde::{Deserialize, Serialize};

pub use dynamo_kv_router::router_hint::{
    DraftTransport, RouterHintEnvelope, SpeculativeDecodingRouterHintV1, validate_endpoint_id,
    validate_transport_address,
};

/// Maximum disconnect-driven cleanup bound accepted from a draft worker.
pub const MAX_ORPHAN_CLEANUP_TIMEOUT_MS: u32 = 300_000;
pub const EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY: &str = "_dynamo_external_speculation_v1";

const MAX_PROTOCOL_LEN: usize = 128;
const JSON_SAFE_RANDOM_MASK: u64 = (1_u64 << 53) - 1;

/// Live per-rank draft transport advertised by one worker lifetime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DraftTransportDescriptorV1 {
    pub protocol: String,
    pub address: String,
    pub draft_incarnation_id: u64,
    pub orphan_cleanup_timeout_ms: u32,
}

impl DraftTransportDescriptorV1 {
    pub fn validate(&self) -> anyhow::Result<()> {
        validate_protocol(&self.protocol)?;
        validate_transport_address(&self.address).map_err(anyhow::Error::msg)?;
        ensure!(
            (1..=JSON_SAFE_RANDOM_MASK).contains(&self.draft_incarnation_id),
            "draft_incarnation_id must be a positive JSON-safe integer"
        );
        ensure!(
            (1..=MAX_ORPHAN_CLEANUP_TIMEOUT_MS).contains(&self.orphan_cleanup_timeout_ms),
            "orphan_cleanup_timeout_ms must be between 1 and {MAX_ORPHAN_CLEANUP_TIMEOUT_MS}"
        );
        Ok(())
    }

    pub fn router_transport(&self) -> DraftTransport {
        DraftTransport {
            protocol: self.protocol.clone(),
            address: self.address.clone(),
            orphan_cleanup_timeout_ms: self.orphan_cleanup_timeout_ms,
        }
    }
}

/// Target-reported proof that retaining the draft reservation is no longer necessary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DraftCleanupOutcomeV1 {
    Acknowledged,
    CleanupBoundElapsed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalSpeculationLifecycleV1 {
    pub schema_version: u16,
    pub draft_cleanup: DraftCleanupOutcomeV1,
}

impl ExternalSpeculationLifecycleV1 {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == Self::SCHEMA_VERSION,
            "unsupported external-speculation lifecycle schema version {}",
            self.schema_version
        );
        Ok(())
    }
}

pub fn new_external_speculation_incarnation() -> anyhow::Result<u64> {
    let mut rng = rand::rngs::OsRng;
    let value = rng.try_next_u64().map_err(|error| {
        anyhow::anyhow!("failed to generate external-speculation incarnation: {error}")
    })?;
    Ok((value & JSON_SAFE_RANDOM_MASK).max(1))
}

pub fn validate_protocol(value: &str) -> anyhow::Result<()> {
    ensure!(!value.is_empty(), "protocol must not be empty");
    ensure!(
        value.len() <= MAX_PROTOCOL_LEN,
        "protocol exceeds the {MAX_PROTOCOL_LEN}-byte limit"
    );
    ensure!(
        value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/')
        }),
        "protocol contains unsupported characters"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_descriptor_validates_wire_bounds() {
        let descriptor = DraftTransportDescriptorV1 {
            protocol: "mock-specdec-zmq-v1".into(),
            address: "tcp://draft:50051".into(),
            draft_incarnation_id: 7,
            orphan_cleanup_timeout_ms: 1_000,
        };
        descriptor.validate().unwrap();

        let mut invalid = descriptor.clone();
        invalid.draft_incarnation_id = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = descriptor;
        invalid.orphan_cleanup_timeout_ms = MAX_ORPHAN_CLEANUP_TIMEOUT_MS + 1;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn generated_incarnations_are_positive_json_safe_integers() {
        for _ in 0..32 {
            let value = new_external_speculation_incarnation().unwrap();
            assert!((1..=JSON_SAFE_RANDOM_MASK).contains(&value));
        }
    }
}
