// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared protocol types for KVBM conditional disaggregation.
//!
//! This module intentionally contains only serializable control-plane data. It
//! is shared by the connector, hub, and future admission code without making
//! any one of those crates depend on another.

use dynamo_tokens::{TokenBlockMmInfo, compute_hash_v2};
use kvbm_common::SequenceHash;
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

/// xxh3_64 seed for `expected_hash_digest`. Distinct from
/// `CHAIN_XXH3_SEED` so an accidental collision with a PLH chain hash
/// can't sneak past the verifier.
pub const CD_DIGEST_SEED: u64 = 0xCD_CD_CD_CD_CD_CD_CD_CD;

/// Compute the defense-in-depth digest of a contiguous PLH slice.
///
/// Decode calls this over the `[0, DNPT/block_size)` slice of its slot's
/// PLH chain (everything decode commits to serve); prefill calls it over
/// the same slice of its own slot's locally-recomputed PLH chain. The
/// decode-side digest rides on the wire as
/// `RemotePrefillRequest.expected_hash_digest`; the prefill-side verifier
/// asserts equality before accepting the dispatch. Any mismatch indicates
/// hashing-input divergence (missed salt/LoRA propagation, hasher version
/// skew, etc.) — fail loud here instead of letting prefill publish
/// unrelated hashes and decode hang at the RDMA pull.
///
/// Stable across builds: xxh3_64 over the big-endian byte representation
/// of each PLH's inner `u128`, prefixed by the slice length so a longer
/// vs. shorter slice with a common prefix produces distinct digests.
pub fn digest_provided_hashes(hashes: &[SequenceHash]) -> u64 {
    let mut buf = Vec::with_capacity(8 + hashes.len() * 16);
    buf.extend_from_slice(&(hashes.len() as u64).to_be_bytes());
    for h in hashes {
        buf.extend_from_slice(&h.as_u128().to_be_bytes());
    }
    compute_hash_v2(&buf, CD_DIGEST_SEED)
}

/// Current disaggregation protocol version.
pub const DISAGG_PROTOCOL_VERSION: u16 = 1;

/// Unique identifier for a disaggregation session.
pub type SessionId = uuid::Uuid;

/// JSON-safe representation of a KVBM sequence hash.
///
/// Native KVBM hashes are currently backed by `u128`, which `serde_json` does
/// not support directly. The wire protocol carries the decimal representation.
pub type DisaggSequenceHash = String;

mod serde_uuid_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;

    pub fn serialize<S>(id: &Uuid, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Uuid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value).map_err(D::Error::custom)
    }
}

mod serde_instance_id_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;
    use velo_ext::InstanceId;

    pub fn serialize<S>(id: &InstanceId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InstanceId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value)
            .map(InstanceId::from)
            .map_err(D::Error::custom)
    }
}

/// Role advertised by a worker or instance to the hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    Decode,
    Prefill,
    Hybrid,
}

/// Opaque endpoint descriptor for a session/control channel.
///
/// Later PRs can standardize concrete endpoint kinds such as velo-streaming
/// anchors without changing where protocol ownership lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEndpoint {
    pub kind: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub payload: serde_json::Value,
}

/// Typed transfer parameters carried in request metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_prefill: Option<RemotePrefillParams>,
}

impl TransferParams {
    pub fn remote_prefill(params: RemotePrefillParams) -> Self {
        Self {
            remote_prefill: Some(params),
        }
    }
}

/// Wire-shape mirror of `kv_hashing::Request`'s hashing inputs.
///
/// `tokens` is carried separately on [`RemotePrefillRequest::token_ids`] (it is
/// also the vLLM dispatch payload). This envelope carries the remaining inputs
/// — `lora_name`, `salt`, `mm_info` — so the prefill side can rebuild the
/// canonical [`kv_hashing::Request`] (field-for-field) and run the same Rust
/// hasher locally instead of trusting decode-shipped `PositionalLineageHash`
/// values. The result: a single source of truth for hashing, no PLH bytes on
/// the wire, and forward-compatible coverage for salt / LoRA / multimodal.
///
/// **Current scope:** decode-side construction asserts each field is empty
/// (no salt, no LoRA, no multimodal yet wired through the CD path). The
/// fields exist so future enablement is a value change, not a wire-format
/// change.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHashingRequestEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub salt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mm_info: Vec<TokenBlockMmInfo>,
}

impl KvHashingRequestEnvelope {
    /// True iff no hashing-affecting auxiliary input is set.
    pub fn is_empty(&self) -> bool {
        self.lora_name.is_none() && self.salt.is_none() && self.mm_info.is_empty()
    }
}

/// Parameters identifying a remote prefill session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillParams {
    pub protocol_version: u16,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    /// **DNPT — Decode's Num Provided Tokens.** Contiguous tokens decode
    /// commits to serve via the session, counted from absolute position 0.
    /// Folds in both the vLLM-decode G1 prefix AND the G2 local-match
    /// window decode resolved during this request. By construction
    /// `DNPT >= D_vllm_decode`; decode may over-commit by promoting
    /// G1→G2 or staging G3→G2 but never under-commits relative to what
    /// vLLM-decode declared computed.
    ///
    /// **Prefill's contract:** prefill provides all full-token blocks
    /// for `[DNPT/block_size, ..)` (everything decode doesn't have,
    /// prefill produces). Prefill independently computes PNCT (its
    /// own local cache view) and pulls hashes `[PNCT, DNPT)` from
    /// decode when `PNCT < DNPT`. If `PNCT >= DNPT`, prefill skips
    /// the prefix pull; a future enhancement is for prefill to
    /// publish back `[DNPT, PNCT)` to the same session so decode
    /// can pull and warm its own cache.
    #[serde(default)]
    pub num_provided_tokens: usize,
    /// Canonical hashing inputs decode used to build its `kv_hashing::Request`.
    /// Prefill rebuilds the same request from these fields + `token_ids` so
    /// the local PLH chain is bit-identical by construction.
    #[serde(default)]
    pub request: KvHashingRequestEnvelope,
    /// xxh3_64 digest of decode's `[0, DNPT/block_size)` PLH slice.
    /// Prefill recomputes the digest from its own slot's PLH slice and
    /// asserts equality before accepting the dispatch. Defense-in-depth
    /// against silent hash divergence (a missed salt/LoRA propagation,
    /// hasher version skew, etc.); cost is one u64 on the wire. `None`
    /// skips the assertion (legacy tests and paths that don't compute it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash_digest: Option<u64>,
}

impl RemotePrefillParams {
    pub fn new(session_id: SessionId, initiator_instance_id: InstanceId) -> Self {
        Self {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id,
            initiator_instance_id,
            decode_endpoint: None,
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
        }
    }
}

/// Payload enqueued by a decode worker and consumed by a prefill worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillRequest {
    pub protocol_version: u16,
    pub request_id: String,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    /// Full prompt token IDs (NOT a windowed suffix). Prefill builds its
    /// slot from these so its PLH chain matches decode's at absolute
    /// positions — required for cross-instance pull keys to align.
    pub token_ids: Vec<u32>,
    /// See [`RemotePrefillParams::num_provided_tokens`].
    #[serde(default)]
    pub num_provided_tokens: usize,
    #[serde(default)]
    pub request: KvHashingRequestEnvelope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash_digest: Option<u64>,
}

impl RemotePrefillRequest {
    pub fn remote_prefill_params(&self) -> RemotePrefillParams {
        RemotePrefillParams {
            protocol_version: self.protocol_version,
            session_id: self.session_id,
            initiator_instance_id: self.initiator_instance_id,
            decode_endpoint: self.decode_endpoint.clone(),
            num_provided_tokens: self.num_provided_tokens,
            request: self.request.clone(),
            expected_hash_digest: self.expected_hash_digest,
        }
    }

    pub fn transfer_params(&self) -> TransferParams {
        TransferParams::remote_prefill(self.remote_prefill_params())
    }
}

/// Hub-issued lifecycle/control signal for decode or prefill workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlSignal {
    Pause {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Resume,
    Drain,
    ShutdownGracefully,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance_id() -> InstanceId {
        uuid::Uuid::new_v4().into()
    }

    #[test]
    fn remote_prefill_request_builds_transfer_params() {
        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "req-1".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: Some(SessionEndpoint {
                kind: "test".to_string(),
                payload: serde_json::json!({"anchor": "a"}),
            }),
            token_ids: vec![1, 2, 3],
            num_provided_tokens: 48,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: Some(0xDEADBEEF),
        };

        let params = request
            .transfer_params()
            .remote_prefill
            .expect("remote params populated");
        assert_eq!(params.protocol_version, request.protocol_version);
        assert_eq!(params.session_id, request.session_id);
        assert_eq!(params.initiator_instance_id, request.initiator_instance_id);
        assert_eq!(params.decode_endpoint, request.decode_endpoint);
        assert_eq!(params.num_provided_tokens, request.num_provided_tokens);
        assert_eq!(params.request, request.request);
        assert_eq!(params.expected_hash_digest, request.expected_hash_digest);
    }

    #[test]
    fn transfer_params_round_trips_json() {
        let params = TransferParams::remote_prefill(RemotePrefillParams {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: None,
            num_provided_tokens: 48,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: Some(0x1234_5678_9ABC_DEF0),
        });

        let encoded = serde_json::to_vec(&params).unwrap();
        let decoded: TransferParams = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, params);
    }

    // Defense-in-depth digest reproducer-first tests. The digest helper
    // did not exist before this refactor — these would fail to compile
    // against pre-fix HEAD, which is the strongest form of "reproducer
    // catches the regression" since the verifier itself is what closes
    // the silent-hash-divergence loop.
    fn fake_plh(seed: u64, pos: u64) -> SequenceHash {
        SequenceHash::new(seed, None, pos)
    }

    #[test]
    fn digest_is_deterministic() {
        let hashes = vec![fake_plh(1, 0), fake_plh(2, 1), fake_plh(3, 2)];
        let a = digest_provided_hashes(&hashes);
        let b = digest_provided_hashes(&hashes);
        assert_eq!(a, b, "digest must be deterministic for identical input");
    }

    #[test]
    fn digest_distinguishes_different_slices() {
        let a = digest_provided_hashes(&[fake_plh(1, 0), fake_plh(2, 1)]);
        let b = digest_provided_hashes(&[fake_plh(1, 0), fake_plh(99, 1)]);
        assert_ne!(a, b, "differing PLHs must produce distinct digests");
    }

    #[test]
    fn digest_distinguishes_length_prefix() {
        // The length prefix in the digest input guarantees that a
        // 2-element slice and a 3-element slice sharing the same prefix
        // produce distinct digests — guards against an off-by-one slicing
        // bug on either side silently passing the verifier.
        let two = vec![fake_plh(1, 0), fake_plh(2, 1)];
        let three = vec![fake_plh(1, 0), fake_plh(2, 1), fake_plh(3, 2)];
        assert_ne!(digest_provided_hashes(&two), digest_provided_hashes(&three));
    }

    #[test]
    fn digest_empty_slice_is_stable() {
        let a = digest_provided_hashes(&[]);
        let b = digest_provided_hashes(&[]);
        assert_eq!(a, b);
        // And distinct from any single-element digest.
        assert_ne!(a, digest_provided_hashes(&[fake_plh(0, 0)]));
    }

    #[test]
    fn envelope_round_trips_with_mm_info() {
        let env = KvHashingRequestEnvelope {
            lora_name: Some("adapter-x".into()),
            salt: Some("model-tag".into()),
            mm_info: vec![TokenBlockMmInfo {
                mm_hash: 0xCAFE,
                offset: 4,
                length: 8,
            }],
        };
        let encoded = serde_json::to_vec(&env).unwrap();
        let decoded: KvHashingRequestEnvelope = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, env);
        assert!(!env.is_empty());
        assert!(KvHashingRequestEnvelope::default().is_empty());
    }

    #[test]
    fn control_signal_round_trips_json() {
        let signal = ControlSignal::Pause {
            reason: Some("operator".to_string()),
        };

        let encoded = serde_json::to_string(&signal).unwrap();
        let decoded: ControlSignal = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, signal);
    }
}
