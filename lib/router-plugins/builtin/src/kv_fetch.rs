// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `kv_fetch` hint policy and KVCR's `kv.fetch` action format.
//!
//! When the worker-selection policy committed the booked worker to a KV copy, as `kvcr` does,
//! this policy sends that copy. The router drops a committed copy whose source left before the
//! booking, with a warning, so this policy never sees it. Otherwise it fetches best effort from
//! the booked worker's copy source when the fetch adds at least `min_additional_blocks`, and
//! skips a fetch whose source has become unreachable.

use std::collections::BTreeMap;
use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

use crate::KVCR_POLICY_TYPE;

/// Hint-policy type listed as `worker_selection.<role>.hints[].type`.
pub const POLICY_TYPE: &str = "kv_fetch";

/// KVCR's point-to-point fetch action.
pub const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
/// The `kv.fetch` payload version this policy sends.
pub const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// The `kv.fetch@1.0` action that carries `payload`. The router numbers it when it composes the
/// hint.
fn fetch_action(payload: KvSourceLocationsPayload) -> KvHintAction {
    let KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes,
    } = payload;
    KvHintAction::unnumbered(
        KV_FETCH_ACTION_TYPE,
        KV_FETCH_ACTION_VERSION,
        BTreeMap::from([
            (
                "source_control_endpoint".to_string(),
                serde_json::json!(source_control_endpoint),
            ),
            ("block_hashes".to_string(), serde_json::json!(block_hashes)),
        ]),
    )
}

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Blocks a best-effort fetch must add beyond the booked worker's device and host-pinned
    /// prefix. Committed copies are always sent.
    min_additional_blocks: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            min_additional_blocks: 1,
        }
    }
}

struct KvFetch {
    min_additional_blocks: usize,
}

impl HintPolicy for KvFetch {
    // The booked worker's copy source, for copies the selection didn't commit.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::KV_COPY
    }

    // The selection's committed copies, which this policy sends.
    fn commitments(&self) -> Commitments {
        Commitments::KV_COPY
    }

    fn build_hints(&self, context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintPolicyError> {
        // The selection's cost assumed this copy: always send it.
        if let Some(copy) = context.committed_kv_copy() {
            return Ok(vec![kvcr_fetch(context, copy)?]);
        }
        // Otherwise send a fetch only when it adds enough blocks.
        Ok(context
            .kv_copy_source()
            .filter(|source| source.additional_blocks() >= self.min_additional_blocks)
            .and_then(|source| kvcr_fetch(context, source).ok())
            .into_iter()
            .collect())
    }
}

/// Encode a copy as KVCR's `kv.fetch` action.
fn kvcr_fetch(
    context: &HintContext<'_>,
    copy: &KvCopySource,
) -> Result<KvHintAction, HintPolicyError> {
    let locations = context
        .kv_copy_locations(copy)
        .ok_or_else(|| HintPolicyError::new("the copy source is no longer reachable"))?;
    Ok(fetch_action(KvSourceLocationsPayload {
        source_control_endpoint: locations.source_address().to_string(),
        // kv.fetch@1.0 wants every hash from block 0 through the source's prefix; the target
        // fetches what it lacks.
        block_hashes: locations.chain()[..locations.source_prefix_blocks()].to_vec(),
    }))
}

/// Register `kv_fetch` on by default under the `kvcr` worker-selection policy, so a `kvcr` role
/// whose entry lists no `hints` sends every committed copy, and best-effort fetches of one block or
/// more when `kvcr` commits none. Any other policy runs it only when its role lists it in `hints`,
/// for example a custom policy whose scorer reads `KV_COPY`.
pub(crate) fn register(registry: &mut RouterPluginRegistry) -> Result<(), HintPolicyRegistryError> {
    registry.register_hint_policy(
        POLICY_TYPE,
        Arc::new(|parameters| {
            let Parameters {
                min_additional_blocks,
            } = parameters.deserialize()?;
            if min_additional_blocks == 0 {
                return Err(HintPolicyProviderError::new(
                    "min_additional_blocks must be at least 1",
                ));
            }
            Ok(HintPolicyConstructor::shared(KvFetch {
                min_additional_blocks,
            }))
        }),
        [HintHost::worker_selection(KVCR_POLICY_TYPE)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_scheduler::kv_hints::KvHint;

    /// The wire shape engines and KVCR read: field names and values are N-2 compatible.
    #[test]
    fn serializes_versioned_fetch_action() {
        let mut action = fetch_action(KvSourceLocationsPayload {
            source_control_endpoint: "tcp://127.0.0.1:23280".to_string(),
            block_hashes: vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(22)],
        });
        action.action_id = "a1".to_string();
        let hint = KvHint::new("msg-123", vec![action]);

        assert_eq!(
            serde_json::to_value(hint).unwrap(),
            serde_json::json!({
                "protocol_version": "0.1",
                "message_id": "msg-123",
                "actions": [{
                    "action_id": "a1",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://127.0.0.1:23280",
                        "block_hashes": [11, 22],
                    },
                }],
            })
        );
    }
}
