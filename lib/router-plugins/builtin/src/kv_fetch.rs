// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kv_fetch`: the hint the default policies can send with the worker they book. It tells the
//! worker to fetch, as `kv.fetch@1.0`, the request's KV prefix from the peer the router found
//! holding more of it than the worker does.
//!
//! `kvcr` enables it by default: with copy credit (`shared_cache_multiplier` above 0) its score
//! assumed the copy, so the fetch always goes out; without, the fetch is best effort and goes out
//! when it adds at least `min_additional_blocks`. `dynamo-default-cost-fn` can enable it too, as a
//! best-effort fetch. Both configure it through their `parameters`:
//!
//! ```yaml
//! worker_selection:
//!   prefill:
//!     type: kvcr
//!     parameters:
//!       shared_cache_multiplier: 0.5
//!       enable_hints: [kv_fetch]        # the default under kvcr; [] sends none
//!       hint_parameters:
//!         kv_fetch:
//!           min_additional_blocks: 2
//! ```
//!
//! A custom policy that wants the same fetch attaches [`KvFetch`] with
//! `WorkerSelectionPolicy::with_hints`, or builds the action itself with
//! [`KvHintAction::kv_fetch`].

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

/// The name `enable_hints` and `hint_parameters` know this hint by.
pub const HINT_NAME: &str = "kv_fetch";

/// `hint_parameters.kv_fetch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct KvFetchParameters {
    /// A fetch the policy's score did not credit goes out only when it adds at least this many
    /// blocks beyond the booked worker's own prefix. At least 1.
    pub min_additional_blocks: usize,
}

impl Default for KvFetchParameters {
    fn default() -> Self {
        Self {
            min_additional_blocks: 1,
        }
    }
}

/// Sends the booked worker's copy source as a `kv.fetch@1.0` action.
pub struct KvFetch {
    min_additional_blocks: usize,
    credited: bool,
}

impl KvFetch {
    /// `credited` is whether the policy's score assumed the copy, in which case the fetch always
    /// goes out; otherwise `parameters.min_additional_blocks` gates it.
    pub fn new(
        parameters: KvFetchParameters,
        credited: bool,
    ) -> Result<Self, WorkerSelectionPolicyProviderError> {
        if parameters.min_additional_blocks == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "hint_parameters.kv_fetch.min_additional_blocks must be at least 1",
            ));
        }
        Ok(Self {
            min_additional_blocks: parameters.min_additional_blocks,
            credited,
        })
    }

    /// [`Self::new`] as the emitter a policy attaches.
    pub fn emitter(
        parameters: KvFetchParameters,
        credited: bool,
    ) -> Result<Arc<dyn HintEmitter>, WorkerSelectionPolicyProviderError> {
        Ok(Arc::new(Self::new(parameters, credited)?))
    }
}

impl HintEmitter for KvFetch {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::KV_COPY
    }

    fn emit(&self, context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintError> {
        let Some(copy) = context.kv_copy() else {
            return Ok(Vec::new());
        };
        // No locations: the source left between selection and booking. The router logged the
        // dropped copy; the worker recomputes instead.
        let Some(locations) = context.kv_copy_locations() else {
            return Ok(Vec::new());
        };
        if !self.credited && copy.additional_blocks() < self.min_additional_blocks {
            return Ok(Vec::new());
        }
        Ok(vec![KvHintAction::kv_fetch(KvSourceLocationsPayload::of(
            locations,
        ))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_hints::KvTransferSourceChoice;

    fn copy(local: usize, source: usize) -> KvCopySource {
        KvCopySource::new(
            WorkerWithDpRank::new(1, 0),
            KvTransferSourceChoice {
                source: WorkerWithDpRank::new(2, 0).into(),
                prefix_blocks: source,
            },
            local,
        )
    }

    #[test]
    fn serializes_versioned_fetch_action() {
        let mut action = KvHintAction::kv_fetch(KvSourceLocationsPayload {
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

    #[test]
    fn credited_fetches_always_go_out_and_others_meet_the_threshold() {
        let chain: Vec<_> = (1..=8).map(ExternalSequenceBlockHash).collect();
        let locations = KvCopyLocations::new("tcp://worker-2:9000".to_string(), &chain, 2, 8);
        let copy = copy(2, 8);
        let context = || HintContext::new("r", WorkerWithDpRank::new(1, 0), 32, 4, None);

        let threshold = |min, credited| {
            KvFetch::new(
                KvFetchParameters {
                    min_additional_blocks: min,
                },
                credited,
            )
            .unwrap()
        };
        let sent = |emitter: KvFetch| {
            emitter
                .emit(&context().with_kv_copy(Some(&copy), Some(&locations)))
                .unwrap()
        };
        // The copy adds 8 - 2 = 6 blocks.
        assert_eq!(sent(threshold(6, false)).len(), 1);
        assert!(sent(threshold(7, false)).is_empty());
        let credited = sent(threshold(7, true));
        assert_eq!(credited.len(), 1);
        assert_eq!(credited[0].action_type, KV_FETCH_ACTION_TYPE);
        assert_eq!(
            credited[0].payload["block_hashes"],
            serde_json::json!(chain)
        );

        // Nothing to fetch, or a source that left, sends nothing.
        assert!(threshold(1, true).emit(&context()).unwrap().is_empty());
        assert!(
            threshold(1, true)
                .emit(&context().with_kv_copy(Some(&copy), None))
                .unwrap()
                .is_empty()
        );
        assert!(
            KvFetch::new(
                KvFetchParameters {
                    min_additional_blocks: 0
                },
                false
            )
            .is_err()
        );
    }
}
