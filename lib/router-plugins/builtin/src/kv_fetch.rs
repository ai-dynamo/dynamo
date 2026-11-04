// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `kv_fetch` hint policy: sends the booked worker a KV fetch from a peer.
//!
//! When the worker-selection policy committed the booked worker to a fetch, as `kvcr` does, this
//! policy delivers that fetch; if it cannot be built, the booking fails. Otherwise it fetches best
//! effort from the booked worker's eligible source when the fetch adds at least
//! `min_additional_blocks`, and skips a fetch whose source has become unreachable.

use std::sync::Arc;

use dynamo_kv_router::kv_hints::KvHintAction;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::hint_policy::{
    HintActionContext, HintPolicy, HintPolicyDefinition, HintPolicyError, HintPolicyInstance,
    HintPolicyProviderError, HintPolicyRegistryError, HintRequirements,
};

/// Hint-policy type selected by `hint_policies.instances[].type`.
pub const POLICY_TYPE: &str = "kv_fetch";

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Blocks a best-effort fetch must add beyond the booked worker's device and host-pinned
    /// prefix. Committed fetches are always delivered.
    min_additional_blocks: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            min_additional_blocks: 1,
        }
    }
}

#[derive(Clone)]
struct KvFetch {
    min_additional_blocks: usize,
}

impl HintPolicy for KvFetch {
    fn requirements(&self) -> HintRequirements {
        HintRequirements::new()
            .with_actions()
            .with_committed_kv_fetch()
            .with_kv_fetch_sources()
    }

    fn hint_kv_locality(
        &self,
        context: &HintActionContext<'_>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        // The selection's cost assumed this fetch: deliver it, or fail the booking.
        if let Some(fetch) = context.committed_kv_fetch() {
            return Ok(vec![context.kv_fetch_action(fetch)?]);
        }
        Ok(context
            .kv_fetch_source()
            .filter(|source| source.additional_blocks() >= self.min_additional_blocks)
            .and_then(|source| context.kv_fetch_action(&source).ok())
            .into_iter()
            .collect())
    }
}

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
            let policy = KvFetch {
                min_additional_blocks,
            };
            Ok(HintPolicyDefinition::new(
                policy.requirements(),
                Arc::new(move |_, _, _| HintPolicyInstance::new(POLICY_TYPE, policy.clone())),
            ))
        }),
    )
}
