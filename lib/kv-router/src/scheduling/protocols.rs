// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router-local request vocabulary: what a host hands the scheduler kernel
//! (`RouterRequest`, `RoutingConstraints`, the worker-config view `WorkerConfigLike`) and
//! what comes back (`RouterResponse`, `PotentialLoad`, `WorkerSelectionResult`), plus the
//! event sink hosts implement. Kernel-owned per SPLIT_V2 section 4 rows 40-41.

use crate::kv_hints::{KvTransferSourceChoice, WorkerKvTransferKeys, WorkerKvTransferKeysExt};
use crate::protocols::{BlockExtraInfo, DpRank, RouterEvent, WorkerId, WorkerWithDpRank};
use dynamo_tokens::Token;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvHintTransferWorkerMetadata<'a> {
    pub worker_type: &'a str,
    pub source_control_endpoint: Option<&'a str>,
}

/// Trait abstracting the worker configuration fields needed by the scheduling layer.
///
/// `ModelRuntimeConfig` (in `lib/llm`) implements this directly so no adapter type is needed.
pub trait WorkerConfigLike {
    fn data_parallel_start_rank(&self) -> u32;
    fn data_parallel_size(&self) -> u32;
    fn max_num_batched_tokens(&self) -> Option<u64>;
    fn total_kv_blocks(&self) -> Option<u64>;

    /// TRANSFER capability and source metadata for a specific global DP rank.
    ///
    /// `None` means this worker/rank does not support TRANSFER. Backends that
    /// support TRANSFER but cannot serve as a source may return `Some` with
    /// `source_control_endpoint: None`. Capability and `worker_type` are the
    /// same for every rank of a worker; only the source endpoint may differ.
    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        _dp_rank: DpRank,
    ) -> Option<KvHintTransferWorkerMetadata<'_>> {
        None
    }

    /// Tokens retained by the backend's native KV offloading tier, if available.
    fn native_offloading_capacity_tokens(&self) -> Option<u64> {
        None
    }

    fn taints(&self) -> &HashSet<String> {
        &EMPTY_WORKER_TAINTS
    }

    /// Stable identifier for the worker, preserved across process restarts.
    ///
    /// In Kubernetes StatefulSet deployments this is the pod hostname (`worker-0`, `worker-1`,
    /// …). Used by rendezvous-style routing (HRW hashing) so cache assignments survive worker
    /// restarts and minimise cache movement when the set of live workers churns. Returns
    /// `None` when the worker did not publish a stable id, in which case callers should fall
    /// back to the (ephemeral) `worker_id`.
    fn stable_routing_id(&self) -> Option<&str> {
        None
    }

    /// Returns the worker's topology domain labels (e.g. {"zone": "us-east-1a", "rack": "rack1"}).
    /// Topology-aware routing turns these labels into canonical worker taints such as
    /// `dynamo.topology/zone=us-east-1a`.
    /// Returns `None` by default for backward compatibility.
    fn topology_domains(&self) -> Option<&HashMap<String, String>> {
        None
    }

    /// Returns the topology domain to enforce for KV-cache transfers (e.g. "zone").
    /// When set, decode worker selection is constrained to workers sharing the same
    /// topology domain value as the prefill worker.
    fn kv_transfer_domain(&self) -> Option<&str> {
        None
    }

    /// Returns the KV transfer topology enforcement mode.
    fn kv_transfer_enforcement(&self) -> Option<KvTransferEnforcement> {
        None
    }

    /// Returns the taint preference weight used when KV transfer topology enforcement is preferred.
    fn kv_transfer_preferred_weight(&self) -> Option<f32> {
        None
    }

    /// This worker's router-hint identity with hashed labels, used to choose
    /// KV fetch sources. Configs read by many selections should cache it.
    fn kv_transfer_keys(&self) -> Cow<'_, WorkerKvTransferKeys> {
        Cow::Owned(WorkerKvTransferKeys::of(self))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferEnforcement {
    /// Put the generated topology taint in `RoutingConstraints.required_taints`.
    Required,
    /// Put the generated topology taint in `RoutingConstraints.preferred_taints`.
    Preferred,
}

/// Request-level taint constraints evaluated against each worker's published taints.
///
/// Topology-aware routing uses the same fields with canonical taints such as
/// `dynamo.topology/zone=us-east-1a`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RoutingConstraints {
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub required_taints: HashSet<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub preferred_taints: HashMap<String, f32>,
}

impl RoutingConstraints {
    pub fn is_empty(&self) -> bool {
        self.required_taints.is_empty() && self.preferred_taints.is_empty()
    }

    pub fn has_hard_constraints(&self) -> bool {
        !self.required_taints.is_empty()
    }

    pub fn is_compatible_with_worker_taints(&self, worker_taints: &HashSet<String>) -> bool {
        if self.required_taints.is_empty() {
            return true;
        }

        self.required_taints
            .iter()
            .all(|taint| worker_taints.contains(taint))
    }

    pub fn preferred_taint_matches(&self, worker_taints: &HashSet<String>) -> usize {
        if self.preferred_taints.is_empty() {
            return 0;
        }

        self.preferred_taints
            .keys()
            .filter(|taint| worker_taints.contains(*taint))
            .count()
    }

    pub fn preferred_taint_multiplier(&self, worker_taints: &HashSet<String>) -> Option<f64> {
        if self.preferred_taints.is_empty() {
            return None;
        }

        // Use exp(-tanh(sum)) so equal-magnitude positive and negative preferences
        // have reciprocal effect around the neutral multiplier 1.0, while keeping the
        // multiplier strictly positive and bounded to [exp(-1), exp(1)] ~= [0.368, 2.718]
        // for numerically stable composition with the existing linear work score.
        let bias = self
            .preferred_taints
            .iter()
            .filter(|(taint, _)| worker_taints.contains(*taint))
            .map(|(_, weight)| f64::from(*weight))
            .sum::<f64>()
            .tanh();

        Some((-bias).exp())
    }
}

static EMPTY_WORKER_TAINTS: LazyLock<HashSet<String>> = LazyLock::new(HashSet::new);

/// Transport abstraction for publishing batched router-visible KV cache events.
pub trait RouterEventSink: Send + Sync {
    fn publish_event(&self, event: &RouterEvent)
    -> impl Future<Output = anyhow::Result<()>> + Send;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RouterRequest {
    #[serde(rename = "new")]
    New {
        tokens: Vec<Token>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        #[serde(default, skip_serializing_if = "RoutingConstraints::is_empty")]
        routing_constraints: RoutingConstraints,
        #[serde(default)]
        priority_jump: f64,
        #[serde(default, skip_serializing_if = "is_zero")]
        strict_priority: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_namespace: Option<String>,
    },
    PotentialLoads {
        tokens: Vec<Token>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_namespace: Option<String>,
    },
    MarkPrefill {
        // once prefill completes, the frontend might not be allowed to send a
        // request with linking the id. In this case, the request_id is provided in the payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
    },
    MarkFree {
        // once request is cancelled, the frontend might not be allowed to send a
        // request with linking the id. In this case, the request_id is provided in the payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
    },
}

impl Default for RouterRequest {
    fn default() -> Self {
        RouterRequest::New {
            tokens: vec![],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 0.0,
            strict_priority: 0,
            lora_name: None,
            cache_namespace: None,
        }
    }
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
    #[serde(default)]
    pub active_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RouterResponse {
    New {
        worker_id: WorkerId,
        #[serde(default)]
        dp_rank: DpRank,
        overlap_blocks: u32,
    },
    QueueRejected {
        rejection: crate::scheduling::QueueRejection,
    },
    PrefillMarked {
        success: bool,
    },
    FreeMarked {
        success: bool,
    },
    PotentialLoads {
        // loads of every worker tracked by the scheduler.
        loads: Vec<PotentialLoad>,
        // the queue sizes for this specific router instance.
        #[serde(default)]
        pending_count: usize,
        #[serde(default)]
        pending_isl_tokens: usize,
    },
}

#[derive(Debug)]
pub struct WorkerSelectionResult {
    /// The full worker information including dp_rank
    pub worker: WorkerWithDpRank,

    /// The total number of blocks required to prefill the request
    pub required_blocks: u64,

    /// Approximate effective cache hit on the selected worker in fractional blocks.
    /// Use `.round() as u32` for a block-count approximation.
    pub effective_overlap_blocks: f64,

    /// Approximate cached-token count derived from the weighted cache hit.
    pub cached_tokens: usize,

    /// Greatest raw router-visible overlap among eligible workers, in tokens,
    /// for tracked requests.
    pub max_raw_cached_tokens: Option<usize>,

    /// Selected worker's raw router-visible overlap, in tokens, for tracked requests.
    pub selected_raw_cached_tokens: Option<usize>,

    /// Selected worker's projected decode load after adding this request's
    /// prompt blocks, in scheduler-tracked block units.
    pub potential_decode_blocks: usize,

    /// The source the selected worker fetches its KV prefix from, chosen with
    /// the same rule that credited the fetch during selection.
    pub kv_transfer_source: Option<KvTransferSourceChoice>,

    /// Fields for the routing task's "Selected worker" log line. Set by
    /// [`crate::WorkerSelectionPolicy`]; `None` from selectors that do not log
    /// their selection.
    pub log_fields: Option<SelectionLogFields>,
}

/// The selection facts the "Selected worker" log line reports beyond the
/// scheduling response's own fields. The scheduler actor copies them so the
/// routing task can format the line after the scheduler replies.
#[derive(Debug, Clone, Copy)]
pub struct SelectionLogFields {
    /// The policy's worker label, e.g. `"decode"` or `"prefill"`.
    pub worker_type: &'static str,
    /// The selected candidate's cost.
    pub logit: f64,
    /// Whether the request pinned the selected worker and rank.
    pub pinned: bool,
    /// Host-pinned overlap reported for the selected rank, in blocks.
    pub host_pinned_blocks: usize,
    /// Disk overlap reported for the selected rank, in blocks.
    pub disk_blocks: usize,
    /// The selected worker's KV capacity as its configuration reports it.
    pub total_kv_blocks: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    #[test]
    fn test_router_request_mark_free_backwards_compatible_deserialization() {
        let request: RouterRequest = serde_json::from_str(r#"{"method":"mark_free"}"#).unwrap();

        assert!(matches!(
            request,
            RouterRequest::MarkFree { request_id: None }
        ));
    }

    #[test]
    fn test_kv_transfer_enforcement_serde() {
        assert_eq!(
            serde_json::to_string(&KvTransferEnforcement::Required).unwrap(),
            r#""required""#
        );
        assert_eq!(
            serde_json::from_str::<KvTransferEnforcement>(r#""preferred""#).unwrap(),
            KvTransferEnforcement::Preferred
        );
        assert!(serde_json::from_str::<KvTransferEnforcement>(r#""fallback""#).is_err());
    }

    #[test]
    fn test_worker_config_like_topology_domains_default() {
        // A minimal implementor that does NOT override topology_domains()
        struct MinimalConfig;
        impl WorkerConfigLike for MinimalConfig {
            fn data_parallel_start_rank(&self) -> u32 {
                0
            }
            fn data_parallel_size(&self) -> u32 {
                1
            }
            fn max_num_batched_tokens(&self) -> Option<u64> {
                None
            }
            fn total_kv_blocks(&self) -> Option<u64> {
                None
            }
        }

        let config = MinimalConfig;
        assert!(
            config.topology_domains().is_none(),
            "Default topology_domains() should return None"
        );
        assert!(
            config.kv_transfer_domain().is_none(),
            "Default kv_transfer_domain() should return None"
        );
        assert!(
            config.kv_transfer_enforcement().is_none(),
            "Default kv_transfer_enforcement() should return None"
        );
        assert!(
            config.kv_transfer_preferred_weight().is_none(),
            "Default kv_transfer_preferred_weight() should return None"
        );
        assert!(config.native_offloading_capacity_tokens().is_none());
        assert!(config.kv_hint_transfer_metadata_for_dp_rank(0).is_none());
    }

    #[test]
    fn test_router_request_mark_free_serialization_with_request_id() {
        let request = RouterRequest::MarkFree {
            request_id: Some("req-123".to_string()),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"mark_free","request_id":"req-123"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::MarkFree {
                request_id: Some(ref request_id)
            } if request_id == "req-123"
        ));
    }

    #[test]
    fn test_router_request_new_serialization_with_priority_jump() {
        let request = RouterRequest::New {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 5.0,
            strict_priority: 0,
            lora_name: None,
            cache_namespace: None,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"new","tokens":[1,2,3],"priority_jump":5.0}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::New {
                priority_jump,
                ..
            } if priority_jump == 5.0
        ));
    }

    #[test]
    fn test_router_request_new_serialization_with_lora_name() {
        let request = RouterRequest::New {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 0.0,
            strict_priority: 0,
            lora_name: Some("adapter-a".to_string()),
            cache_namespace: None,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"new","tokens":[1,2,3],"priority_jump":0.0,"lora_name":"adapter-a"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::New {
                tokens,
                lora_name: Some(ref lora_name),
                ..
            } if tokens == vec![1, 2, 3] && lora_name == "adapter-a"
        ));
    }

    #[test]
    fn test_router_request_new_defaults_lora_name() {
        let deserialized: RouterRequest =
            serde_json::from_str(r#"{"method":"new","tokens":[1,2,3]}"#).unwrap();

        assert!(matches!(
            deserialized,
            RouterRequest::New {
                tokens,
                lora_name: None,
                ..
            } if tokens == vec![1, 2, 3]
        ));
    }

    #[test]
    fn test_router_request_new_strict_priority_compatibility() {
        let request = RouterRequest::New {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 0.0,
            strict_priority: 4,
            lora_name: None,
            cache_namespace: None,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serialized,
            r#"{"method":"new","tokens":[1,2,3],"priority_jump":0.0,"strict_priority":4}"#
        );

        let missing: RouterRequest =
            serde_json::from_str(r#"{"method":"new","tokens":[1,2,3]}"#).unwrap();
        assert!(matches!(
            missing,
            RouterRequest::New {
                strict_priority: 0,
                ..
            }
        ));

        let zero = RouterRequest::New {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 0.0,
            strict_priority: 0,
            lora_name: None,
            cache_namespace: None,
        };
        assert_eq!(
            serde_json::to_string(&zero).unwrap(),
            r#"{"method":"new","tokens":[1,2,3],"priority_jump":0.0}"#
        );
    }

    #[test]
    fn test_router_request_new_serialization_with_cache_namespace() {
        let request = RouterRequest::New {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            routing_constraints: RoutingConstraints::default(),
            priority_jump: 0.0,
            strict_priority: 0,
            lora_name: None,
            cache_namespace: Some("tenant-a".to_string()),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"new","tokens":[1,2,3],"priority_jump":0.0,"cache_namespace":"tenant-a"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::New {
                tokens,
                cache_namespace: Some(ref cache_namespace),
                ..
            } if tokens == vec![1, 2, 3] && cache_namespace == "tenant-a"
        ));
    }

    #[test]
    fn test_router_request_potential_loads_serialization_with_lora_name() {
        let request = RouterRequest::PotentialLoads {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            lora_name: Some("adapter-a".to_string()),
            cache_namespace: None,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"potential_loads","tokens":[1,2,3],"lora_name":"adapter-a"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::PotentialLoads {
                tokens,
                block_mm_infos: None,
                lora_name: Some(ref lora_name),
                cache_namespace: None,
            } if tokens == vec![1, 2, 3] && lora_name == "adapter-a"
        ));
    }

    #[test]
    fn test_router_request_potential_loads_serialization_with_cache_namespace() {
        let request = RouterRequest::PotentialLoads {
            tokens: vec![1, 2, 3],
            block_mm_infos: None,
            lora_name: None,
            cache_namespace: Some("tenant-a".to_string()),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"potential_loads","tokens":[1,2,3],"cache_namespace":"tenant-a"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::PotentialLoads {
                tokens,
                cache_namespace: Some(ref cache_namespace),
                ..
            } if tokens == vec![1, 2, 3] && cache_namespace == "tenant-a"
        ));
    }

    #[test]
    fn test_router_request_potential_loads_defaults_lora_name() {
        let deserialized: RouterRequest =
            serde_json::from_str(r#"{"method":"potential_loads","tokens":[1,2,3]}"#).unwrap();

        assert!(matches!(
            deserialized,
            RouterRequest::PotentialLoads {
                tokens,
                block_mm_infos: None,
                lora_name: None,
                cache_namespace: None,
            } if tokens == vec![1, 2, 3]
        ));
    }

    #[test]
    fn test_router_request_mark_prefill_serialization_with_request_id() {
        let request = RouterRequest::MarkPrefill {
            request_id: Some("req-123".to_string()),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RouterRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            serialized,
            r#"{"method":"mark_prefill","request_id":"req-123"}"#
        );
        assert!(matches!(
            deserialized,
            RouterRequest::MarkPrefill {
                request_id: Some(ref request_id)
            } if request_id == "req-123"
        ));
    }

    #[test]
    fn test_potential_load_defaults_active_requests() {
        let load = serde_json::from_str::<PotentialLoad>(
            r#"{"worker_id":1,"dp_rank":0,"potential_prefill_tokens":16,"potential_decode_blocks":4}"#,
        )
        .unwrap();

        assert_eq!(load.worker_id, 1);
        assert_eq!(load.dp_rank, 0);
        assert_eq!(load.potential_prefill_tokens, 16);
        assert_eq!(load.potential_decode_blocks, 4);
        assert_eq!(load.active_requests, 0);
    }

    #[test]
    fn test_potential_load_serializes_active_requests() {
        let load = PotentialLoad {
            worker_id: 1,
            dp_rank: 0,
            potential_prefill_tokens: 16,
            potential_decode_blocks: 4,
            active_requests: 2,
        };

        assert_eq!(
            serde_json::to_string(&load).unwrap(),
            r#"{"worker_id":1,"dp_rank":0,"potential_prefill_tokens":16,"potential_decode_blocks":4,"active_requests":2}"#
        );
    }
}
