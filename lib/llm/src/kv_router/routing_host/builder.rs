// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction of a [`RoutingHost`].
//!
//! One builder serves every routing policy and every host. The frontend, the
//! prefill binding, the Python bindings and tests all describe what they have
//! (a discovery client, the role's [`RoutingPolicy`], an optional load context,
//! an affinity coordinator, LoRA, a dispatch override) and the builder derives
//! the host's transport and its capability checks from that in one place.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::WorkerInputs;
use dynamo_runtime::{
    component::Client,
    pipeline::{
        MultimodalCacheIndex, MultimodalCacheKeyExtractor, PushRouter, PushRouterOptions,
        StreamingDispatch,
    },
    protocols::annotated::Annotated,
};

use super::{HostedOccupancy, RoutingHost, RoutingPolicy};
use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::{KvRouter, RoutingLoadContext, metrics::RouterRequestMetrics},
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    routing_policy::HostedCandidatePolicy,
    session_affinity::AffinityCoordinator,
};

/// The transport a [`RoutingHost`] sends through.
type LlmPushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

/// A caller-supplied final hop in place of the request plane.
pub type LlmDispatch = Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>;

/// Multimodal embedding-cache state for [`RoutingPolicy::DeviceAwareWeighted`].
pub type MultimodalCache = (
    Arc<dyn MultimodalCacheIndex>,
    MultimodalCacheKeyExtractor<PreprocessedRequest>,
);

/// Builds a [`RoutingHost`]. Obtain one from [`RoutingHost::builder`].
pub struct RoutingHostBuilder {
    client: Client,
    policy: RoutingPolicy,
    load_context: Option<Arc<RoutingLoadContext>>,
    affinity: Option<AffinityCoordinator>,
    lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    candidate_policy: Option<Arc<dyn HostedCandidatePolicy>>,
    worker_configs: Option<RuntimeConfigWatch>,
    dispatch: Option<LlmDispatch>,
    multimodal_cache: Option<MultimodalCache>,
}

impl RoutingHost {
    /// Start building a host that discovers workers through `client` and selects
    /// them with `policy`.
    pub fn builder(client: Client, policy: RoutingPolicy) -> RoutingHostBuilder {
        RoutingHostBuilder {
            client,
            policy,
            load_context: None,
            affinity: None,
            lora: None,
            candidate_policy: None,
            worker_configs: None,
            dispatch: None,
            multimodal_cache: None,
        }
    }
}

impl RoutingHostBuilder {
    pub(crate) fn candidate_policy(
        mut self,
        policy: Option<Arc<dyn HostedCandidatePolicy>>,
    ) -> Self {
        self.candidate_policy = policy;
        self
    }

    /// The shared client, overload state and cancellation subtree this host
    /// belongs to. Required for every policy except [`RoutingPolicy::Kv`].
    pub fn load_context(
        mut self,
        load_context: impl Into<Option<Arc<RoutingLoadContext>>>,
    ) -> Self {
        self.load_context = load_context.into();
        self
    }

    /// Session affinity with a coordinator the caller built. `None` disables it.
    pub fn affinity_coordinator(
        mut self,
        affinity: impl Into<Option<AffinityCoordinator>>,
    ) -> Self {
        self.affinity = affinity.into();
        self
    }

    /// The per-worker runtime configs of the endpoint `client` discovers. Required for a
    /// hosted policy that honors required taints, which reads each worker's taints from
    /// them; no other policy reads them.
    pub fn worker_configs(mut self, worker_configs: impl Into<Option<RuntimeConfigWatch>>) -> Self {
        self.worker_configs = worker_configs.into();
        self
    }

    /// LoRA-aware selection for the round-robin and random policies.
    pub fn lora(mut self, lora: impl Into<Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>>) -> Self {
        self.lora = lora.into();
        self
    }

    /// Send through `dispatch` instead of the request plane. A host that does not
    /// send requests itself, such as a gateway endpoint picker, supplies the hop
    /// that records the selection and relays observed progress.
    pub fn dispatch(mut self, dispatch: impl Into<Option<LlmDispatch>>) -> Self {
        self.dispatch = dispatch.into();
        self
    }

    /// Multimodal embedding-cache state for [`RoutingPolicy::DeviceAwareWeighted`].
    pub fn multimodal_cache(mut self, cache: impl Into<Option<MultimodalCache>>) -> Self {
        self.multimodal_cache = cache.into();
        self
    }

    /// Check the inputs against the policy, build the transport, and
    /// assemble the host.
    pub async fn build(self) -> anyhow::Result<RoutingHost> {
        let Self {
            client,
            policy,
            load_context,
            affinity,
            lora,
            candidate_policy,
            worker_configs,
            dispatch,
            multimodal_cache,
        } = self;
        let candidate_policy = match (candidate_policy, lora) {
            (Some(policy), _) => Some(policy),
            (None, Some((filter, estimator))) => Some(
                crate::lora_routing::LoraHostedPolicy::hosted(filter, estimator, &policy, None)?,
            ),
            (None, None) => None,
        };
        let transport_mode = policy.transport_mode();
        let inputs = PolicyInputs::check(
            policy,
            load_context,
            affinity.is_some(),
            candidate_policy,
            worker_configs,
        )?;
        let (multimodal_cache_indexer, multimodal_cache_key_extractor) = multimodal_cache
            .map_or((None, None), |(indexer, extractor)| {
                (Some(indexer), Some(extractor))
            });
        let inner = LlmPushRouter::from_client_with_options(
            client,
            transport_mode,
            PushRouterOptions {
                multimodal_cache_indexer,
                multimodal_cache_key_extractor,
                dispatch,
                ..crate::kv_router::plugins::push_router_options()
            },
        )
        .await?;
        assemble(inner, inputs, affinity)
    }
}

/// The policy a host runs and what it needs, checked before the transport exists.
enum PolicyInputs {
    Kv {
        kv_router: Arc<KvRouter>,
        load_context: Option<Arc<RoutingLoadContext>>,
    },
    Builtin {
        policy: RoutingPolicy,
        load_context: Arc<RoutingLoadContext>,
        candidate_policy: Option<Arc<dyn HostedCandidatePolicy>>,
        worker_configs: Option<RuntimeConfigWatch>,
    },
}

impl PolicyInputs {
    fn check(
        policy: RoutingPolicy,
        load_context: Option<Arc<RoutingLoadContext>>,
        has_affinity: bool,
        candidate_policy: Option<Arc<dyn HostedCandidatePolicy>>,
        worker_configs: Option<RuntimeConfigWatch>,
    ) -> anyhow::Result<Self> {
        let policy = match policy {
            RoutingPolicy::Kv(kv_router) => {
                if candidate_policy.is_some() {
                    anyhow::bail!("KV candidate policy is supplied by its KvRouter");
                }
                return Ok(Self::Kv {
                    kv_router,
                    load_context,
                });
            }
            policy => policy,
        };
        let name = policy.name();
        if candidate_policy.is_some() && policy.hosted_without_occupancy().is_none() {
            anyhow::bail!("candidate policy is unsupported with {name} routing");
        }
        let Some(load_context) = load_context else {
            anyhow::bail!("{name} routing requires a RoutingLoadContext");
        };
        if has_affinity && candidate_policy.is_some() {
            anyhow::bail!("session affinity and LoRA filtering cannot both be enabled");
        }
        // Without the configs the host could not tell which workers have a required taint.
        let worker_configs = if policy.honors_required_taints() {
            let Some(worker_configs) = worker_configs else {
                anyhow::bail!("{name} routing honors required taints and requires worker configs");
            };
            Some(worker_configs)
        } else {
            None
        };
        Ok(Self::Builtin {
            policy,
            load_context,
            candidate_policy,
            worker_configs,
        })
    }
}

/// Wire a host over its transport. This is the one place that turns
/// construction inputs into a host and its capabilities.
fn assemble(
    inner: LlmPushRouter,
    inputs: PolicyInputs,
    affinity: Option<AffinityCoordinator>,
) -> anyhow::Result<RoutingHost> {
    let session_affinity_mode = affinity
        .as_ref()
        .map(AffinityCoordinator::mode)
        .unwrap_or_default();
    match inputs {
        PolicyInputs::Kv {
            kv_router,
            load_context,
        } => {
            // Eagerly register router request metrics (as zeros) so they are
            // scrapeable before any requests arrive. Both the frontend pipeline
            // and the standalone router create RoutingHost, so this covers both.
            let request_metrics =
                RouterRequestMetrics::from_component(kv_router.client().endpoint.component());
            Ok(RoutingHost {
                inner,
                policy: RoutingPolicy::Kv(kv_router),
                request_metrics,
                session_affinity_mode,
                affinity,
                hosted_occupancy: None,
                candidate_policy: None,
                worker_configs: None,
                routing_context: load_context,
            })
        }
        PolicyInputs::Builtin {
            policy,
            load_context,
            candidate_policy,
            worker_configs,
        } => {
            let hosted_occupancy = matches!(
                &policy,
                RoutingPolicy::Hosted(hosted)
                    if hosted.picker().inputs().contains(WorkerInputs::ACTIVE_REQUESTS)
            )
            .then(|| HostedOccupancy::new(&inner))
            .transpose()?;
            let request_metrics =
                RouterRequestMetrics::from_component(inner.client.endpoint.component());
            Ok(RoutingHost {
                inner,
                policy,
                request_metrics,
                session_affinity_mode,
                affinity,
                hosted_occupancy,
                candidate_policy,
                worker_configs,
                routing_context: Some(load_context),
            })
        }
    }
}

#[cfg(test)]
mod policy_input_tests {
    use dynamo_runtime::pipeline::RouterMode;

    use super::*;
    use crate::kv_router::role_routing::non_kv_policy;
    use crate::lora::{LoraRoutingTable, LoraStateTracker};

    #[test]
    fn candidate_policy_requires_supported_hosted_mode() {
        for mode in [
            RouterMode::Direct,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
        ] {
            let policy = crate::lora_routing::LoraHostedPolicy::hosted(
                Arc::new(LoraFilter::new(
                    LoraRoutingTable::new(),
                    LoraStateTracker::new(),
                )),
                Arc::new(LoadEstimator::new()),
                &non_kv_policy(RouterMode::RoundRobin),
                None,
            )
            .unwrap();
            let Err(error) =
                PolicyInputs::check(non_kv_policy(mode), None, false, Some(policy), None)
            else {
                panic!("{mode:?} should reject candidate policies");
            };
            assert!(
                error
                    .to_string()
                    .contains("candidate policy is unsupported"),
                "{mode:?}: {error}"
            );
        }
        for mode in [RouterMode::RoundRobin, RouterMode::Random] {
            let policy = crate::lora_routing::LoraHostedPolicy::hosted(
                Arc::new(LoraFilter::new(
                    LoraRoutingTable::new(),
                    LoraStateTracker::new(),
                )),
                Arc::new(LoadEstimator::new()),
                &non_kv_policy(RouterMode::RoundRobin),
                None,
            )
            .unwrap();
            let Err(error) =
                PolicyInputs::check(non_kv_policy(mode), None, false, Some(policy), None)
            else {
                panic!("{mode:?} requires a load context");
            };
            assert!(
                error.to_string().contains("requires a RoutingLoadContext"),
                "{mode:?}: {error}"
            );
        }
    }
}
