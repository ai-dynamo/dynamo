// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction of a [`RoutingHost`].
//!
//! One builder serves every routing mode and every host. The frontend, the
//! prefill binding, the Python bindings and tests all describe what they have
//! (a discovery client, a routing mode, an optional [`KvRouter`], an optional
//! load context, an affinity coordinator, LoRA, a dispatch override) and the
//! builder derives the host's policy, its transport and its capability checks
//! from that in one place.

use std::sync::Arc;

use dynamo_kv_router::selector::{WorkerInputs, WorkerSelector};
use dynamo_runtime::{
    component::Client,
    pipeline::{
        MultimodalCacheIndex, MultimodalCacheKeyExtractor, PushRouter, PushRouterOptions,
        RouterMode, StreamingDispatch,
    },
    protocols::annotated::Annotated,
};

use super::{BuiltinWorkerSelector, HostedOccupancy, LoraRouting, RoutingHost, RoutingPolicy};
use crate::{
    kv_router::{KvRouter, RoutingLoadContext, metrics::RouterRequestMetrics},
    lora::{LoadEstimator, LoraFilter},
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    session_affinity::AffinityCoordinator,
};

/// The transport a [`RoutingHost`] sends through.
type LlmPushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

/// A caller-supplied final hop in place of the request plane.
pub type LlmDispatch = Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>;

/// Multimodal embedding-cache state for [`RouterMode::DeviceAwareWeighted`].
pub type MultimodalCache = (
    Arc<dyn MultimodalCacheIndex>,
    MultimodalCacheKeyExtractor<PreprocessedRequest>,
);

/// Builds a [`RoutingHost`]. Obtain one from [`RoutingHost::builder`].
pub struct RoutingHostBuilder {
    client: Client,
    router_mode: RouterMode,
    kv_router: Option<Arc<KvRouter>>,
    load_context: Option<Arc<RoutingLoadContext>>,
    affinity: Option<AffinityCoordinator>,
    lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    dispatch: Option<LlmDispatch>,
    multimodal_cache: Option<MultimodalCache>,
}

impl RoutingHost {
    /// Start building a host that discovers workers through `client` and selects
    /// them with `router_mode`. KV routing also needs [`RoutingHostBuilder::kv_router`].
    pub fn builder(client: Client, router_mode: RouterMode) -> RoutingHostBuilder {
        RoutingHostBuilder {
            client,
            router_mode,
            kv_router: None,
            load_context: None,
            affinity: None,
            lora: None,
            dispatch: None,
            multimodal_cache: None,
        }
    }
}

impl RoutingHostBuilder {
    /// The KV-aware selection plane. Required for [`RouterMode::KV`], rejected otherwise.
    pub fn kv_router(mut self, kv_router: impl Into<Option<Arc<KvRouter>>>) -> Self {
        self.kv_router = kv_router.into();
        self
    }

    /// The shared client, overload state and cancellation subtree this host
    /// belongs to. Required for every mode except KV.
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

    /// LoRA-aware selection for the round-robin and random modes.
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

    /// Multimodal embedding-cache state for [`RouterMode::DeviceAwareWeighted`].
    pub fn multimodal_cache(mut self, cache: impl Into<Option<MultimodalCache>>) -> Self {
        self.multimodal_cache = cache.into();
        self
    }

    /// Check the inputs against the routing mode, build the transport, and
    /// assemble the host.
    pub async fn build(self) -> anyhow::Result<RoutingHost> {
        let Self {
            client,
            router_mode,
            kv_router,
            load_context,
            affinity,
            lora,
            dispatch,
            multimodal_cache,
        } = self;
        let inputs = PolicyInputs::check(
            router_mode,
            kv_router,
            load_context,
            affinity.is_some(),
            lora,
        )?;
        let (multimodal_cache_indexer, multimodal_cache_key_extractor) = multimodal_cache
            .map_or((None, None), |(indexer, extractor)| {
                (Some(indexer), Some(extractor))
            });
        let inner = LlmPushRouter::from_client_with_options(
            client,
            router_mode,
            PushRouterOptions {
                multimodal_cache_indexer,
                multimodal_cache_key_extractor,
                dispatch,
                ..PushRouterOptions::default()
            },
        )
        .await?;
        assemble(inner, inputs, affinity)
    }
}

/// The policy a host runs and what it needs, checked against the routing mode
/// before the transport exists.
enum PolicyInputs {
    Kv {
        kv_router: Arc<KvRouter>,
        load_context: Option<Arc<RoutingLoadContext>>,
    },
    Builtin {
        policy: RoutingPolicy,
        load_context: Arc<RoutingLoadContext>,
        lora: Option<LoraRouting>,
    },
}

impl PolicyInputs {
    fn check(
        router_mode: RouterMode,
        kv_router: Option<Arc<KvRouter>>,
        load_context: Option<Arc<RoutingLoadContext>>,
        has_affinity: bool,
        lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
    ) -> anyhow::Result<Self> {
        match (router_mode, kv_router) {
            (RouterMode::KV, Some(kv_router)) => {
                if lora.is_some() {
                    anyhow::bail!("LoRA filtering is unsupported with KV routing");
                }
                Ok(Self::Kv {
                    kv_router,
                    load_context,
                })
            }
            (RouterMode::KV, None) => anyhow::bail!("RouterMode::KV requires a KvRouter"),
            (mode, Some(_)) => {
                anyhow::bail!("a KvRouter is only used with RouterMode::KV, not {mode:?}")
            }
            (mode, None) => {
                let Some(load_context) = load_context else {
                    anyhow::bail!("{mode:?} routing requires a RoutingLoadContext");
                };
                if has_affinity && lora.is_some() {
                    anyhow::bail!("session affinity and LoRA filtering cannot both be enabled");
                }
                let policy = match mode {
                    RouterMode::Direct => RoutingPolicy::Direct,
                    RouterMode::DeviceAwareWeighted => RoutingPolicy::DeviceAwareWeighted,
                    mode => RoutingPolicy::Builtin(builtin_selector(mode)?),
                };
                let lora = match (lora, mode) {
                    (None, _) => None,
                    (
                        Some((filter, load_estimator)),
                        RouterMode::RoundRobin | RouterMode::Random,
                    ) => Some(LoraRouting {
                        filter,
                        load_estimator,
                        selector: builtin_selector(mode)?,
                    }),
                    (Some(_), mode) => {
                        anyhow::bail!("LoRA filtering is unsupported with {mode:?} routing")
                    }
                };
                Ok(Self::Builtin {
                    policy,
                    load_context,
                    lora,
                })
            }
        }
    }
}

fn builtin_selector(mode: RouterMode) -> anyhow::Result<BuiltinWorkerSelector> {
    BuiltinWorkerSelector::new(mode)
        .ok_or_else(|| anyhow::anyhow!("{mode:?} routing is not a first-party policy"))
}

/// Wire a host over its transport. This is the one place that turns
/// construction inputs into a [`RoutingPolicy`] and its capabilities.
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
                lora: None,
                routing_context: load_context,
            })
        }
        PolicyInputs::Builtin {
            policy,
            load_context,
            lora,
        } => {
            let hosted_occupancy = matches!(
                &policy,
                RoutingPolicy::Builtin(selector)
                    if selector.required_worker_inputs().contains(WorkerInputs::ACTIVE_REQUESTS)
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
                lora,
                routing_context: Some(load_context),
            })
        }
    }
}
