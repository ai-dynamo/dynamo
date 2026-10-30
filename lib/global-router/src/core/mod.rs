// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool selection through the embedded router core.
//!
//! The router core (`dynamo-kv-router`) selects a worker for each request
//! and books the request on it until the host frees it. Here each pool is
//! one of its workers ([`catalog`]), the Global Router's policies are its
//! plugins ([`policies`]), and [`CoreSelector`] is the [`RequestSelector`]
//! that asks it. The booking comes back as a [`Reservation`], which the
//! forwarding engine holds until the response ends, so the core's own
//! request counts drive [`CorePolicy::LeastRequests`] the way
//! [`crate::InFlightTracker`] drives [`crate::LeastInFlightSelector`].

mod catalog;
mod policies;

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::scheduling::RouterPolicyConfig;
use dynamo_kv_router::services::selection::{
    CatalogReconciler, PromptRequest, SelectAndReserveRequest, SelectionService,
    SelectionServiceBuilder,
};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::selector::{Cursors, RequestLifecycle, RequestSelector, Selection, reachable, target};
use crate::{Pool, RoutingRequest};

pub use catalog::{PoolCatalogSource, ROUTING_GROUP, worker_id};
pub use policies::{LeastRequestsPicker, RoundRobinPicker};

/// How the core picks among a model's pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorePolicy {
    /// Every pool in turn ([`RoundRobinPicker`]).
    RoundRobin,
    /// The pool with the fewest requests booked ([`LeastRequestsPicker`]).
    LeastRequests,
}

/// Selects a pool through the router core, then rotates over the pool's
/// frontends.
pub struct CoreSelector {
    service: Arc<SelectionService>,
    frontends: Cursors<PoolId>,
}

impl CoreSelector {
    /// Builds a core with `policy`. Nothing is selectable until
    /// [`Self::follow`] has applied the first pool snapshot.
    pub async fn new(policy: CorePolicy) -> Result<Self> {
        let service =
            SelectionServiceBuilder::new(config(policy)?, WorkerType::Aggregated, registry()?)
                .indexer_threads(1)
                .build()
                .await
                .context("build the router core")?;
        Ok(Self {
            service: Arc::new(service),
            frontends: Cursors::default(),
        })
    }

    /// Keeps the core's workers matching `pools`, the frontend's catalog
    /// subscription, until `cancel` fires or the catalog is dropped. Run the
    /// future as a task; the receiver resolves once the core has taken the
    /// first snapshot, so the caller can wait for the configured pools to be
    /// selectable before serving. It fails if the task ends first.
    pub fn follow<S>(
        &self,
        pools: watch::Receiver<Arc<S>>,
        cancel: CancellationToken,
    ) -> (
        oneshot::Receiver<()>,
        impl Future<Output = ()> + Send + 'static,
    )
    where
        S: AsRef<[Pool]> + Send + Sync + 'static,
    {
        let reconciler = CatalogReconciler::new(self.service.core().clone());
        let (source, ready) = PoolCatalogSource::with_ready(pools);
        (ready, reconciler.run(source, cancel))
    }

    /// Books a request on `pool`, whatever the policy would pick.
    #[cfg(test)]
    async fn reserve_on(&self, pool: &Pool) -> Reservation {
        use dynamo_kv_router::protocols::WorkerWithDpRank;

        let mut booking = booking(&pool.model, [pool].into_iter());
        booking.pinned_worker = Some(WorkerWithDpRank {
            worker_id: worker_id(&pool.id),
            dp_rank: 0,
        });
        let selected = self.service.select_and_reserve(booking).await.unwrap();
        Reservation::new(self.service.clone(), selected.selection_id)
    }

    #[cfg(test)]
    fn booked(&self, pool: &Pool) -> usize {
        self.service
            .loads(Some(&pool.model), Some(ROUTING_GROUP))
            .iter()
            .flat_map(|model| model.loads.iter())
            .find(|load| load.worker_id == worker_id(&pool.id))
            .map_or(0, |load| load.active_requests)
    }
}

#[async_trait]
impl RequestSelector for CoreSelector {
    async fn route<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<Selection<'a>> {
        let candidates = candidates
            .iter()
            .copied()
            .filter(|pool| reachable(pool))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }
        let selected = match self
            .service
            .select_and_reserve(booking(request.model, candidates.iter().copied()))
            .await
        {
            Ok(selected) => selected,
            Err(error) => {
                tracing::warn!(model = request.model, %error, "the router core selected no pool");
                return None;
            }
        };
        // Taken before the pool lookup so a worker the frontend cannot map
        // back is freed, not leaked.
        let reservation = Reservation::new(self.service.clone(), selected.selection_id);
        let Some(pool) = candidates
            .into_iter()
            .find(|pool| worker_id(&pool.id) == selected.worker_id)
        else {
            tracing::warn!(
                model = request.model,
                worker = selected.worker_id,
                "the router core chose a worker that is not a candidate pool"
            );
            return None;
        };
        let target = target(&self.frontends, pool)?;
        Some(Selection {
            target,
            lifecycle: Some(Box::new(reservation)),
        })
    }
}

/// A booking of a `model` request on one of `candidates`. The core sees no
/// prompt: a pool has no KV cache it could match, and the frontend has no
/// tokens, so every request is one block-free booking.
fn booking<'a>(model: &str, candidates: impl Iterator<Item = &'a Pool>) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: model.to_owned(),
        routing_group: ROUTING_GROUP.to_owned(),
        selection_id: None,
        prompt: PromptRequest {
            token_ids: Some(Vec::new()),
            ..PromptRequest::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        // Sessions reach the core once it keeps session affinity.
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: Some(candidates.map(|pool| worker_id(&pool.id)).collect()),
        routing_constraints: RoutingConstraints::default(),
    }
}

/// The core's configuration: no KV events, no queue, and `policy` for the
/// one worker pool.
///
/// The core reads its policy YAML from a file named by `router_policy_config`
/// and caches the parsed document; the cache is filled here so no file is
/// written.
fn config(policy: CorePolicy) -> Result<KvRouterConfig> {
    let parsed = RouterPolicyConfig::from_yaml(&policies::policy_config(policy))
        .context("parse the router policy configuration")?;
    let config = KvRouterConfig {
        use_kv_events: false,
        router_queue_threshold: None,
        router_policy_config: Some("<inline>".to_owned()),
        ..KvRouterConfig::default()
    };
    config
        .policy_config_cache
        .set(parsed)
        .map_err(|_| anyhow!("the router policy configuration was already loaded"))?;
    Ok(config)
}

fn registry() -> Result<dynamo_kv_router::services::selection::RouterPluginRegistry> {
    policies::registry().context("register the pool selection policies")
}

/// The core's booking of one request on its pool, held from selection until
/// the request ends. Dropping it frees the booking in the background.
pub struct Reservation {
    service: Arc<SelectionService>,
    selection_id: Option<String>,
}

impl Reservation {
    fn new(service: Arc<SelectionService>, selection_id: Option<String>) -> Self {
        if selection_id.is_none() {
            tracing::warn!("the router core booked a request without a selection id");
        }
        Self {
            service,
            selection_id,
        }
    }

    /// Frees the booking now and waits for it. Dropping frees it in the
    /// background instead.
    pub async fn release(mut self) {
        if let Some(selection_id) = self.selection_id.take() {
            free(&self.service, &selection_id).await;
        }
    }
}

impl RequestLifecycle for Reservation {}

impl Drop for Reservation {
    fn drop(&mut self) {
        let Some(selection_id) = self.selection_id.take() else {
            return;
        };
        let service = self.service.clone();
        spawn(async move { free(&service, &selection_id).await });
    }
}

async fn free(service: &SelectionService, selection_id: &str) {
    if let Err(error) = service.free_reservation(selection_id).await {
        tracing::warn!(%selection_id, %error, "the router core did not free a booking");
    }
}

/// Runs `task` on the current runtime; a drop outside one, as at process
/// exit, has no runtime to free on.
fn spawn(task: impl Future<Output = ()> + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => {
            runtime.spawn(task);
        }
        Err(_) => tracing::debug!("no runtime to finish a router core booking on"),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::selector::tests::pool;
    use crate::{InFlightTracker, LeastInFlightSelector, PoolSelector, RoundRobinSelector};

    use super::*;

    /// The catalog the core follows: dropping it ends the core's view of
    /// the pools.
    type Catalog = watch::Sender<Arc<Vec<Pool>>>;

    /// A core following `pools`, once every pool with a frontend is one of
    /// its workers.
    async fn selector(policy: CorePolicy, pools: &[Pool]) -> (CoreSelector, Catalog) {
        let (catalog, subscription) = watch::channel(Arc::new(pools.to_vec()));
        let selector = CoreSelector::new(policy).await.unwrap();
        let (ready, follower) = selector.follow(subscription, CancellationToken::new());
        tokio::spawn(follower);
        ready.await.expect("the core took the first snapshot");
        let expected = pools.iter().filter(|pool| reachable(pool)).count();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let registered = selector
                    .service
                    .list_workers(None, Some(ROUTING_GROUP))
                    .len();
                if registered == expected {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the pools became the core's workers");
        (selector, catalog)
    }

    async fn booked_settles(selector: &CoreSelector, pool: &Pool, booked: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while selector.booked(pool) != booked {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{} did not settle at {booked} booked", pool.name));
    }

    /// The pool `selector` routes a `model` request to, with the booking
    /// freed again before returning, as the legacy selectors count nothing
    /// per selection.
    async fn picks(selector: &CoreSelector, model: &str, candidates: &[&Pool]) -> Option<PoolId> {
        let selection = selector
            .route(&RoutingRequest::new(model), candidates)
            .await?;
        let pool = selection.target.pool;
        let booked = selector.booked(pool);
        drop(selection.lifecycle);
        booked_settles(selector, pool, booked - 1).await;
        Some(pool.id.clone())
    }

    async fn sequence(
        selector: &CoreSelector,
        model: &str,
        candidates: &[&Pool],
        n: usize,
    ) -> Vec<PoolId> {
        let mut picked = Vec::with_capacity(n);
        for _ in 0..n {
            picked.push(picks(selector, model, candidates).await.unwrap());
        }
        picked
    }

    /// The sequence as rounds of `size`, each sorted: the pools a round
    /// visits, whatever order it visits them in.
    fn rounds(sequence: &[PoolId], size: usize) -> Vec<Vec<PoolId>> {
        sequence
            .chunks(size)
            .map(|round| {
                let mut round = round.to_vec();
                round.sort_by(|a, b| a.as_str().cmp(b.as_str()));
                round
            })
            .collect()
    }

    #[tokio::test]
    async fn round_robin_visits_every_pool_once_per_round_like_the_legacy_selector() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::RoundRobin, &[blue.clone(), green.clone()]).await;
        let legacy = RoundRobinSelector::new();

        let from_core = sequence(&core, "qwen", &[&blue, &green], 6).await;
        let from_legacy = (0..6)
            .map(|_| {
                legacy
                    .select(&RoutingRequest::new("qwen"), &[&blue, &green])
                    .unwrap()
                    .pool
                    .id
                    .clone()
            })
            .collect::<Vec<_>>();

        // The core orders a round by worker id and the legacy selector by
        // catalog order, so rounds compare as sets.
        assert_eq!(rounds(&from_core, 2), rounds(&from_legacy, 2));
        assert_eq!(from_core[0], from_core[2]);
        assert_ne!(from_core[0], from_core[1]);
    }

    #[tokio::test]
    async fn round_robin_rotates_models_independently() {
        let qwen = [
            pool("qwen-a", "qwen", &["http://qwen-a:8000"]),
            pool("qwen-b", "qwen", &["http://qwen-b:8000"]),
        ];
        let llama = [
            pool("llama-a", "llama", &["http://llama-a:8000"]),
            pool("llama-b", "llama", &["http://llama-b:8000"]),
        ];
        let all = [qwen.clone(), llama.clone()].concat();
        let (core, _catalog) = selector(CorePolicy::RoundRobin, &all).await;
        let qwen = qwen.iter().collect::<Vec<_>>();
        let llama = llama.iter().collect::<Vec<_>>();

        let first_qwen = picks(&core, "qwen", &qwen).await.unwrap();
        let first_llama = picks(&core, "llama", &llama).await.unwrap();
        let second_qwen = picks(&core, "qwen", &qwen).await.unwrap();
        let second_llama = picks(&core, "llama", &llama).await.unwrap();

        assert_ne!(first_qwen, second_qwen);
        assert_ne!(first_llama, second_llama);
        assert!(qwen.iter().any(|pool| pool.id == first_qwen));
        assert!(llama.iter().any(|pool| pool.id == first_llama));
    }

    #[tokio::test]
    async fn skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let live = pool("live", "qwen", &["http://live:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::RoundRobin, &[empty.clone(), live.clone()]).await;

        for _ in 0..3 {
            assert_eq!(
                picks(&core, "qwen", &[&empty, &live]).await,
                Some(live.id.clone())
            );
        }
        assert_eq!(picks(&core, "qwen", &[&empty]).await, None);
        assert_eq!(picks(&core, "qwen", &[]).await, None);
    }

    #[tokio::test]
    async fn least_requests_picks_the_pool_with_fewest_bookings_like_the_legacy_selector() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::LeastRequests, &[blue.clone(), green.clone()]).await;
        let in_flight = Arc::new(InFlightTracker::new());
        let legacy = LeastInFlightSelector::new(in_flight.clone());

        let _busy = [core.reserve_on(&blue).await, core.reserve_on(&blue).await];
        let _one = core.reserve_on(&green).await;
        let _legacy_busy = [in_flight.start(&blue.id), in_flight.start(&blue.id)];
        let _legacy_one = in_flight.start(&green.id);
        assert_eq!((core.booked(&blue), core.booked(&green)), (2, 1));

        for _ in 0..10 {
            let from_legacy = legacy
                .select(&RoutingRequest::new("qwen"), &[&blue, &green])
                .unwrap()
                .pool
                .id
                .clone();
            assert_eq!(from_legacy, green.id);
            assert_eq!(
                picks(&core, "qwen", &[&blue, &green]).await,
                Some(from_legacy)
            );
        }
    }

    #[tokio::test]
    async fn least_requests_spreads_ties_and_skips_pools_without_frontends() {
        let empty = pool("draining", "qwen", &[]);
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let (core, _catalog) = selector(
            CorePolicy::LeastRequests,
            &[empty.clone(), blue.clone(), green.clone()],
        )
        .await;
        let legacy = LeastInFlightSelector::new(Arc::new(InFlightTracker::new()));

        let from_core = sequence(&core, "qwen", &[&empty, &blue, &green], 4).await;
        let from_legacy = (0..4)
            .map(|_| {
                legacy
                    .select(&RoutingRequest::new("qwen"), &[&empty, &blue, &green])
                    .unwrap()
                    .pool
                    .id
                    .clone()
            })
            .collect::<Vec<_>>();

        assert_eq!(rounds(&from_core, 2), rounds(&from_legacy, 2));
        assert_eq!(from_core[0], from_core[2]);
        assert_ne!(from_core[0], from_core[1]);
        assert_eq!(picks(&core, "qwen", &[&empty]).await, None);
    }

    #[tokio::test]
    async fn a_held_reservation_counts_until_it_is_released() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::LeastRequests, &[blue.clone(), green.clone()]).await;

        let on_blue = core.reserve_on(&blue).await;
        let selection = core
            .route(&RoutingRequest::new("qwen"), &[&blue, &green])
            .await
            .unwrap();
        assert_eq!(selection.target.pool.id, green.id);
        assert_eq!((core.booked(&blue), core.booked(&green)), (1, 1));

        drop(selection.lifecycle);
        booked_settles(&core, &green, 0).await;
        on_blue.release().await;
        assert_eq!((core.booked(&blue), core.booked(&green)), (0, 0));
    }
}
