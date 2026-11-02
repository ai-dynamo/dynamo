// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool selection through the embedded router core.
//!
//! The router core (`dynamo-kv-router`) selects a worker for each request.
//! Here each pool is one of its workers ([`catalog`]), the Global Router's
//! policies are its plugins ([`policies`]), and [`CoreSelector`] is the
//! [`PoolSelector`] that asks it.
//!
//! Every pick is the core's: [`CoreSelector`] sends an advisory select,
//! which books nothing, and the core runs the policy's plugin. Round-robin
//! and least-in-flight read nothing the core holds, so nothing is booked
//! for them.
//!
//! Least-in-flight ranks pools by the Global Frontend's requests in flight,
//! which it counts only once it forwards a request. So that picks racing
//! each other see one another, the plugin counts its pick as pending on the
//! pool ([`FrontendSignals`]); the core runs one model's picks one at a
//! time, so the next pick sees it. When the select returns, the pending
//! pick becomes the request's count in flight ([`Selection::in_flight`]).
//! A select that fails, or whose pool is not a candidate, gives the pick
//! back; one dropped before it returns finishes on a task of its own and
//! gives it back there ([`PendingSelect`]), so no pick stays pending.

mod catalog;
mod directory;
mod policies;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::scheduling::RouterPolicyConfig;
use dynamo_kv_router::services::selection::{
    CatalogReconciler, PromptRequest, SelectRequest, SelectResponse, SelectionError,
    SelectionService, SelectionServiceBuilder,
};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::frontend_signals::FrontendSignals;
use crate::selector::{Cursors, PoolSelector, Selection, reachable, target};
use crate::signals::PoolSignals;
use crate::{Pool, RoutingRequest};

pub use catalog::{PoolCatalogSource, ROUTING_GROUP, worker_id};
pub use directory::{PoolDirectory, PoolEntry};
pub use policies::{LeastInFlightPicker, RoundRobinPicker};

/// How the core picks among a model's pools.
#[derive(Clone, Debug)]
pub enum CorePolicy {
    /// Every pool in turn ([`RoundRobinPicker`]).
    RoundRobin,
    /// The nearest pool with the fewest requests in flight from every Global
    /// Frontend, counted by the frontends themselves, with affinity's
    /// placement rule ([`LeastInFlightPicker`]). Nothing is booked.
    LeastInFlight(LeastInFlightConfig),
}

/// What [`CorePolicy::LeastInFlight`] counts and weighs.
#[derive(Clone, Debug)]
pub struct LeastInFlightConfig {
    /// The Global Frontends' requests in flight per pool.
    pub counts: Arc<FrontendSignals>,
    /// Each pool's capacity, from its Relay, and the round trip to each
    /// region.
    pub signals: Arc<PoolSignals>,
    /// The region this process runs in, which every request enters.
    pub region: String,
    /// A request leaves the nearest, least loaded pool only for one with
    /// more than this many times its spare room.
    pub headroom_factor: f64,
}

/// Selects a pool through the router core, then rotates over the pool's
/// frontends.
pub struct CoreSelector {
    service: Arc<SelectionService>,
    directory: Arc<PoolDirectory>,
    frontends: Cursors<PoolId>,
    /// Where the policy's picks are pending until their requests take them
    /// over: least-in-flight's counts.
    pending: Option<Arc<FrontendSignals>>,
}

impl CoreSelector {
    /// Builds a core with `policy`. Nothing is selectable until
    /// [`Self::reconcile_catalog`] has applied the first pool snapshot.
    pub async fn new(policy: CorePolicy) -> Result<Self> {
        let directory = Arc::new(PoolDirectory::new());
        let registry = policies::registry(&policy, directory.clone())
            .context("register the pool selection policies")?;
        let builder =
            SelectionServiceBuilder::new(config(&policy)?, WorkerType::Aggregated, registry)
                .indexer_threads(1);
        let service = Arc::new(builder.build().await.context("build the router core")?);
        let pending = match &policy {
            CorePolicy::LeastInFlight(config) => Some(config.counts.clone()),
            CorePolicy::RoundRobin => None,
        };
        Ok(Self {
            service,
            directory,
            frontends: Cursors::default(),
            pending,
        })
    }

    /// Keeps the core's workers matching `pools`, the frontend's catalog
    /// subscription, until `cancel` fires or the catalog is dropped: it runs
    /// the core's `CatalogReconciler` over a [`PoolCatalogSource`], as the
    /// local router does over its discovery source. Run the
    /// future as a task; the receiver resolves once the core has taken the
    /// first snapshot, so the caller can wait for the configured pools to be
    /// selectable before serving. It fails if the task ends first.
    pub fn reconcile_catalog<S>(
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
        let (source, first_taken) = PoolCatalogSource::with_ready(pools);
        let source = source.with_directory(self.directory.clone());
        let reconcile = async move { reconciler.run(source, cancel).await };
        (first_taken, reconcile)
    }
}

#[async_trait]
impl PoolSelector for CoreSelector {
    async fn select<'a>(
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
        let selected = self.pick(request.model, &candidates).await?;
        let Some(pool) = candidates
            .into_iter()
            .find(|pool| worker_id(&pool.id) == selected.worker_id)
        else {
            if let Some(pending) = &self.pending {
                pending.release_pending(selected.worker_id);
            }
            tracing::warn!(
                model = request.model,
                worker = selected.worker_id,
                "the router core chose a worker that is not a candidate pool"
            );
            return None;
        };
        let in_flight = self
            .pending
            .as_ref()
            .map(|pending| pending.take_pending(&pool.id));
        let target = target(&self.frontends, pool)?;
        tracing::trace!(model = request.model, pool = %pool.name, "the router core picked a pool");
        Some(Selection {
            target,
            lifecycle: None,
            in_flight,
        })
    }
}

impl CoreSelector {
    /// The core's pick among `candidates`. It books nothing; under a policy
    /// whose picks are pending, the pick is pending until the caller takes
    /// it over or gives it back.
    async fn pick(&self, model: &str, candidates: &[&Pool]) -> Option<SelectResponse> {
        let request = advisory(model, candidates.iter().copied());
        let selected = match &self.pending {
            None => self.service.select(request).await,
            Some(pending) => {
                let service = self.service.clone();
                let select = Box::pin(async move { service.select(request).await });
                PendingSelect::new(select, pending.clone()).finish().await
            }
        };
        selected
            .map_err(|error| tracing::warn!(model, %error, "the router core selected no pool"))
            .ok()
    }
}

type Select = Pin<Box<dyn Future<Output = Result<SelectResponse, SelectionError>> + Send>>;

/// An advisory select whose pick may be pending. Dropped before it returns,
/// as when the client leaves, it finishes the select on a task of its own
/// and gives the pick back there: the core may already have picked, or may
/// pick still, so the pick is released only once the select returns.
struct PendingSelect {
    select: Option<Select>,
    pending: Arc<FrontendSignals>,
}

impl PendingSelect {
    fn new(select: Select, pending: Arc<FrontendSignals>) -> Self {
        Self {
            select: Some(select),
            pending,
        }
    }

    /// The select's result; the pick, if any, is the caller's to take over
    /// or give back.
    async fn finish(mut self) -> Result<SelectResponse, SelectionError> {
        let select = self.select.as_mut().expect("finished once");
        let selected = select.await;
        self.select = None;
        selected
    }
}

impl Drop for PendingSelect {
    fn drop(&mut self) {
        let Some(select) = self.select.take() else {
            return;
        };
        let pending = self.pending.clone();
        // Without a runtime the core is gone too, and its picks with it.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Ok(selected) = select.await {
                    pending.release_pending(selected.worker_id);
                }
            });
        }
    }
}

/// A pick among `candidates` that books nothing: no queue admission, no
/// reservation, no sequence tracking, nothing to free. The core sees no
/// prompt: a pool has no KV cache it could match, and the frontend has no
/// tokens.
fn advisory<'a>(model: &str, candidates: impl Iterator<Item = &'a Pool>) -> SelectRequest {
    SelectRequest {
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
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: Some(candidates.map(|pool| worker_id(&pool.id)).collect()),
        routing_constraints: RoutingConstraints::default(),
        advisory: true,
    }
}

/// The core's configuration: no KV events, no queue, and `policy` for the
/// one worker pool.
///
/// The core reads its policy YAML from a file named by `router_policy_config`
/// and caches the parsed document; the cache is filled here so no file is
/// written.
fn config(policy: &CorePolicy) -> Result<KvRouterConfig> {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use url::Url;

    use crate::selector::tests::pool;
    use crate::{InFlightTracker, RoundRobinSelector};

    use super::*;

    /// The catalog the core follows: dropping it ends the core's view of
    /// the pools.
    type Catalog = watch::Sender<Arc<Vec<Pool>>>;

    /// A core following `pools`, once every pool with a frontend is one of
    /// its workers.
    async fn selector(policy: CorePolicy, pools: &[Pool]) -> (CoreSelector, Catalog) {
        let (catalog, subscription) = watch::channel(Arc::new(pools.to_vec()));
        let selector = CoreSelector::new(policy).await.unwrap();
        let (first_snapshot, reconcile) =
            selector.reconcile_catalog(subscription, CancellationToken::new());
        tokio::spawn(reconcile);
        first_snapshot
            .await
            .expect("the core took the first snapshot");
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

    /// The pool `selector` routes a `model` request to.
    async fn picks(selector: &CoreSelector, model: &str, candidates: &[&Pool]) -> Option<PoolId> {
        let selection = selector
            .select(&RoutingRequest::new(model), candidates)
            .await?;
        Some(selection.target.pool.id.clone())
    }

    /// The core lists the candidates in its own order, not the request's,
    /// so it rotates in another order than the current selector: each round
    /// visits the same pools, and each pool's frontends rotate alike.
    #[tokio::test]
    async fn round_robin_picks_as_the_current_selector_in_the_core_s_order() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let red = pool("red", "qwen", &["http://red-0:8000"]);
        let (core, _catalog) = selector(
            CorePolicy::RoundRobin,
            &[blue.clone(), green.clone(), red.clone()],
        )
        .await;
        let current = RoundRobinSelector::new();

        // Red loses its frontend for a while, then gets it back.
        let mut red_down = red.clone();
        red_down.frontends.clear();
        let all = [&blue, &green, &red];
        let without_red = [&blue, &green, &red_down];
        let phases = [(&all, 3), (&without_red, 2), (&all, 3)];
        let sorted = |mut picks: Vec<(PoolId, Url)>| {
            picks.sort_by(|a, b| (a.0.as_str(), a.1.as_str()).cmp(&(b.0.as_str(), b.1.as_str())));
            picks
        };
        for (phase, (candidates, rounds)) in phases.into_iter().enumerate() {
            let reachable = candidates.iter().filter(|pool| reachable(pool)).count();
            for round in 0..rounds {
                let request = RoutingRequest::new("qwen");
                let (mut from_core, mut from_current) = (Vec::new(), Vec::new());
                for _ in 0..reachable {
                    let target = core.select(&request, candidates.as_slice()).await;
                    let target = target.unwrap().target;
                    from_core.push((target.pool.id.clone(), target.frontend.clone()));
                    let target = current.select(&request, candidates.as_slice()).await;
                    let target = target.unwrap().target;
                    from_current.push((target.pool.id.clone(), target.frontend.clone()));
                }
                assert_eq!(
                    sorted(from_core),
                    sorted(from_current),
                    "phase {phase}, round {round}"
                );
            }
        }
    }

    /// The same pool as the current selector whenever one has the fewest
    /// requests; a tie may break the other way, since the core lists the
    /// candidates in its own order.
    #[tokio::test]
    async fn least_in_flight_picks_as_the_current_selector() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let red = pool("red", "qwen", &["http://red:8000"]);
        let candidates = [&blue, &green, &red];
        let in_flight = Arc::new(InFlightTracker::new());
        let (core, _catalog) = selector(
            least_in_flight(&in_flight),
            &[blue.clone(), green.clone(), red.clone()],
        )
        .await;
        let current = crate::selector::tests::least_in_flight(&in_flight);

        let mut held = Vec::new();
        for turn in 0..12 {
            match turn {
                // Green and red busy: blue has the fewest.
                0 => held.extend([in_flight.start(&green.id), in_flight.start(&red.id)]),
                // Blue busier than both: green and red tie.
                4 => held.extend([in_flight.start(&blue.id), in_flight.start(&blue.id)]),
                // Red busiest: blue and green tie.
                8 => held.extend([in_flight.start(&red.id), in_flight.start(&red.id)]),
                _ => {}
            }
            let request = RoutingRequest::new("qwen");
            let Selection {
                target,
                in_flight: counted,
                ..
            } = core.select(&request, &candidates).await.unwrap();
            // Compared before either counts the request.
            drop(counted);
            let from_current = current.select(&request, &candidates).await.unwrap().target;
            let fewest = candidates
                .iter()
                .map(|pool| in_flight.get(&pool.id))
                .min()
                .unwrap();
            let tied = candidates
                .iter()
                .filter(|pool| in_flight.get(&pool.id) == fewest)
                .count();
            if tied == 1 {
                assert_eq!(target.pool.id, from_current.pool.id, "turn {turn}");
            } else {
                assert_eq!(in_flight.get(&target.pool.id), fewest, "turn {turn}");
            }
        }
    }

    /// Least-in-flight over `in_flight`, the tracker its frontend counts in.
    fn least_in_flight(in_flight: &Arc<InFlightTracker>) -> CorePolicy {
        least_in_flight_counting(&Arc::new(FrontendSignals::new(in_flight.clone())))
    }

    /// Least-in-flight ranking by `counts`, in Ohio, with no pool's capacity
    /// or round trip known.
    fn least_in_flight_counting(counts: &Arc<FrontendSignals>) -> CorePolicy {
        least_in_flight_weighing(
            counts,
            &Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new()))),
        )
    }

    /// Least-in-flight ranking by `counts`, in Ohio, weighing `signals`.
    fn least_in_flight_weighing(
        counts: &Arc<FrontendSignals>,
        signals: &Arc<PoolSignals>,
    ) -> CorePolicy {
        CorePolicy::LeastInFlight(LeastInFlightConfig {
            counts: counts.clone(),
            signals: signals.clone(),
            region: "us-east-2".into(),
            headroom_factor: 2.0,
        })
    }

    /// The same pool as the current selector on the same counts, capacities
    /// and round trips: the nearest until another region has much more room.
    #[tokio::test]
    async fn least_in_flight_places_by_region_like_the_current_selector() {
        use crate::selector::tests::{least_in_flight_with, regional_pool, set_capacity};

        let ohio = regional_pool("us-east-2", "ohio", "qwen", &["http://ohio:8000"]);
        let tokyo = regional_pool("ap-northeast-1", "tokyo", "qwen", &["http://tokyo:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let signals = Arc::new(PoolSignals::new(in_flight.clone()));
        signals.set_region_round_trips(std::collections::HashMap::from([(
            "ap-northeast-1".to_owned(),
            Duration::from_millis(135),
        )]));
        set_capacity(&signals, &ohio, 100);
        set_capacity(&signals, &tokyo, 100);
        let counts = Arc::new(FrontendSignals::new(in_flight.clone()));
        let (core, _catalog) = selector(
            least_in_flight_weighing(&counts, &signals),
            &[ohio.clone(), tokyo.clone()],
        )
        .await;
        let current = least_in_flight_with(&in_flight, &signals);
        let candidates = [&tokyo, &ohio];

        let mut held = Vec::new();
        for (busy, expected) in [(5, &ohio), (50, &tokyo)] {
            held.extend((0..busy).map(|_| in_flight.start(&ohio.id)));
            let request = RoutingRequest::new("qwen");
            let from_core = core.select(&request, &candidates).await.unwrap();
            let pool = from_core.target.pool.id.clone();
            drop(from_core);
            let from_current = current.select(&request, &candidates).await.unwrap().target;
            assert_eq!(pool, expected.id);
            assert_eq!(from_current.pool.id, expected.id);
        }
    }

    #[tokio::test]
    async fn least_in_flight_counts_the_frontends_requests_like_the_current_selector_and_books_nothing()
     {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let (core, _catalog) =
            selector(least_in_flight(&in_flight), &[blue.clone(), green.clone()]).await;
        let current = crate::selector::tests::least_in_flight(&in_flight);

        let busy = [in_flight.start(&blue.id), in_flight.start(&blue.id)];
        let one = in_flight.start(&green.id);
        for _ in 0..10 {
            let request = RoutingRequest::new("qwen");
            let Selection {
                target,
                lifecycle,
                in_flight: counted,
            } = core.select(&request, &[&blue, &green]).await.unwrap();
            assert!(lifecycle.is_none(), "least-in-flight books nothing");
            // Compared as the current selector would pick for the same
            // request, before either counts it.
            drop(counted);
            let from_current = current
                .select(&request, &[&blue, &green])
                .await
                .unwrap()
                .target;
            assert_eq!(target.pool.id, from_current.pool.id);
            assert_eq!(target.pool.id, green.id);
        }

        // Blue's requests end: blue has fewer than green's one.
        drop(busy);
        assert_eq!(
            picks(&core, "qwen", &[&blue, &green]).await,
            Some(blue.id.clone())
        );
        // Tied once green's ends too: the picks alternate.
        drop(one);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            seen.insert(picks(&core, "qwen", &[&blue, &green]).await.unwrap());
        }
        assert_eq!(seen.len(), 2);
    }

    /// Picks that race see each other: each is pending on its pool from the
    /// core's pick until its request counts it, and the core runs a model's
    /// picks one at a time.
    #[tokio::test]
    async fn concurrent_least_in_flight_picks_see_each_other() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let counts = Arc::new(FrontendSignals::new(in_flight.clone()));
        let (core, _catalog) = selector(
            least_in_flight_counting(&counts),
            &[blue.clone(), green.clone()],
        )
        .await;
        let _busy = in_flight.start(&green.id);

        // Three selects at once, none forwarded yet. The first goes to idle
        // blue and the next two split once blue has as many as green;
        // without the pending picks, all three would go to blue.
        let request = RoutingRequest::new("qwen");
        let candidates = [&blue, &green];
        let (a, b, c) = tokio::join!(
            core.select(&request, &candidates),
            core.select(&request, &candidates),
            core.select(&request, &candidates),
        );
        let held = [a.unwrap(), b.unwrap(), c.unwrap()];
        assert!(held.iter().all(|selection| selection.in_flight.is_some()));
        let to = |pool: &Pool| {
            held.iter()
                .filter(|selection| selection.target.pool.id == pool.id)
                .count()
        };
        assert_eq!((to(&blue), to(&green)), (2, 1));
        assert_eq!((in_flight.get(&blue.id), in_flight.get(&green.id)), (2, 2));
        assert_eq!(counts.pending(), 0, "the requests took their picks over");

        // The count lasts as long as the selection's guard.
        drop(held);
        assert_eq!((in_flight.get(&blue.id), in_flight.get(&green.id)), (0, 1));
    }

    /// A select dropped before it returns, as when the client leaves, leaves
    /// no pick pending, whether or not the core picked.
    #[tokio::test]
    async fn an_abandoned_select_leaves_no_pick_pending() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let in_flight = Arc::new(InFlightTracker::new());
        let counts = Arc::new(FrontendSignals::new(in_flight.clone()));
        let (core, _catalog) = selector(
            least_in_flight_counting(&counts),
            &[blue.clone(), green.clone()],
        )
        .await;
        let request = RoutingRequest::new("qwen");
        let candidates = [&blue, &green];

        // Sent to the core, then dropped; and dropped before it was sent.
        for _ in 0..8 {
            let mut select = std::pin::pin!(core.select(&request, &candidates));
            // Polled once, so the request reaches the core, then dropped.
            tokio::select! {
                biased;
                _ = select.as_mut() => panic!("the core answers on its own task"),
                () = std::future::ready(()) => {}
            }
        }
        drop(core.select(&request, &candidates));
        // A select that completes runs after the abandoned ones' picks.
        let finished = core.select(&request, &candidates).await.unwrap();
        drop(finished);

        tokio::time::timeout(Duration::from_secs(10), async {
            while counts.pending() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("every abandoned pick was given back");
        assert_eq!((in_flight.get(&blue.id), in_flight.get(&green.id)), (0, 0));
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
}
