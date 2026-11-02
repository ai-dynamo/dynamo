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
//!
//! With [`CorePolicy::Affinity`] the core also keeps the session table:
//! each request carries its session id, the core holds the session's
//! binding across the pick and binds it to the pool picked, and the
//! [`policies::HeadroomPicker`] sees the binding as the request's affinity
//! target. A session books on its first request, a move, and a refresh; the
//! booking comes back as a [`Reservation`], which the forwarding engine
//! holds until the response ends.

mod catalog;
mod directory;
mod policies;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use dynamo_global_common::PoolId;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::multi_worker_sequence::ReplicaRequestLeaseObserver;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::scheduling::RouterPolicyConfig;
use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_ENTRIES, SessionAffinityMode,
};
use dynamo_kv_router::services::selection::{
    CatalogReconciler, HostReplication, PromptRequest, SelectAndReserveRequest, SelectRequest,
    SelectResponse, SelectionError, SelectionHost, SelectionService, SelectionServiceBuilder,
};
use dynamo_kv_router::{WorkerId, WorkerType};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::frontend_signals::FrontendSignals;
use crate::selector::{Cursors, PoolSelector, RequestLifecycle, Selection, reachable, target};
use crate::signals::PoolSignals;
use crate::{Pool, RoutingRequest};

pub use catalog::{PoolCatalogSource, ROUTING_GROUP, worker_id};
pub use directory::{PoolDirectory, PoolEntry};
/// The longest session id the core's session table accepts; the frontend
/// rejects longer ones before selection, so they are a client error and not
/// an unavailable pool.
pub use dynamo_kv_router::services::selection::affinity::MAX_SESSION_AFFINITY_ID_BYTES as MAX_SESSION_ID_BYTES;
pub use policies::{HeadroomPicker, LeastInFlightPicker, RoundRobinPicker};

/// How the core picks among a model's pools.
#[derive(Clone, Debug)]
pub enum CorePolicy {
    /// Every pool in turn ([`RoundRobinPicker`]).
    RoundRobin,
    /// The nearest pool with the fewest requests in flight from every Global
    /// Frontend, counted by the frontends themselves, with affinity's
    /// placement rule ([`LeastInFlightPicker`]). Nothing is booked.
    LeastInFlight(LeastInFlightConfig),
    /// Each session on one pool, moved only for much more room
    /// ([`HeadroomPicker`] over the core's session table).
    Affinity(AffinityConfig),
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

/// What [`CorePolicy::Affinity`] weighs, as [`crate::AffinitySelector`]
/// is configured.
#[derive(Clone, Debug)]
pub struct AffinityConfig {
    /// Each pool's occupancy and each region's round trip.
    pub signals: Arc<PoolSignals>,
    /// The region this process runs in, which every request enters.
    pub region: String,
    /// A request leaves its pool only for one with more than this many
    /// times its spare capacity.
    pub headroom_factor: f64,
    /// How long a session keeps its pool without a request.
    pub idle: Duration,
}

/// Selects a pool through the router core, then rotates over the pool's
/// frontends.
pub struct CoreSelector {
    service: Arc<SelectionService>,
    directory: Arc<PoolDirectory>,
    frontends: Cursors<PoolId>,
    /// Requests the core has booked, for the frontend's metrics.
    pin_bookings: std::sync::atomic::AtomicU64,
    /// Whether requests carry their session to the core, which only keeps
    /// sessions under [`CorePolicy::Affinity`].
    sessions: bool,
    /// Under affinity, when each session's pin was last booked.
    pins: Option<PinRefresh>,
    /// Where the policy's picks are pending until their requests take them
    /// over: least-in-flight's counts.
    pending: Option<Arc<FrontendSignals>>,
}

/// When each session's pin was last booked, so only the requests that set,
/// move or keep alive a pin book. The core binds a session only on a
/// booking, and a pin's idle time runs from its last booking's release, so
/// a session books on its first request, whenever the picker moves it, and
/// at least once per half idle window; the requests between pick with an
/// advisory selection that honours the pin and books nothing.
struct PinRefresh {
    refresh: Duration,
    /// The most sessions remembered: the core keeps no more pins than this.
    capacity: usize,
    booked: parking_lot::Mutex<Pins>,
}

#[derive(Default)]
struct Pins {
    by_session: HashMap<String, (WorkerId, Instant)>,
    /// When stale pins were last dropped, so a full table is swept at most
    /// once a refresh window rather than on every new session.
    swept: Option<Instant>,
}

impl PinRefresh {
    fn new(idle: Duration) -> Self {
        Self::with_capacity(idle, MAX_SESSION_AFFINITY_ENTRIES)
    }

    fn with_capacity(idle: Duration, capacity: usize) -> Self {
        Self {
            refresh: idle / 2,
            capacity,
            booked: parking_lot::Mutex::default(),
        }
    }

    /// The pool `session` was last booked on, if that was within the
    /// refresh window as of `now`.
    fn fresh(&self, session: &str, now: Instant) -> Option<WorkerId> {
        let booked = self.booked.lock();
        let &(worker, at) = booked.by_session.get(session)?;
        (now.saturating_duration_since(at) < self.refresh).then_some(worker)
    }

    /// Remembers that `session` booked on `worker` at `now`. A full table
    /// drops its stale pins, at most once a refresh window; if it is still
    /// full, a new session is not remembered and books on every request,
    /// as it would were the core's own table full.
    fn record(&self, session: &str, worker: WorkerId, now: Instant) {
        let mut booked = self.booked.lock();
        let Pins { by_session, swept } = &mut *booked;
        if by_session.len() >= self.capacity && !by_session.contains_key(session) {
            let refresh = self.refresh;
            if swept.is_none_or(|at| now.saturating_duration_since(at) >= refresh) {
                by_session.retain(|_, (_, at)| now.saturating_duration_since(*at) < refresh);
                *swept = Some(now);
            }
            if by_session.len() >= self.capacity {
                return;
            }
        }
        by_session.insert(session.to_owned(), (worker, now));
    }
}

impl CoreSelector {
    /// Builds a core with `policy`. Nothing is selectable until
    /// [`Self::reconcile_catalog`] has applied the first pool snapshot.
    pub async fn new(policy: CorePolicy) -> Result<Self> {
        let directory = Arc::new(PoolDirectory::new());
        let registry = policies::registry(&policy, directory.clone())
            .context("register the pool selection policies")?;
        let mut builder =
            SelectionServiceBuilder::new(config(&policy)?, WorkerType::Aggregated, registry)
                .indexer_threads(1);
        if let CorePolicy::Affinity(affinity) = &policy {
            // Soft: the binding follows the pool picked, as the current pin
            // moves with the request.
            builder = builder
                .session_affinity(affinity.idle)
                .session_affinity_mode(SessionAffinityMode::Soft);
        }
        // Every booking is local and lasts as long as its request: the host
        // holds the lease, so the core never expires it by age.
        builder = builder.host(SelectionHost {
            replication: HostReplication {
                request_leases: Some(Arc::new(ReservationLeases)),
                ..HostReplication::default()
            },
            ..SelectionHost::default()
        });
        let service = Arc::new(builder.build().await.context("build the router core")?);
        let pending = match &policy {
            CorePolicy::LeastInFlight(config) => Some(config.counts.clone()),
            _ => None,
        };
        Ok(Self {
            pin_bookings: std::sync::atomic::AtomicU64::new(0),
            service,
            directory,
            frontends: Cursors::default(),
            sessions: matches!(policy, CorePolicy::Affinity(_)),
            pins: match &policy {
                CorePolicy::Affinity(affinity) => Some(PinRefresh::new(affinity.idle)),
                _ => None,
            },
            pending,
        })
    }

    /// Affinity pin bookings since start: a session's first request, a move,
    /// and a refresh. Round-robin and least-in-flight book nothing.
    pub fn pin_bookings(&self) -> u64 {
        self.pin_bookings.load(std::sync::atomic::Ordering::Relaxed)
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

    /// Books a request on `pool`, whatever the policy would pick.
    #[cfg(test)]
    async fn reserve_on(&self, pool: &Pool) -> Reservation {
        use dynamo_kv_router::protocols::WorkerWithDpRank;

        let mut booking = booking(&pool.model, [pool].into_iter(), None);
        booking.pinned_worker = Some(WorkerWithDpRank {
            worker_id: worker_id(&pool.id),
            dp_rank: 0,
        });
        let selected = self.service.select_and_reserve(booking).await.unwrap();
        Reservation::new(self.service.clone(), selected.selection_id)
    }

    /// Requests booked on `pool`, this replica's and its peers'.
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
        let session = request.session.filter(|_| self.sessions);
        let now = Instant::now();
        let pinned = session
            .zip(self.pins.as_ref())
            .and_then(|(session, pins)| pins.fresh(session, now));
        // A session books when its pin is new or due for a refresh.
        let mut book = session.is_some() && pinned.is_none();
        let mut selected = self.pick(request.model, &candidates, session, book).await?;
        if !book && pinned.is_some_and(|pinned| pinned != selected.worker_id) {
            // The picker moved the session: only a booking moves its pin.
            book = true;
            selected = self.pick(request.model, &candidates, session, true).await?;
        }
        if let (true, Some(session), Some(pins)) = (book, session, &self.pins) {
            pins.record(session, selected.worker_id, now);
        }
        // Taken before the pool lookup so a worker the frontend cannot map
        // back is freed, not leaked.
        let reservation =
            book.then(|| Reservation::new(self.service.clone(), selected.selection_id.clone()));
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
        tracing::trace!(
            model = request.model,
            session,
            pool = %pool.name,
            "the router core picked a pool"
        );
        Some(Selection {
            target,
            lifecycle: reservation
                .map(|reservation| Box::new(reservation) as Box<dyn RequestLifecycle>),
            in_flight,
        })
    }
}

impl CoreSelector {
    /// The core's pick among `candidates`, booked or advisory. Under a
    /// policy whose picks are pending, an advisory pick is pending until the
    /// caller takes it over or gives it back.
    async fn pick(
        &self,
        model: &str,
        candidates: &[&Pool],
        session: Option<&str>,
        book: bool,
    ) -> Option<SelectResponse> {
        let selected = if book {
            self.pin_bookings
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.service
                .select_and_reserve(booking(model, candidates.iter().copied(), session))
                .await
        } else {
            let request = advisory(model, candidates.iter().copied(), session);
            match &self.pending {
                None => self.service.select(request).await,
                Some(pending) => {
                    let service = self.service.clone();
                    let select = Box::pin(async move { service.select(request).await });
                    PendingSelect::new(select, pending.clone()).finish().await
                }
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
/// reservation, no sequence tracking, nothing to free. A `session`'s pin is
/// honoured but neither set nor refreshed.
fn advisory<'a>(
    model: &str,
    candidates: impl Iterator<Item = &'a Pool>,
    session: Option<&str>,
) -> SelectRequest {
    let booking = booking(model, candidates, session);
    SelectRequest {
        model_name: booking.model_name,
        routing_group: booking.routing_group,
        selection_id: None,
        prompt: booking.prompt,
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: booking.session_id,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: booking.allowed_worker_ids,
        routing_constraints: booking.routing_constraints,
        advisory: true,
    }
}

/// A booking of a `model` request on one of `candidates`, for `session`
/// if the core keeps sessions. The core sees no prompt: a pool has no KV
/// cache it could match, and the frontend has no tokens, so every request
/// is one block-free booking.
fn booking<'a>(
    model: &str,
    candidates: impl Iterator<Item = &'a Pool>,
    session: Option<&str>,
) -> SelectAndReserveRequest {
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
        session_id: session.map(str::to_owned),
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

/// The host owns each booking's lifetime: a booking lasts as long as its
/// [`Reservation`], which the frontend holds until the response ends or the
/// client leaves, never a fixed age. Supplying this to the core turns its
/// age-based expiry off; bookings are never mirrored to peers, so there is
/// nothing to observe.
struct ReservationLeases;

impl ReplicaRequestLeaseObserver for ReservationLeases {
    fn admitted(&self, _: SchedulerBookingDescriptor) {}
    fn progressed(&self, _: &SchedulerBookingDescriptor) {}
    fn completed(&self, _: &SchedulerBookingDescriptor) {}
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
    use url::Url;

    use crate::affinity::tests::{OHIO, fleet, set_occupancy};
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

    /// The affinity policy of [`crate::AffinitySelector`]'s tests: Ohio,
    /// factor 2, pins idle for a minute.
    fn affinity(signals: &Arc<PoolSignals>) -> CorePolicy {
        CorePolicy::Affinity(AffinityConfig {
            signals: signals.clone(),
            region: OHIO.into(),
            headroom_factor: 2.0,
            idle: Duration::from_secs(60),
        })
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

    /// The pool `selector` routes a `model` request of `session` to, with
    /// any booking freed again before returning, as the current selectors
    /// count nothing per selection. The session's pin stays.
    async fn picks_for(
        selector: &CoreSelector,
        model: &str,
        candidates: &[&Pool],
        session: Option<&str>,
    ) -> Option<PoolId> {
        let request = RoutingRequest::new(model).with_session(session);
        let selection = selector.select(&request, candidates).await?;
        let pool = selection.target.pool;
        if let Some(lifecycle) = selection.lifecycle {
            let booked = selector.booked(pool);
            drop(lifecycle);
            booked_settles(selector, pool, booked - 1).await;
        }
        Some(pool.id.clone())
    }

    async fn picks(selector: &CoreSelector, model: &str, candidates: &[&Pool]) -> Option<PoolId> {
        picks_for(selector, model, candidates, None).await
    }

    /// The host of the first frontend of the pool `selector` picks for
    /// `session` among `pools`.
    async fn host_for(selector: &CoreSelector, pools: &[Pool], session: Option<&str>) -> String {
        let candidates = pools.iter().collect::<Vec<_>>();
        let picked = picks_for(selector, "qwen", &candidates, session)
            .await
            .expect("a pool was picked");
        let pool = pools.iter().find(|pool| pool.id == picked).unwrap();
        pool.frontends[0].host_str().unwrap().to_owned()
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
        assert_eq!((core.booked(&blue), core.booked(&green)), (0, 0));

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
    async fn round_robin_books_nothing() {
        let blue = pool("blue", "qwen", &["http://blue-0:8000"]);
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::RoundRobin, &[blue.clone(), green.clone()]).await;
        let held = [
            core.select(&RoutingRequest::new("qwen"), &[&blue, &green])
                .await
                .unwrap(),
            core.select(&RoutingRequest::new("qwen"), &[&blue, &green])
                .await
                .unwrap(),
        ];
        assert!(held.iter().all(|selection| selection.lifecycle.is_none()));
        assert_eq!((core.booked(&blue), core.booked(&green)), (0, 0));
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
    async fn a_held_reservation_counts_until_it_is_released() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        let candidates = pools.iter().collect::<Vec<_>>();

        // A new session books, and the booking lasts as long as the
        // selection's lifecycle; a reservation as long as it is held.
        let request = RoutingRequest::new("qwen").with_session(Some("s"));
        let selection = core.select(&request, &candidates).await.unwrap();
        let pool = selection.target.pool;
        assert_eq!(core.booked(pool), 1);
        let other = &pools[3];
        let held = core.reserve_on(other).await;
        assert_eq!(core.booked(other), 1);

        drop(selection.lifecycle);
        booked_settles(&core, pool, 0).await;
        held.release().await;
        assert_eq!(core.booked(other), 0);
    }

    // The affinity cases below replay `affinity.rs`'s tests through the
    // core: same fleet, same occupancies, same expected pools.

    #[tokio::test]
    async fn affinity_moves_for_twice_the_room_and_does_not_flip_back() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");

        // 10% against 2%: 0.9 spare against 0.98, nothing to gain.
        set_occupancy(&pools, &signals, [0.1, 0.02, 0.02, 0.02]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");
        // 50% against 30%: 0.5 spare against 0.7, under twice.
        set_occupancy(&pools, &signals, [0.5, 0.3, 0.3, 0.3]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");
        // 80% against 10%: 0.2 against 0.9, so to the nearest such pool.
        set_occupancy(&pools, &signals, [0.8, 0.1, 0.1, 0.1]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-b");
        // Ohio-a recovers to 40% and Ohio-b fills to 60%: 0.4 spare
        // against 0.6 is under twice, so no flip back.
        set_occupancy(&pools, &signals, [0.4, 0.6, 0.6, 0.6]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-b");
    }

    #[test]
    fn a_pin_is_due_for_a_booking_after_half_its_idle_window() {
        let pins = PinRefresh::new(Duration::from_secs(60));
        let start = Instant::now();
        assert_eq!(pins.fresh("s", start), None);
        pins.record("s", 7, start);
        assert_eq!(pins.fresh("s", start + Duration::from_secs(29)), Some(7));
        assert_eq!(pins.fresh("s", start + Duration::from_secs(30)), None);
        assert_eq!(pins.fresh("t", start), None);
    }

    #[tokio::test]
    async fn affinity_books_only_to_set_move_or_refresh_a_pin() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        let candidates = pools.iter().collect::<Vec<_>>();
        let model = pools[0].model.clone();
        let route = |session: Option<&'static str>| {
            let request = RoutingRequest::new(&model).with_session(session);
            let core = &core;
            let candidates = &candidates;
            async move {
                let selection = core.select(&request, candidates).await.unwrap();
                let booked = selection.lifecycle.is_some();
                let pool = &selection.target.pool.name;
                (pool.rsplit('/').next().unwrap_or(pool).to_owned(), booked)
            }
        };
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);

        // The first request sets the pin; the next ones only read it.
        assert_eq!(route(Some("s")).await, ("ohio-a".to_owned(), true));
        assert_eq!(route(Some("s")).await, ("ohio-a".to_owned(), false));
        assert_eq!(route(Some("s")).await, ("ohio-a".to_owned(), false));
        // No session, no pin to keep: nothing is booked.
        assert!(!route(None).await.1);

        // A move books, so the pin follows the session.
        set_occupancy(&pools, &signals, [0.8, 0.1, 0.1, 0.1]);
        assert_eq!(route(Some("s")).await, ("ohio-b".to_owned(), true));
        set_occupancy(&pools, &signals, [0.4, 0.6, 0.6, 0.6]);
        assert_eq!(route(Some("s")).await, ("ohio-b".to_owned(), false));
    }

    #[tokio::test]
    async fn affinity_fails_over_from_a_full_pool_or_stays_if_all_are_full() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");

        // Ohio-a is full; Ohio-b has a little room, Oregon more, but the
        // nearest pool with any room wins.
        set_occupancy(&pools, &signals, [1.2, 0.95, 0.5, 0.5]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-b");
        // Every pool is full: moving would only cost a cold prefill.
        set_occupancy(&pools, &signals, [1.1, 1.2, 1.1, 1.3]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-b");
    }

    #[tokio::test]
    async fn affinity_lets_a_quiet_region_take_load_from_busy_ones() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.1, 0.2, 0.2, 0.2]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");

        // Ohio and Oregon at 90%, Tokyo at 20%: only Tokyo has twice the room.
        set_occupancy(&pools, &signals, [0.9, 0.9, 0.9, 0.2]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "tokyo-a");
        // A new session from Ohio goes there too.
        assert_eq!(host_for(&core, &pools, Some("t")).await, "tokyo-a");
        // With Ohio at 60%, a new session stays local: 0.4 spare against
        // Tokyo's 0.8 is not more than twice.
        set_occupancy(&pools, &signals, [0.6, 0.7, 0.9, 0.2]);
        assert_eq!(host_for(&core, &pools, Some("u")).await, "ohio-a");
    }

    #[tokio::test]
    async fn affinity_places_requests_without_a_session_by_the_same_rule() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.5, 0.3, 0.9, 0.0]);
        assert_eq!(host_for(&core, &pools, None).await, "ohio-b");
        set_occupancy(&pools, &signals, [0.9, 0.9, 0.9, 0.0]);
        assert_eq!(host_for(&core, &pools, None).await, "tokyo-a");
        // Nothing was pinned: the next sessionless request follows the load.
        set_occupancy(&pools, &signals, [0.1, 0.9, 0.9, 0.0]);
        assert_eq!(host_for(&core, &pools, None).await, "ohio-a");
    }

    #[tokio::test]
    async fn affinity_counts_a_pool_whose_load_is_unknown_as_full() {
        let (pools, signals) = fleet();
        let (core, _catalog) = selector(affinity(&signals), &pools).await;
        set_occupancy(&pools, &signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");

        // Ohio-a's Relay goes quiet while every other pool is full: the
        // session stays rather than move to a full pool.
        set_occupancy(&pools, &signals, [0.1, 1.0, 1.1, 1.0]);
        signals.forget(&pools[0].id);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "ohio-a");
        // Once a pool has room, the session goes to the nearest such pool.
        set_occupancy(&pools, &signals, [0.1, 1.0, 0.9, 0.2]);
        signals.forget(&pools[0].id);
        assert_eq!(host_for(&core, &pools, Some("s")).await, "oregon-a");
    }
}
