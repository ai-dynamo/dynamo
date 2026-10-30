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
//!
//! With [`CorePolicy::Affinity`] the core also keeps the session table:
//! each request carries its session id, the core holds the session's
//! binding across the pick and binds it to the pool picked, and the
//! [`policies::HeadroomPicker`] sees the binding as the request's affinity
//! target. With a [`ReplicaTransport`] the core's bookings and session
//! bindings replicate to the other Global Frontends, so every replica sees
//! every replica's requests in flight and pins: over the core's own mesh
//! ([`ReplicaSync`]) or over the gRPC stream [`replica_sync`] serves on the
//! API port ([`GrpcReplicaSync`]).

mod catalog;
mod directory;
mod policies;
pub mod replica_sync;

use std::collections::HashMap;
use std::future::Future;
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
    SelectResponse, SelectionHost, SelectionService, SelectionServiceBuilder,
};
use dynamo_kv_router::{WorkerId, WorkerType};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::selector::{Cursors, RequestLifecycle, RequestSelector, Selection, reachable, target};
use crate::signals::PoolSignals;
use crate::{Pool, RoutingRequest};

pub use catalog::{PoolCatalogSource, ROUTING_GROUP, worker_id};
pub use directory::{PoolDirectory, PoolEntry};
/// The longest session id the core's session table accepts; the frontend
/// rejects longer ones before selection, so they are a client error and not
/// an unavailable pool.
pub use dynamo_kv_router::services::selection::affinity::MAX_SESSION_AFFINITY_ID_BYTES as MAX_SESSION_ID_BYTES;
pub use policies::{HeadroomPicker, LeastRequestsPicker, RoundRobinPicker};
pub use replica_sync::{GrpcReplicaSync, ReplicaHub, ReplicaSyncStats};

/// How the core picks among a model's pools.
#[derive(Clone, Debug)]
pub enum CorePolicy {
    /// Every pool in turn ([`RoundRobinPicker`]).
    RoundRobin,
    /// The pool with the fewest requests booked ([`LeastRequestsPicker`]).
    LeastRequests,
    /// Each session on one pool, moved only for much more room
    /// ([`HeadroomPicker`] over the core's session table).
    Affinity(AffinityConfig),
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

/// Replicate bookings and session bindings with other Global Frontends over
/// the core's replica mesh.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplicaSync {
    /// The port this replica publishes on; peers subscribe to it.
    pub port: u16,
    /// Peers to subscribe to at start, as `tcp://host:port`. More can join
    /// through [`CoreSelector::register_peer`].
    pub peers: Vec<String>,
}

/// How bookings and session bindings reach the other Global Frontends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplicaTransport {
    /// The core's own ZMQ mesh, on a port of its own.
    Mesh(ReplicaSync),
    /// The gRPC `ReplicaSync` stream, served on the API port
    /// ([`CoreSelector::replica_sync_routes`]).
    Grpc(GrpcReplicaSync),
}

/// The gRPC transport's state: the hub the stream runs through and the
/// token that stops its tasks.
struct GrpcTransport {
    hub: Arc<ReplicaHub>,
    cancel: CancellationToken,
}

impl Drop for GrpcTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Selects a pool through the router core, then rotates over the pool's
/// frontends.
pub struct CoreSelector {
    service: Arc<SelectionService>,
    directory: Arc<PoolDirectory>,
    frontends: Cursors<PoolId>,
    /// Whether the policy's picker reads the core's bookings, as it declares
    /// ([`policies::reads_bookings`]). A picker that does not gets advisory
    /// selections, booked only where the session table needs it.
    books: bool,
    /// Whether requests carry their session to the core, which only keeps
    /// sessions under [`CorePolicy::Affinity`].
    sessions: bool,
    /// Under affinity, when each session's pin was last booked.
    pins: Option<PinRefresh>,
    grpc: Option<GrpcTransport>,
}

/// When each session's pin was last booked, so only the requests that set,
/// move or keep alive a pin book. The core binds a session only on a
/// booking, and a pin's idle time runs from its last booking's release, so
/// a session books on its first request, whenever the picker moves it, and
/// at least once per half idle window; the requests between pick with an
/// advisory selection that honours the pin and books nothing.
struct PinRefresh {
    refresh: Duration,
    booked: parking_lot::Mutex<HashMap<String, (WorkerId, Instant)>>,
}

impl PinRefresh {
    /// The most sessions remembered; the core keeps no more pins than this.
    const CAPACITY: usize = MAX_SESSION_AFFINITY_ENTRIES;

    fn new(idle: Duration) -> Self {
        Self {
            refresh: idle / 2,
            booked: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// The pool `session` was last booked on, if that was within the
    /// refresh window as of `now`.
    fn fresh(&self, session: &str, now: Instant) -> Option<WorkerId> {
        let booked = self.booked.lock();
        let &(worker, at) = booked.get(session)?;
        (now.saturating_duration_since(at) < self.refresh).then_some(worker)
    }

    fn record(&self, session: &str, worker: WorkerId, now: Instant) {
        let mut booked = self.booked.lock();
        if booked.len() >= Self::CAPACITY && !booked.contains_key(session) {
            let refresh = self.refresh;
            booked.retain(|_, (_, at)| now.saturating_duration_since(*at) < refresh);
        }
        booked.insert(session.to_owned(), (worker, now));
    }
}

impl CoreSelector {
    /// Builds a core with `policy`, replicating with peers over the core's
    /// mesh when `replica_sync` is given. Nothing is selectable until
    /// [`Self::follow`] has applied the first pool snapshot.
    pub async fn new(policy: CorePolicy, replica_sync: Option<ReplicaSync>) -> Result<Self> {
        Self::with_transport(policy, replica_sync.map(ReplicaTransport::Mesh)).await
    }

    /// Builds a core with `policy`, replicating with peers over `transport`
    /// when one is given. Nothing is selectable until [`Self::follow`] has
    /// applied the first pool snapshot, and gRPC peers are followed only
    /// from then on, so their events find the partitions they name. The
    /// gRPC transport's tasks run on the current runtime and stop at
    /// [`Self::stop_replication`] or when the selector drops.
    pub async fn with_transport(
        policy: CorePolicy,
        transport: Option<ReplicaTransport>,
    ) -> Result<Self> {
        let directory = Arc::new(PoolDirectory::new());
        let registry = policies::registry(&policy, directory.clone())
            .context("register the pool selection policies")?;
        let mut builder =
            SelectionServiceBuilder::new(config(&policy)?, WorkerType::Aggregated, registry)
                .indexer_threads(1)
                .host(SelectionHost {
                    replication: HostReplication {
                        request_leases: Some(Arc::new(ReservationLeases)),
                        ..HostReplication::default()
                    },
                    ..SelectionHost::default()
                });
        let affinity = matches!(policy, CorePolicy::Affinity(_));
        if let CorePolicy::Affinity(affinity) = &policy {
            // Soft: the binding follows the pool picked, as the current pin
            // moves with the request.
            builder = builder
                .session_affinity(affinity.idle)
                .session_affinity_mode(SessionAffinityMode::Soft);
        }
        let mut grpc = None;
        match transport {
            Some(ReplicaTransport::Mesh(sync)) => {
                builder = builder.replica_sync(sync.port, sync.peers);
            }
            Some(ReplicaTransport::Grpc(sync)) => {
                let cancel = CancellationToken::new();
                let hub = ReplicaHub::new(cancel.clone());
                for peer in sync.peers {
                    replica_sync::peer_url(peer.as_str())?;
                    hub.add_peer(peer);
                }
                // With peers' bookings mirrored here, the core's age expiry
                // stays on as the backstop for a free that never arrives.
                builder = builder.host(SelectionHost {
                    replication: HostReplication {
                        channels: Some(hub.factory()),
                        ..HostReplication::default()
                    },
                    ..SelectionHost::default()
                });
                if affinity {
                    builder = builder.session_binding_replication(hub.process_id());
                }
                grpc = Some(GrpcTransport { hub, cancel });
            }
            None => {}
        }
        let service = Arc::new(builder.build().await.context("build the router core")?);
        if let Some(GrpcTransport { hub, .. }) = &grpc {
            if affinity {
                let updates = service
                    .session_binding_updates()
                    .context("take the core's session binding updates")?;
                tokio::spawn(hub.clone().publish_bindings(updates));
            }
            tracing::info!(
                process_id = hub.process_id(),
                peers = ?hub.peers().iter().map(Url::as_str).collect::<Vec<_>>(),
                "replicating over the gRPC ReplicaSync stream"
            );
        }
        let books = policies::reads_bookings(&policy, &directory);
        Ok(Self {
            service,
            directory,
            frontends: Cursors::default(),
            books,
            sessions: matches!(policy, CorePolicy::Affinity(_)),
            pins: match &policy {
                CorePolicy::Affinity(affinity) => Some(PinRefresh::new(affinity.idle)),
                _ => None,
            },
            grpc,
        })
    }

    /// The gRPC `ReplicaSync` routes to merge into the API router, when
    /// this replica replicates over gRPC.
    pub fn replica_sync_routes(&self) -> Option<axum::Router> {
        self.grpc
            .as_ref()
            .map(|grpc| replica_sync::server::routes(grpc.hub.clone()))
    }

    /// What the gRPC transport has dropped and seen, when this replica
    /// replicates over gRPC.
    pub fn replica_sync_stats(&self) -> Option<ReplicaSyncStats> {
        self.grpc.as_ref().map(|grpc| grpc.hub.stats())
    }

    /// Keeps the core's workers matching `pools`, the frontend's catalog
    /// subscription, until `cancel` fires or the catalog is dropped. Run the
    /// future as a task; the receiver resolves once the core has taken the
    /// first snapshot, so the caller can wait for the configured pools to be
    /// selectable before serving. It fails if the task ends first.
    ///
    /// With the gRPC transport the future also follows the peers, starting
    /// once that first snapshot is in: an event that arrived before the
    /// core had the partition it names would be dropped for good.
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
        let (source, first_taken) = PoolCatalogSource::with_ready(pools);
        let source = source.with_directory(self.directory.clone());
        let (ready, is_ready) = oneshot::channel();
        let dialer = self
            .grpc
            .as_ref()
            .map(|grpc| (grpc.hub.clone(), grpc.cancel.clone()));
        let service = self.service.clone();
        let stop = cancel.clone();
        let follower = async move {
            let peers = async move {
                if first_taken.await.is_err() {
                    return;
                }
                let _ = ready.send(());
                if let Some((hub, transport)) = dialer {
                    tokio::select! {
                        _ = stop.cancelled() => {}
                        _ = replica_sync::dialer::run_peers(hub, service, transport) => {}
                    }
                }
            };
            tokio::join!(reconciler.run(source, cancel), peers);
        };
        (is_ready, follower)
    }

    /// Ends the gRPC transport now: open `ReplicaSync` streams close and
    /// peers are no longer followed. The Global Frontend calls it when
    /// shutdown begins, so the long-lived streams do not hold the graceful
    /// drain open until its grace runs out. Dropping the selector does the
    /// same.
    pub fn stop_replication(&self) {
        if let Some(grpc) = &self.grpc {
            grpc.cancel.cancel();
        }
    }

    /// Subscribes to another replica's events, `endpoint` as the transport
    /// names peers: `tcp://host:port` over the mesh, `http://host:port` over
    /// gRPC. `Ok(false)` if it was already a peer.
    pub async fn register_peer(&self, endpoint: String) -> Result<bool> {
        if let Some(grpc) = &self.grpc {
            return Ok(grpc.hub.add_peer(replica_sync::peer_url(&endpoint)?));
        }
        self.service
            .register_replica_peer(endpoint.clone())
            .await
            .with_context(|| format!("register replica peer {endpoint}"))
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
        let session = request.session.filter(|_| self.sessions);
        let now = Instant::now();
        let pinned = session
            .zip(self.pins.as_ref())
            .and_then(|(session, pins)| pins.fresh(session, now));
        // A session books when its pin is new or due for a refresh.
        let mut book = self.books || (session.is_some() && pinned.is_none());
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
            tracing::warn!(
                model = request.model,
                worker = selected.worker_id,
                "the router core chose a worker that is not a candidate pool"
            );
            return None;
        };
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
        })
    }
}

impl CoreSelector {
    /// The core's pick among `candidates`, booked or advisory.
    async fn pick(
        &self,
        model: &str,
        candidates: &[&Pool],
        session: Option<&str>,
        book: bool,
    ) -> Option<SelectResponse> {
        let selected = if book {
            self.service
                .select_and_reserve(booking(model, candidates.iter().copied(), session))
                .await
        } else {
            self.service
                .select(advisory(model, candidates.iter().copied(), session))
                .await
        };
        selected
            .map_err(|error| tracing::warn!(model, %error, "the router core selected no pool"))
            .ok()
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
/// age-based expiry off. Nothing is mirrored from peers here, so there is
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
    use axum::serve::ListenerExt;

    use crate::affinity::tests::{OHIO, fleet, set_occupancy};
    use crate::selector::tests::pool;
    use crate::{InFlightTracker, LeastInFlightSelector, PoolSelector, RoundRobinSelector};

    use super::*;

    /// The catalog the core follows: dropping it ends the core's view of
    /// the pools.
    type Catalog = watch::Sender<Arc<Vec<Pool>>>;

    /// A core following `pools`, once every pool with a frontend is one of
    /// its workers.
    async fn selector(policy: CorePolicy, pools: &[Pool]) -> (CoreSelector, Catalog) {
        synced_selector(policy, pools, None).await
    }

    /// [`selector`], replicating over `transport`.
    async fn synced_selector(
        policy: CorePolicy,
        pools: &[Pool],
        transport: Option<ReplicaTransport>,
    ) -> (CoreSelector, Catalog) {
        let (catalog, subscription) = watch::channel(Arc::new(pools.to_vec()));
        let selector = CoreSelector::with_transport(policy, transport)
            .await
            .unwrap();
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
        let selection = selector.route(&request, candidates).await?;
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

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    /// Two cores on the same pools, each subscribed to the other. The
    /// subscriptions need a moment to connect before events flow.
    async fn replicas(
        policy: impl Fn() -> CorePolicy,
        pools: &[Pool],
    ) -> (CoreSelector, CoreSelector, Catalog, Catalog) {
        let (port_a, port_b) = (free_port(), free_port());
        let (a, catalog_a) = synced_selector(
            policy(),
            pools,
            Some(ReplicaTransport::Mesh(ReplicaSync {
                port: port_a,
                peers: Vec::new(),
            })),
        )
        .await;
        let (b, catalog_b) = synced_selector(
            policy(),
            pools,
            Some(ReplicaTransport::Mesh(ReplicaSync {
                port: port_b,
                peers: vec![format!("tcp://127.0.0.1:{port_a}")],
            })),
        )
        .await;
        assert!(
            a.register_peer(format!("tcp://127.0.0.1:{port_b}"))
                .await
                .unwrap()
        );
        assert!(
            !a.register_peer(format!("tcp://127.0.0.1:{port_b}"))
                .await
                .unwrap()
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
        (a, b, catalog_a, catalog_b)
    }

    #[tokio::test]
    async fn round_robin_visits_every_pool_once_per_round_like_the_current_selector() {
        let blue = pool(
            "blue",
            "qwen",
            &["http://blue-0:8000", "http://blue-1:8000"],
        );
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::RoundRobin, &[blue.clone(), green.clone()]).await;
        let current = RoundRobinSelector::new();

        let from_core = sequence(&core, "qwen", &[&blue, &green], 6).await;
        let from_current = (0..6)
            .map(|_| {
                current
                    .select(&RoutingRequest::new("qwen"), &[&blue, &green])
                    .unwrap()
                    .pool
                    .id
                    .clone()
            })
            .collect::<Vec<_>>();

        // The core orders a round by worker id and the current selector by
        // catalog order, so rounds compare as sets.
        assert_eq!(rounds(&from_core, 2), rounds(&from_current, 2));
        assert_eq!(from_core[0], from_core[2]);
        assert_ne!(from_core[0], from_core[1]);
    }

    #[test]
    fn a_policy_books_only_if_its_picker_reads_load() {
        let directory = Arc::new(PoolDirectory::new());
        let (_, signals) = fleet();
        assert!(!policies::reads_bookings(
            &CorePolicy::RoundRobin,
            &directory
        ));
        assert!(policies::reads_bookings(
            &CorePolicy::LeastRequests,
            &directory
        ));
        assert!(!policies::reads_bookings(&affinity(&signals), &directory));
    }

    #[tokio::test]
    async fn round_robin_books_nothing() {
        let blue = pool("blue", "qwen", &["http://blue-0:8000"]);
        let green = pool("green", "qwen", &["http://green-0:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::RoundRobin, &[blue.clone(), green.clone()]).await;
        let held = [
            core.route(&RoutingRequest::new("qwen"), &[&blue, &green])
                .await
                .unwrap(),
            core.route(&RoutingRequest::new("qwen"), &[&blue, &green])
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
    async fn least_requests_picks_the_pool_with_fewest_bookings_like_the_current_selector() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let (core, _catalog) =
            selector(CorePolicy::LeastRequests, &[blue.clone(), green.clone()]).await;
        let in_flight = Arc::new(InFlightTracker::new());
        let current = LeastInFlightSelector::new(in_flight.clone());

        let _busy = [core.reserve_on(&blue).await, core.reserve_on(&blue).await];
        let _one = core.reserve_on(&green).await;
        let _current_busy = [in_flight.start(&blue.id), in_flight.start(&blue.id)];
        let _current_one = in_flight.start(&green.id);
        assert_eq!((core.booked(&blue), core.booked(&green)), (2, 1));

        for _ in 0..10 {
            let from_current = current
                .select(&RoutingRequest::new("qwen"), &[&blue, &green])
                .unwrap()
                .pool
                .id
                .clone();
            assert_eq!(from_current, green.id);
            assert_eq!(
                picks(&core, "qwen", &[&blue, &green]).await,
                Some(from_current)
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
        let current = LeastInFlightSelector::new(Arc::new(InFlightTracker::new()));

        let from_core = sequence(&core, "qwen", &[&empty, &blue, &green], 4).await;
        let from_current = (0..4)
            .map(|_| {
                current
                    .select(&RoutingRequest::new("qwen"), &[&empty, &blue, &green])
                    .unwrap()
                    .pool
                    .id
                    .clone()
            })
            .collect::<Vec<_>>();

        assert_eq!(rounds(&from_core, 2), rounds(&from_current, 2));
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
                let selection = core.route(&request, candidates).await.unwrap();
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

    /// A replica's `ReplicaSync` routes served on a port of their own, as
    /// the Global Frontend serves them on its API port. Dropping it stops
    /// accepting; open streams end when the replica drops.
    struct Served {
        port: u16,
        task: tokio::task::JoinHandle<()>,
    }

    impl Served {
        async fn new(selector: &CoreSelector, port: u16) -> Self {
            let routes = selector.replica_sync_routes().expect("gRPC routes");
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
                .await
                .unwrap();
            let port = listener.local_addr().unwrap().port();
            // As the Global Frontend's `serve` does: the frames are small
            // writes, which Nagle's algorithm would hold for the delayed ACK.
            let listener = listener.tap_io(|tcp| tcp.set_nodelay(true).unwrap());
            let task = tokio::spawn(async move {
                axum::serve(listener, routes).await.unwrap();
            });
            Self { port, task }
        }

        fn url(&self) -> String {
            format!("http://127.0.0.1:{}", self.port)
        }
    }

    impl Drop for Served {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    /// Waits until `selector` has `peers` streams open.
    async fn connected(selector: &CoreSelector, peers: u64) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while selector.replica_sync_stats().unwrap().connected_peers != peers {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the replica did not reach {peers} open streams"));
    }

    /// Two cores on the same pools, each serving the gRPC stream and
    /// subscribed to the other's, once both streams are open.
    async fn grpc_replicas(
        policy: impl Fn() -> CorePolicy,
        pools: &[Pool],
    ) -> (CoreSelector, CoreSelector, Catalog, Catalog, Served, Served) {
        let grpc = || Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (a, catalog_a) = synced_selector(policy(), pools, grpc()).await;
        let (b, catalog_b) = synced_selector(policy(), pools, grpc()).await;
        let served_a = Served::new(&a, 0).await;
        let served_b = Served::new(&b, 0).await;
        assert!(a.register_peer(served_b.url()).await.unwrap());
        assert!(!a.register_peer(served_b.url()).await.unwrap());
        assert!(b.register_peer(served_a.url()).await.unwrap());
        connected(&a, 1).await;
        connected(&b, 1).await;
        (a, b, catalog_a, catalog_b, served_a, served_b)
    }

    /// Two requests booked through A show up at B, which then avoids blue,
    /// and B's own requests reach A.
    async fn assert_replicas_see_each_other_s_bookings(
        a: &CoreSelector,
        b: &CoreSelector,
        blue: &Pool,
        green: &Pool,
    ) {
        let _busy = [a.reserve_on(blue).await, a.reserve_on(blue).await];
        booked_settles(b, blue, 2).await;
        for _ in 0..4 {
            assert_eq!(
                picks(b, "qwen", &[blue, green]).await,
                Some(green.id.clone())
            );
        }
        let _on_green = [
            b.reserve_on(green).await,
            b.reserve_on(green).await,
            b.reserve_on(green).await,
        ];
        booked_settles(a, green, 3).await;
        assert_eq!(
            picks(a, "qwen", &[blue, green]).await,
            Some(blue.id.clone())
        );
    }

    /// A's pin of a session is honoured by B, and B's move of it by A.
    async fn assert_replicas_agree_on_a_session_s_pool(
        a: &CoreSelector,
        b: &CoreSelector,
        pools: &[Pool],
        signals: &Arc<PoolSignals>,
    ) {
        // A pins the session to ohio-a while it is the emptiest local pool.
        set_occupancy(pools, signals, [0.1, 0.5, 0.5, 0.5]);
        assert_eq!(host_for(a, pools, Some("s")).await, "ohio-a");
        // Now ohio-b is emptier, but not by twice the room. Without the
        // replicated pin B would start the session on ohio-b.
        set_occupancy(pools, signals, [0.5, 0.3, 0.5, 0.5]);
        assert_eq!(host_for(b, pools, Some("fresh")).await, "ohio-b");
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if host_for(b, pools, Some("s")).await == "ohio-a" {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("B learned A's pin");
        // The pin moving on B is seen by A: ohio-b fills, Tokyo empties.
        set_occupancy(pools, signals, [0.9, 0.9, 0.9, 0.1]);
        assert_eq!(host_for(b, pools, Some("s")).await, "tokyo-a");
        set_occupancy(pools, signals, [0.5, 0.5, 0.5, 0.5]);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if host_for(a, pools, Some("s")).await == "tokyo-a" {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("A learned B's move");
    }

    #[tokio::test]
    async fn replicas_see_each_other_s_bookings() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let pools = [blue.clone(), green.clone()];
        let (a, b, _catalog_a, _catalog_b) = replicas(|| CorePolicy::LeastRequests, &pools).await;
        assert_replicas_see_each_other_s_bookings(&a, &b, &blue, &green).await;
    }

    #[tokio::test]
    async fn replicas_agree_on_a_session_s_pool() {
        let (pools, signals) = fleet();
        let (a, b, _catalog_a, _catalog_b) = replicas(|| affinity(&signals), &pools).await;
        assert_replicas_agree_on_a_session_s_pool(&a, &b, &pools, &signals).await;
    }

    #[tokio::test]
    async fn grpc_replicas_see_each_other_s_bookings() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let pools = [blue.clone(), green.clone()];
        let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
            grpc_replicas(|| CorePolicy::LeastRequests, &pools).await;
        assert_replicas_see_each_other_s_bookings(&a, &b, &blue, &green).await;
        let stats = a.replica_sync_stats().unwrap();
        assert_eq!(
            (
                stats.events_dropped,
                stats.events_unroutable,
                stats.subscribers_lagged
            ),
            (0, 0, 0)
        );
    }

    #[tokio::test]
    async fn grpc_replicas_agree_on_a_session_s_pool() {
        let (pools, signals) = fleet();
        let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
            grpc_replicas(|| affinity(&signals), &pools).await;
        assert_replicas_agree_on_a_session_s_pool(&a, &b, &pools, &signals).await;
        assert_eq!(a.replica_sync_stats().unwrap().bindings_rejected, 0);
    }

    /// A client of `served` with `marker`, and the result of subscribing.
    async fn subscribe(
        served: &Served,
        marker: u32,
    ) -> Result<tonic::Streaming<replica_sync::proto::ReplicaFrame>, tonic::Status> {
        let channel = tonic::transport::Endpoint::from_shared(served.url())
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client = replica_sync::proto::replica_sync_client::ReplicaSyncClient::new(channel);
        client
            .subscribe_replica_events(replica_sync::proto::SubscribeRequest {
                subscriber_id: 1,
                routing_group: ROUTING_GROUP.to_owned(),
                contract_marker: marker,
            })
            .await
            .map(tonic::Response::into_inner)
    }

    #[tokio::test]
    async fn grpc_rejects_a_contract_mismatch() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let grpc = Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (a, _catalog) = synced_selector(CorePolicy::LeastRequests, &[blue], grpc).await;
        let served = Served::new(&a, 0).await;

        let status = subscribe(&served, 0).await.expect_err("rejected");
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
        assert!(
            subscribe(&served, replica_sync::CONTRACT_MARKER)
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn grpc_sends_a_snapshot_first_then_events() {
        use replica_sync::proto::replica_frame::Frame;

        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let grpc = Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (a, _catalog) =
            synced_selector(CorePolicy::LeastRequests, std::slice::from_ref(&blue), grpc).await;
        let served = Served::new(&a, 0).await;

        let mut frames = subscribe(&served, replica_sync::CONTRACT_MARKER)
            .await
            .unwrap();
        let first = frames.message().await.unwrap().unwrap();
        assert!(matches!(first.frame, Some(Frame::Snapshot(_))), "{first:?}");

        let held = a.reserve_on(&blue).await;
        let second = tokio::time::timeout(Duration::from_secs(5), frames.message())
            .await
            .expect("an event frame")
            .unwrap()
            .unwrap();
        let Some(Frame::Events(batch)) = second.frame else {
            panic!("expected events, got {second:?}");
        };
        assert_eq!(batch.events.len(), 1);
        assert_eq!(
            batch.events[0].request_id,
            held.selection_id.clone().unwrap()
        );
        assert_eq!(batch.events[0].worker_id, worker_id(&blue.id));
        assert_eq!(batch.events[0].routing_group, ROUTING_GROUP);
    }

    #[tokio::test]
    async fn grpc_peer_reconnects_after_the_server_restarts() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let pools = [blue.clone(), green.clone()];
        let (a, b, _catalog_a, catalog_b, _served_a, served_b) =
            grpc_replicas(|| CorePolicy::LeastRequests, &pools).await;
        let port_b = served_b.port;

        // B goes away: its listener closes and its open streams end.
        drop(served_b);
        drop(b);
        drop(catalog_b);
        connected(&a, 0).await;

        // A new B on the same port: A's dialer finds it within its retry.
        let grpc = Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (b, _catalog_b) = synced_selector(CorePolicy::LeastRequests, &pools, grpc).await;
        let served_b = Served::new(&b, port_b).await;
        assert!(b.register_peer(_served_a.url()).await.unwrap());
        connected(&a, 1).await;
        connected(&b, 1).await;
        assert_replicas_see_each_other_s_bookings(&a, &b, &blue, &green).await;
        drop(served_b);
    }

    #[tokio::test]
    async fn grpc_peers_are_followed_only_once_the_first_snapshot_is_in() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let pools = [blue.clone()];
        let grpc = || Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (b, _catalog_b) = synced_selector(CorePolicy::LeastRequests, &pools, grpc()).await;
        let served_b = Served::new(&b, 0).await;

        // A knows B before it follows any catalog: nothing is dialed yet.
        let a = CoreSelector::with_transport(
            CorePolicy::LeastRequests,
            Some(ReplicaTransport::Grpc(GrpcReplicaSync {
                peers: vec![replica_sync::peer_url(&served_b.url()).unwrap()],
            })),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(a.replica_sync_stats().unwrap().connected_peers, 0);

        // Following the catalog takes the first snapshot, then dials.
        let (_catalog_a, subscription) = watch::channel(Arc::new(pools.to_vec()));
        let (ready, follower) = a.follow(subscription, CancellationToken::new());
        tokio::spawn(follower);
        ready.await.unwrap();
        connected(&a, 1).await;
        drop(served_b);
    }

    #[tokio::test]
    async fn grpc_a_peer_removed_and_added_back_at_once_is_followed_again() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let pools = [blue.clone(), green.clone()];
        let (a, b, _catalog_a, _catalog_b, _served_a, served_b) =
            grpc_replicas(|| CorePolicy::LeastRequests, &pools).await;

        // The old subscription is still winding down when the peer is
        // back; the dialer must start a new one once it has ended.
        let hub = &a.grpc.as_ref().unwrap().hub;
        let peer_b = replica_sync::peer_url(&served_b.url()).unwrap();
        assert!(hub.remove_peer(&peer_b));
        assert!(hub.add_peer(peer_b));
        tokio::time::sleep(Duration::from_millis(200)).await;

        let _held = [b.reserve_on(&blue).await, b.reserve_on(&blue).await];
        booked_settles(&a, &blue, 2).await;
    }

    #[tokio::test]
    async fn grpc_stop_replication_ends_the_open_streams() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let green = pool("green", "qwen", &["http://green:8000"]);
        let pools = [blue.clone(), green.clone()];
        let (a, b, _catalog_a, _catalog_b, _served_a, _served_b) =
            grpc_replicas(|| CorePolicy::LeastRequests, &pools).await;

        // B stops: A's stream from B closes, and B follows nobody.
        b.stop_replication();
        connected(&a, 0).await;
        connected(&b, 0).await;
    }

    #[tokio::test]
    async fn grpc_refuses_tls_peers_and_other_schemes() {
        let blue = pool("blue", "qwen", &["http://blue:8000"]);
        let grpc = Some(ReplicaTransport::Grpc(GrpcReplicaSync::default()));
        let (a, _catalog) = synced_selector(CorePolicy::LeastRequests, &[blue], grpc).await;
        for peer in ["https://peer:8000", "tcp://peer:5600", "peer:8000"] {
            assert!(a.register_peer(peer.to_owned()).await.is_err(), "{peer}");
        }
        assert!(
            a.register_peer("http://peer:8000".to_owned())
                .await
                .unwrap()
        );
        assert_eq!(a.replica_sync_stats().unwrap().connected_peers, 0);
    }
}
