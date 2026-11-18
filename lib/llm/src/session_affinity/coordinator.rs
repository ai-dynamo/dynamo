// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's view of session affinity: the shared `dynamo_kv_router`
//! [`SessionAffinity`] table plus what only the frontend knows about, namely
//! request contexts (to abandon a wait when the client cancels), request
//! guards (a committed lease lives exactly as long as the request's response
//! stream), and the runtime event plane that replicates bindings between
//! frontends.

use std::{collections::HashMap, sync::Arc, time::Duration};

#[cfg(test)]
use super::replica_sync::SessionAffinityUpdate;
use super::{SessionAffinityMode, replica_sync::ReplicaSyncRuntime};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};
use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
    pipeline::Error,
};
use dynamo_session_affinity::host::{
    AcquireStep, AffinityError, AffinityLease, Hold, SessionAffinity,
};

#[cfg(test)]
pub(super) use dynamo_session_affinity::host::ReplicaApplyOutcome;

/// The pipeline's routing target; the table's `AffinityTarget` has the same
/// shape and the two convert at this boundary.
pub type AffinityTarget = dynamo_runtime::pipeline::RouteTarget;

type TableTarget = dynamo_session_affinity::host::AffinityTarget;

pub(crate) fn to_table(target: AffinityTarget) -> TableTarget {
    TableTarget::new(target.worker_id, target.dp_rank)
}

pub(crate) fn from_table(target: TableTarget) -> AffinityTarget {
    AffinityTarget::new(target.worker_id, target.dp_rank)
}

/// The worker with the lowest `(live sessions, active requests, active blocks)`;
/// ties are random so equal workers share a burst.
fn least_loaded(loads: &WorkerLoads, live_sessions: impl Fn(u64) -> usize) -> Option<u64> {
    let mut best_key = None;
    let mut best = Vec::new();
    for (&worker_id, &(active_requests, active_blocks)) in loads {
        let key = (live_sessions(worker_id), active_requests, active_blocks);
        match best_key {
            Some(current) if key > current => {}
            Some(current) if key == current => best.push(worker_id),
            _ => {
                best_key = Some(key);
                best.clear();
                best.push(worker_id);
            }
        }
    }
    match best.len() {
        0 => None,
        1 => Some(best[0]),
        n => Some(best[rand::random_range(0..n)]),
    }
}

/// How a session with no live binding picks its first worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionPlacement {
    /// The router's normal selection (KV overlap and load) decides.
    #[default]
    Kv,
    /// The eligible worker with the fewest live sessions, then the fewest active
    /// requests, then the fewest active blocks; remaining ties are random. A session
    /// between turns has no request in flight, so load alone cannot see it.
    Balanced,
}

impl std::str::FromStr for SessionPlacement {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "kv" => Ok(Self::Kv),
            "balanced" => Ok(Self::Balanced),
            _ => Err(format!(
                "invalid session affinity placement {value:?}; expected 'kv' or 'balanced'"
            )),
        }
    }
}

/// Where new sessions start. The default changes nothing: normal selection
/// places new sessions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlacementSettings {
    pub placement: SessionPlacement,
    /// A binding counts as a live session while a request holds it or for this
    /// long after its last request; `None` counts every unexpired binding.
    pub active_within: Option<Duration>,
}

/// `(active requests, active blocks)` per eligible worker.
pub(crate) type WorkerLoads = HashMap<u64, (usize, usize)>;

struct Inner {
    table: SessionAffinity,
    replica: tokio::sync::OnceCell<ReplicaSyncRuntime>,
    placement: PlacementSettings,
    /// Serializes count-then-plan so concurrent new sessions see each other.
    placement_lock: parking_lot::Mutex<()>,
}

#[derive(Clone)]
pub struct AffinityCoordinator {
    inner: Arc<Inner>,
}

impl AffinityCoordinator {
    pub fn new(ttl: Duration, mode: SessionAffinityMode) -> Result<Self, Error> {
        Self::wrap_with_env_placement(
            SessionAffinity::with_config(super::affinity_config(ttl, mode)?)
                .map_err(affinity_error)?,
        )
    }

    /// Wrap `table` with the placement settings from the environment. Every table
    /// a router serves goes through here, including the KV router's partition table.
    pub(crate) fn wrap_with_env_placement(table: SessionAffinity) -> Result<Self, Error> {
        Ok(Self::with_placement(
            table,
            super::placement_settings_from_env()?,
        ))
    }

    pub(crate) fn with_placement(table: SessionAffinity, placement: PlacementSettings) -> Self {
        if placement != PlacementSettings::default() {
            tracing::info!(?placement, "session affinity placement");
        }
        Self {
            inner: Arc::new(Inner {
                table,
                replica: tokio::sync::OnceCell::new(),
                placement,
                placement_lock: parking_lot::Mutex::new(()),
            }),
        }
    }

    pub(crate) fn placement(&self) -> PlacementSettings {
        self.inner.placement
    }

    /// Plan an initializing session on the eligible worker in `loads` with the fewest
    /// live sessions, then active requests, then active blocks; ties are random.
    ///
    /// Counting and planning share one lock, so each placement sees every session
    /// placed before it, committed or not. Returns a worker-only target (selection
    /// still picks the rank), or `None` when the hold is bound, there is no
    /// candidate, or the initialization was cancelled.
    pub(crate) fn place_by_live_sessions(
        &self,
        hold: &Hold,
        loads: &WorkerLoads,
    ) -> Option<AffinityTarget> {
        if !matches!(hold, Hold::Initialize(_)) {
            return None;
        }
        let _placing = self.inner.placement_lock.lock();
        let sessions = self.live_sessions();
        let worker_id = least_loaded(loads, |worker_id| {
            sessions.get(&worker_id).copied().unwrap_or(0)
        })?;
        let target = AffinityTarget::new(worker_id, None);
        if !hold.plan(to_table(target)) {
            return None;
        }
        tracing::debug!(
            worker_id,
            live_sessions = sessions.get(&worker_id).copied().unwrap_or(0),
            candidates = loads.len(),
            "session affinity: placed a session by live sessions"
        );
        Some(target)
    }

    /// Live sessions per worker: bindings with a request in flight or released
    /// within the activity window, plus planned initializations.
    pub(crate) fn live_sessions(&self) -> HashMap<u64, usize> {
        self.inner
            .table
            .sessions_per_worker(self.inner.placement.active_within)
    }

    pub(crate) async fn enable_replica_sync(
        &self,
        client: dynamo_runtime::component::Client,
    ) -> Result<(), Error> {
        self.inner
            .replica
            .get_or_try_init(|| async {
                let (replica, router_id) =
                    ReplicaSyncRuntime::start(client, self.inner.table.downgrade()).await?;
                if !self
                    .inner
                    .table
                    .enable_replication(router_id, replica.sink())
                {
                    return Err(anyhow::anyhow!(
                        "session affinity table already has a replica sink installed"
                    ));
                }
                Ok(replica)
            })
            .await?;
        Ok(())
    }

    /// Take a slot without cancelling on the request context.
    ///
    /// The caller owns the deadline. `RoutingHost` uses this for a decode leg
    /// that has staged KV to release: that request must reach its worker even
    /// after the client disconnects, so it waits through a stop under the
    /// routing host's shared cleanup budget instead of being cancelled here.
    /// Every other caller wants [`Self::acquire_with_context`].
    pub(crate) async fn acquire(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Hold, Error> {
        self.inner
            .table
            .acquire(session_id.as_str(), requested_target.map(to_table))
            .await
            .map_err(affinity_error)
    }

    pub(crate) async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        context: &dyn AsyncEngineContext,
    ) -> Result<Hold, Error> {
        let requested = requested_target.map(to_table);
        loop {
            match self
                .inner
                .table
                .try_acquire(session_id.as_str(), requested)
                .map_err(affinity_error)?
            {
                AcquireStep::Held(hold) => return Ok(hold),
                AcquireStep::Wait(notified) => {
                    tokio::select! {
                        biased;
                        _ = context.stopped() => return Err(cancelled(context.id())),
                        _ = context.killed() => return Err(cancelled(context.id())),
                        _ = notified => {}
                    }
                }
            }
        }
    }

    pub fn query_target(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.inner
            .table
            .query_target(session_id.as_str(), requested_target.map(to_table))
            .map(|target| target.map(from_table))
            .map_err(affinity_error)
    }

    pub(crate) fn mode(&self) -> SessionAffinityMode {
        self.inner.table.mode()
    }

    /// `SessionAffinity::commit`: bind the held session to the worker the
    /// request was sent to. The caller owns the lease for as long as the
    /// request lives; dropping it releases the binding.
    pub(crate) fn commit(
        &self,
        hold: Hold,
        dispatched_target: AffinityTarget,
    ) -> Result<AffinityLease, Error> {
        self.inner
            .table
            .commit(hold, to_table(dispatched_target))
            .map_err(affinity_error)
    }

    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        self.inner.table.entry_count()
    }

    #[cfg(test)]
    pub(super) fn cancellation_token(&self) -> tokio_util::sync::CancellationToken {
        self.inner.table.cancellation_token()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_reaper(&self) {
        self.inner.table.wait_for_reaper().await;
    }

    #[cfg(test)]
    pub(crate) async fn wait_for_initializing_waiter(&self) {
        self.inner.table.wait_for_initializing_waiter().await;
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self, session_id: &SessionAffinityId) {
        self.inner.table.expire_for_test(session_id.as_str());
    }

    #[cfg(test)]
    pub(super) fn with_test_limits(max_entries: usize, max_session_id_bytes: usize) -> Self {
        use dynamo_session_affinity::host::SessionAffinityConfig;

        Self::with_placement(
            SessionAffinity::with_config(SessionAffinityConfig {
                max_entries,
                max_session_id_bytes,
                ..SessionAffinityConfig::new(Duration::from_secs(10))
            })
            .unwrap(),
            PlacementSettings::default(),
        )
    }

    #[cfg(test)]
    pub(super) fn enable_test_replica(
        &self,
        router_id: u64,
        capacity: usize,
    ) -> tokio::sync::mpsc::Receiver<SessionAffinityUpdate> {
        let (replica, rx) = ReplicaSyncRuntime::for_test(capacity);
        assert!(
            self.inner
                .table
                .enable_replication(router_id, replica.sink()),
            "session affinity test replica already enabled"
        );
        self.inner
            .replica
            .set(replica)
            .unwrap_or_else(|_| panic!("session affinity test replica already enabled"));
        rx
    }

    #[cfg(test)]
    pub(super) fn next_version_for_test(&self) -> dynamo_session_affinity::host::AffinityVersion {
        self.inner.table.next_version()
    }

    #[cfg(test)]
    pub(super) fn table_for_test(&self) -> dynamo_session_affinity::host::WeakSessionAffinity {
        self.inner.table.downgrade()
    }

    #[cfg(test)]
    pub(super) fn apply_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
    ) -> ReplicaApplyOutcome {
        self.apply_versioned_replica_update_for_test(session_id, target, 0, 0)
    }

    #[cfg(test)]
    pub(super) fn apply_versioned_replica_update_for_test(
        &self,
        session_id: impl Into<String>,
        target: AffinityTarget,
        sequence: u64,
        writer_id: u64,
    ) -> ReplicaApplyOutcome {
        use dynamo_session_affinity::host::AffinityVersion;
        self.inner.table.apply_replica_update(
            session_id.into(),
            to_table(target),
            AffinityVersion {
                sequence,
                writer_id,
            },
        )
    }
}

pub fn affinity_id(
    request: &dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
) -> Result<Option<Arc<SessionAffinityId>>, Error> {
    request
        .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
        .map_err(|message| invalid_argument(format!("invalid session affinity context: {message}")))
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (
            routing.prefill_worker_id.or(routing.backend_instance_id),
            routing.prefill_dp_rank.or(routing.dp_rank),
        ),
        RequestPhase::Decode | RequestPhase::Aggregated => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for session affinity",
        ));
    }
    Ok(worker_id.map(|worker_id| AffinityTarget { worker_id, dp_rank }))
}

fn affinity_error(error: AffinityError) -> Error {
    match error {
        AffinityError::InvalidArgument(message) => invalid_argument(message),
        AffinityError::ResourceExhausted(message) => DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message(message)
            .build()
            .into(),
        AffinityError::Dropped => anyhow::anyhow!("session affinity coordinator dropped"),
    }
}

/// Session and worker identifiers are private diagnostics and must not be copied into client responses.
pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn cancelled(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "request {context_id} was cancelled while waiting for session affinity"
        ))
        .build()
        .into()
}
