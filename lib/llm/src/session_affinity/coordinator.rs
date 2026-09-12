// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's view of session affinity: the shared `dynamo_kv_router`
//! [`SessionAffinity`] table plus what only the frontend knows about, namely
//! request contexts (to abandon a wait when the client cancels), pipeline
//! response streams (a lease lives until the stream ends), and the runtime
//! event plane that replicates bindings between frontends.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityError, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;

#[cfg(test)]
use super::replica_sync::SessionAffinityUpdate;
use super::{LlmResponse, SessionAffinityMode, replica_sync::ReplicaSyncRuntime};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
};

#[cfg(test)]
pub(super) use dynamo_kv_router::services::selection::affinity::ReplicaApplyOutcome;

/// The pipeline's routing target; the table's `AffinityTarget` has the same
/// shape and the two convert at this boundary.
pub type AffinityTarget = dynamo_runtime::pipeline::RouteTarget;

type TableTarget = dynamo_kv_router::services::selection::affinity::AffinityTarget;

/// The table-side form of a pipeline routing target.
pub(crate) fn to_table(target: AffinityTarget) -> TableTarget {
    TableTarget::new(target.worker_id, target.dp_rank)
}

pub(crate) fn from_table(target: TableTarget) -> AffinityTarget {
    AffinityTarget::new(target.worker_id, target.dp_rank)
}

struct Inner {
    table: SessionAffinity,
    replica: tokio::sync::OnceCell<ReplicaSyncRuntime>,
}

#[derive(Clone)]
pub struct AffinityCoordinator {
    inner: Arc<Inner>,
}

impl AffinityCoordinator {
    pub fn new(ttl: Duration, mode: SessionAffinityMode) -> Result<Self, Error> {
        Ok(Self::wrap(
            SessionAffinity::with_config(SessionAffinityConfig::new(ttl).with_mode(mode))
                .map_err(affinity_error)?,
        ))
    }

    pub(crate) fn wrap(table: SessionAffinity) -> Self {
        Self {
            inner: Arc::new(Inner {
                table,
                replica: tokio::sync::OnceCell::new(),
            }),
        }
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

    /// The mode the table binds sessions with.
    pub(crate) fn mode(&self) -> SessionAffinityMode {
        self.inner.table.mode()
    }

    /// `SessionAffinity::commit`, holding the lease until `stream` ends.
    pub(crate) fn commit_to_stream(
        &self,
        hold: Hold,
        dispatched_target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let lease = self
            .inner
            .table
            .commit(hold, to_table(dispatched_target))
            .map_err(affinity_error)?;
        Ok(tracked_stream(lease, stream))
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
        Self::wrap(
            SessionAffinity::with_config(SessionAffinityConfig {
                max_entries,
                max_session_id_bytes,
                ..SessionAffinityConfig::new(Duration::from_secs(10))
            })
            .unwrap(),
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
    pub(super) fn next_version_for_test(
        &self,
    ) -> dynamo_kv_router::services::selection::affinity::AffinityVersion {
        self.inner.table.next_version()
    }

    #[cfg(test)]
    pub(super) fn table_for_test(
        &self,
    ) -> dynamo_kv_router::services::selection::affinity::WeakSessionAffinity {
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
        use dynamo_kv_router::services::selection::affinity::AffinityVersion;
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

pub(super) fn tracked_stream(
    lease: AffinityLease,
    stream: ManyOut<LlmResponse>,
) -> ManyOut<LlmResponse> {
    let context = stream.context();
    ResponseStream::new(
        Box::pin(AffinityTrackedStream {
            stream,
            lease: Some(lease),
        }),
        context,
    )
}

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                drop(self.lease.take());
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            poll => poll,
        }
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
