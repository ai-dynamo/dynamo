// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use super::{
    LlmResponse, MAX_SESSION_AFFINITY_ENTRIES, MAX_SESSION_AFFINITY_ID_BYTES,
    MAX_SESSION_AFFINITY_TTL_SECS,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        timing::RequestPhase,
    },
    session_placement::{
        PlacementAcquire, PlacementInitialization, PlacementLease, SessionPlacement,
        SessionPlacementConfig, SessionPlacementError,
    },
};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinityTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

#[derive(Clone)]
pub struct AffinityCoordinator {
    placement: SessionPlacement<AffinityTarget>,
    max_session_id_bytes: usize,
}

impl AffinityCoordinator {
    pub fn new(ttl: Duration) -> Result<Self, Error> {
        Self::new_with_limits(
            ttl,
            MAX_SESSION_AFFINITY_ENTRIES,
            MAX_SESSION_AFFINITY_ID_BYTES,
        )
    }

    fn new_with_limits(
        ttl: Duration,
        max_entries: usize,
        max_session_id_bytes: usize,
    ) -> Result<Self, Error> {
        if !(Duration::from_secs(1)..=Duration::from_secs(MAX_SESSION_AFFINITY_TTL_SECS))
            .contains(&ttl)
        {
            return Err(invalid_argument(format!(
                "session affinity TTL must be between 1 and {MAX_SESSION_AFFINITY_TTL_SECS} seconds"
            )));
        }
        let placement = SessionPlacement::new(SessionPlacementConfig {
            idle_ttl: ttl,
            // Existing worker affinity keeps initialization until its owner finishes or drops.
            initialization_timeout: None,
            max_entries,
            max_key_bytes: max_session_id_bytes,
        })
        .map_err(map_placement_error)?;
        Ok(Self {
            placement,
            max_session_id_bytes,
        })
    }

    #[cfg(test)]
    pub(crate) async fn acquire(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, None).await
    }

    pub(crate) async fn acquire_with_context(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: &dyn AsyncEngineContext,
    ) -> Result<AffinityAcquire, Error> {
        self.acquire_inner(session_id, requested_target, Some(request_context))
            .await
    }

    async fn acquire_inner(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
        request_context: Option<&dyn AsyncEngineContext>,
    ) -> Result<AffinityAcquire, Error> {
        self.validate_session_id(session_id)?;
        let acquired = if let Some(context) = request_context {
            let cancellation = async {
                tokio::select! {
                    biased;
                    _ = context.stopped() => {},
                    _ = context.killed() => {},
                }
            };
            let acquired = self
                .placement
                .acquire_with_cancellation(session_id.as_str(), cancellation)
                .await;
            match acquired {
                Ok(acquired) => acquired,
                Err(SessionPlacementError::AcquireCancelled) => {
                    return Err(cancelled(context.id()));
                }
                Err(error) => return Err(map_placement_error(error)),
            }
        } else {
            self.placement
                .acquire(session_id.as_str())
                .await
                .map_err(map_placement_error)?
        };

        match acquired {
            PlacementAcquire::Initialize(initialization) => {
                Ok(AffinityAcquire::Initialize(AffinityInitialization {
                    initialization,
                    session_id: session_id.as_str().to_string(),
                    requested_target,
                }))
            }
            PlacementAcquire::Bound { target, mut lease } => {
                if let Err(error) =
                    validate_bound_target(session_id.as_str(), target, requested_target)
                {
                    lease.abandon();
                    return Err(error);
                }
                Ok(AffinityAcquire::Bound {
                    target,
                    lease: AffinityLease { lease },
                })
            }
        }
    }

    pub fn query_target(
        &self,
        session_id: &SessionAffinityId,
        requested_target: Option<AffinityTarget>,
    ) -> Result<Option<AffinityTarget>, Error> {
        self.validate_session_id(session_id)?;
        let target = self
            .placement
            .query(session_id.as_str())
            .map_err(map_placement_error)?;
        if let Some(target) = target {
            validate_bound_target(session_id.as_str(), target, requested_target)?;
        }
        Ok(target)
    }

    fn validate_session_id(&self, session_id: &SessionAffinityId) -> Result<(), Error> {
        if session_id.as_str().len() > self.max_session_id_bytes {
            return Err(invalid_argument(format!(
                "session affinity ID must not exceed {} bytes",
                self.max_session_id_bytes
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        self.placement.entry_count()
    }

    #[cfg(test)]
    pub(super) fn cancellation_token(&self) -> tokio_util::sync::CancellationToken {
        self.placement.cancellation_token()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_reaper(&self) {
        self.placement.wait_for_reaper().await;
    }

    #[cfg(test)]
    pub(super) async fn wait_for_initializing_waiter(&self) {
        self.placement.wait_for_initializing_waiter().await;
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self, session_id: &SessionAffinityId) {
        self.placement.expire_for_test(session_id.as_str());
    }

    #[cfg(test)]
    pub(super) fn with_test_limits(max_entries: usize, max_session_id_bytes: usize) -> Self {
        Self::new_with_limits(Duration::from_secs(10), max_entries, max_session_id_bytes).unwrap()
    }
}

pub(crate) enum AffinityAcquire {
    Initialize(AffinityInitialization),
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

impl AffinityAcquire {
    pub(crate) fn target(&self) -> Option<AffinityTarget> {
        match self {
            Self::Initialize(_) => None,
            Self::Bound { target, .. } => Some(*target),
        }
    }

    pub(crate) fn into_stream(
        self,
        selected_target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        match self {
            Self::Initialize(initialization) => {
                Ok(initialization.commit(selected_target)?.into_stream(stream))
            }
            Self::Bound { target, mut lease } => {
                if let Err(error) = validate_bound_target("session", target, Some(selected_target))
                {
                    lease.invalidate();
                    return Err(error);
                }
                Ok(lease.into_stream(stream))
            }
        }
    }

    pub(crate) fn invalidate(self) {
        if let Self::Bound { mut lease, .. } = self {
            lease.invalidate();
        }
    }
}

pub(crate) struct AffinityInitialization {
    initialization: PlacementInitialization<AffinityTarget>,
    session_id: String,
    requested_target: Option<AffinityTarget>,
}

impl AffinityInitialization {
    pub(crate) fn commit(self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        validate_bound_target(&self.session_id, target, self.requested_target)?;
        let lease = self
            .initialization
            .commit(target)
            .map_err(map_placement_error)?;
        Ok(AffinityLease { lease })
    }
}

pub(crate) struct AffinityLease {
    lease: PlacementLease<AffinityTarget>,
}

impl AffinityLease {
    pub(crate) fn into_stream(self, stream: ManyOut<LlmResponse>) -> ManyOut<LlmResponse> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(AffinityTrackedStream {
                stream,
                lease: Some(self),
            }),
            context,
        )
    }

    fn invalidate(&mut self) {
        self.lease.invalidate();
    }
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
        RequestPhase::Decode => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
        RequestPhase::Aggregated => (
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

fn validate_bound_target(
    session_id: &str,
    bound: AffinityTarget,
    requested: Option<AffinityTarget>,
) -> Result<(), Error> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if bound.worker_id != requested.worker_id {
        return Err(invalid_argument(format!(
            "session {session_id} is bound to worker {}, not {}",
            bound.worker_id, requested.worker_id
        )));
    }
    match (bound.dp_rank, requested.dp_rank) {
        (Some(bound), Some(requested)) if bound != requested => Err(invalid_argument(format!(
            "session {session_id} is bound to DP rank {bound}, not {requested}"
        ))),
        (None, Some(requested)) => Err(invalid_argument(format!(
            "session {session_id} has worker-only affinity and cannot add DP rank {requested}"
        ))),
        _ => Ok(()),
    }
}

fn map_placement_error(error: SessionPlacementError) -> Error {
    match error {
        SessionPlacementError::KeyTooLong { max_bytes, .. } => invalid_argument(format!(
            "session affinity ID must not exceed {max_bytes} bytes"
        )),
        SessionPlacementError::Capacity { .. } => {
            resource_exhausted("session affinity entry limit reached")
        }
        SessionPlacementError::CoordinatorDropped => {
            anyhow::anyhow!("session affinity coordinator dropped")
        }
        SessionPlacementError::RuntimeUnavailable => {
            invalid_argument("session affinity requires a Tokio runtime")
        }
        SessionPlacementError::InitializationCancelled => {
            invalid_argument("session affinity initialization was cancelled")
        }
        SessionPlacementError::InitializationChanged => {
            invalid_argument("session affinity initialization changed")
        }
        SessionPlacementError::AcquireCancelled => {
            invalid_argument("session affinity acquisition was cancelled")
        }
        SessionPlacementError::InvalidConfig(message) => invalid_argument(message),
    }
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message.into())
        .build()
        .into()
}

fn resource_exhausted(message: impl Into<String>) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::ResourceExhausted)
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
