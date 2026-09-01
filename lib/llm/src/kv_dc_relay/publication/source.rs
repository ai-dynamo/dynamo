// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decouples publication drivers from Relay internals through state watches and generation-safe
//! streams.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio_util::sync::CancellationToken;

use super::super::identity::{DcPoolCatalog, DcRelayIdentity};
use super::super::load::PoolLoadSnapshot;
use super::super::pool_registry::PoolRegistry;
use super::super::topology::{TopologyPublisher, TopologySnapshot};
use super::hub::PublicationHubError;
use super::stream::{self, PoolPublicationStream, TerminalPublicationFailure};

pub(in crate::kv_dc_relay) const DEFAULT_SNAPSHOT_PROGRESS_TIMEOUT: Duration =
    Duration::from_secs(60);
pub(in crate::kv_dc_relay) const DEFAULT_SNAPSHOT_ENCODING_CONCURRENCY: usize = 2;
pub(in crate::kv_dc_relay) const DEFAULT_ACTIVE_POOL_STREAMS: usize = 64;

mod private {
    pub trait Sealed {}
}

/// Stable error categories that publication drivers can map to their transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PublicationErrorKind {
    /// The requested pool is not active.
    NotFound,
    /// The pool is active, but its generation differs from the request.
    ProducerMismatch,
    /// Publication state is temporarily unavailable and requires a fresh stream.
    Unavailable,
    /// A bounded Relay publication resource was exhausted.
    ResourceExhausted,
    /// The source produced an invalid identity, format, or sequence transition.
    InvalidPublication,
    /// Internal snapshot encoding failed.
    Internal,
}

/// Failure to open or advance a pool publication stream.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct PublicationError {
    kind: PublicationErrorKind,
    message: String,
}

impl PublicationError {
    /// Transport-neutral classification for driver-specific error mapping.
    pub const fn kind(&self) -> PublicationErrorKind {
        self.kind
    }

    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self::new(PublicationErrorKind::Unavailable, message)
    }

    pub(crate) fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::new(PublicationErrorKind::ResourceExhausted, message)
    }

    pub(crate) fn invalid_publication(message: impl Into<String>) -> Self {
        Self::new(PublicationErrorKind::InvalidPublication, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(PublicationErrorKind::Internal, message)
    }

    fn new(kind: PublicationErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl From<PublicationHubError> for PublicationError {
    fn from(error: PublicationHubError) -> Self {
        let kind = match error {
            PublicationHubError::UnknownPool(_) => PublicationErrorKind::NotFound,
            PublicationHubError::ProducerMismatch(_) => PublicationErrorKind::ProducerMismatch,
            PublicationHubError::Unavailable(_) => PublicationErrorKind::Unavailable,
            PublicationHubError::SubscriberLimit { .. }
            | PublicationHubError::InitializedHubLimit { .. }
            | PublicationHubError::SubscriberLagged(_) => PublicationErrorKind::ResourceExhausted,
            PublicationHubError::IdentityChanged { .. }
            | PublicationHubError::LeaseChanged { .. }
            | PublicationHubError::SequenceGap { .. }
            | PublicationHubError::BucketOutOfRange { .. } => {
                PublicationErrorKind::InvalidPublication
            }
        };
        Self::new(kind, error.to_string())
    }
}

/// A driver-facing, read-only view of the Relay's publication state.
///
/// Implementations own snapshot bootstrapping and stream continuity. Drivers
/// receive canonical frames and must not interact with actors or publication
/// hubs directly.
#[async_trait]
pub trait RelayPublicationSource: private::Sealed + Send + Sync {
    /// Identity of the Relay runtime that owns all advertised generations.
    fn relay_identity(&self) -> DcRelayIdentity;

    /// Watches complete pool-catalog snapshots, including the current value.
    fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog>;

    /// Watches complete serving-readiness snapshots, including the current value.
    fn watch_readiness(&self) -> watch::Receiver<Arc<TopologySnapshot>>;

    /// Watches complete authoritative pool-load snapshots, including the current value.
    fn watch_load(&self) -> watch::Receiver<Vec<PoolLoadSnapshot>>;

    /// Completes when the Relay begins shutting down.
    ///
    /// This is a read-only lifecycle signal. Dropping the returned future or the
    /// source does not affect Relay lifetime.
    async fn wait_for_shutdown(&self);

    /// Opens the exact producer generation requested by the driver.
    ///
    /// The returned stream begins with a complete snapshot and then emits
    /// contiguous deltas. Any lag, generation change, or sequence gap terminates
    /// the stream with an error; callers must open a new stream and snapshot.
    async fn subscribe_pool(
        &self,
        expected: ProducerIdentity,
    ) -> Result<PoolPublicationStream, PublicationError>;
}

#[derive(Clone)]
pub(crate) struct RegistryPublicationSource {
    pools: Arc<PoolRegistry>,
    topology: Arc<TopologyPublisher>,
    relay_identity: DcRelayIdentity,
    lifecycle: CancellationToken,
    snapshot_encoding_permits: Arc<Semaphore>,
    active_stream_permits: Arc<Semaphore>,
    max_active_streams: usize,
    snapshot_progress_timeout: Duration,
}

impl RegistryPublicationSource {
    pub(crate) fn new(
        pools: Arc<PoolRegistry>,
        topology: Arc<TopologyPublisher>,
        relay_identity: DcRelayIdentity,
        lifecycle: CancellationToken,
        snapshot_encoding_permits: Arc<Semaphore>,
        max_active_streams: usize,
        snapshot_progress_timeout: Duration,
    ) -> Self {
        debug_assert!(!snapshot_progress_timeout.is_zero());
        debug_assert_ne!(max_active_streams, 0);
        Self {
            pools,
            topology,
            relay_identity,
            lifecycle,
            snapshot_encoding_permits,
            active_stream_permits: Arc::new(Semaphore::new(max_active_streams)),
            max_active_streams,
            snapshot_progress_timeout,
        }
    }

    async fn acquire_stream_permits(
        &self,
        generation_cancel: &CancellationToken,
    ) -> Result<(OwnedSemaphorePermit, OwnedSemaphorePermit), PublicationError> {
        if self.lifecycle.is_cancelled() {
            return Err(PublicationError::unavailable(
                "publication source is shutting down",
            ));
        }

        let active_stream_permit = self
            .active_stream_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                PublicationError::resource_exhausted(format!(
                    "Relay reached its active pool stream limit {}",
                    self.max_active_streams
                ))
            })?;

        let snapshot_encoding_permit = tokio::select! {
            biased;
            _ = self.lifecycle.cancelled() => {
                return Err(PublicationError::unavailable(
                    "publication source is shutting down",
                ));
            }
            _ = generation_cancel.cancelled() => {
                return Err(PublicationError::unavailable(
                    "pool generation retired while waiting for publication admission",
                ));
            }
            permit = self.snapshot_encoding_permits.clone().acquire_owned() => permit
                .map_err(|_| PublicationError::unavailable(
                    "publication snapshot encoder is shutting down",
                ))?,
        };

        Ok((active_stream_permit, snapshot_encoding_permit))
    }

    fn terminal_failure(&self, expected: ProducerIdentity) -> TerminalPublicationFailure {
        let pools = Arc::downgrade(&self.pools);
        Arc::new(move |reason| {
            let Some(pools) = pools.upgrade() else {
                return;
            };
            let reason = format!("publication stream: {reason}");
            pools.fence_generation(expected.pool_id(), expected.layout_generation(), &reason);
        })
    }
}

impl private::Sealed for RegistryPublicationSource {}

#[async_trait]
impl RelayPublicationSource for RegistryPublicationSource {
    fn relay_identity(&self) -> DcRelayIdentity {
        self.relay_identity
    }

    fn watch_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.pools.watch_catalog()
    }

    fn watch_readiness(&self) -> watch::Receiver<Arc<TopologySnapshot>> {
        self.topology.watch()
    }

    fn watch_load(&self) -> watch::Receiver<Vec<PoolLoadSnapshot>> {
        self.pools.watch_load()
    }

    async fn wait_for_shutdown(&self) {
        self.lifecycle.cancelled().await;
    }

    async fn subscribe_pool(
        &self,
        expected: ProducerIdentity,
    ) -> Result<PoolPublicationStream, PublicationError> {
        if self.lifecycle.is_cancelled() {
            return Err(PublicationError::unavailable(
                "publication source is shutting down",
            ));
        }
        let generation_cancel = self.pools.validate_active_producer(expected)?;
        // Acquire encoding capacity first so permit waiters cannot pin a generation snapshot and
        // force full-lane COW while publication advances.
        let (active_stream_permit, snapshot_encoding_permit) =
            self.acquire_stream_permits(&generation_cancel).await?;
        let subscription = tokio::select! {
            biased;
            _ = self.lifecycle.cancelled() => {
                return Err(PublicationError::unavailable(
                    "publication source is shutting down",
                ));
            }
            subscription = self.pools.subscribe_pool(expected) => subscription?,
        };
        if self.lifecycle.is_cancelled() {
            return Err(PublicationError::unavailable(
                "publication source is shutting down",
            ));
        }
        stream::open(
            subscription,
            expected,
            active_stream_permit,
            snapshot_encoding_permit,
            self.snapshot_progress_timeout,
            self.lifecycle.clone(),
            self.terminal_failure(expected),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};

    use super::*;
    use crate::kv_dc_relay::discovery::DcMembershipView;
    use crate::kv_dc_relay::pool_registry::{PoolActorConfig, PoolPublicationConfig};

    fn producer() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn source(
        lifecycle: CancellationToken,
        snapshot_encoding_permits: Arc<Semaphore>,
        max_active_streams: usize,
    ) -> Arc<RegistryPublicationSource> {
        let relay_identity = DcRelayIdentity::new(11, 7);
        let pools = Arc::new(PoolRegistry::new_with_publication_config(
            relay_identity,
            PoolActorConfig {
                expected_unique_blocks: 32,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
            PoolPublicationConfig::default(),
        ));
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &pools.catalog(),
        ));
        Arc::new(RegistryPublicationSource::new(
            pools,
            topology,
            relay_identity,
            lifecycle,
            snapshot_encoding_permits,
            max_active_streams,
            DEFAULT_SNAPSHOT_PROGRESS_TIMEOUT,
        ))
    }

    #[tokio::test]
    async fn lifecycle_cancels_snapshot_permit_wait_without_cancel_authority() {
        let lifecycle = CancellationToken::new();
        let source = source(lifecycle.clone(), Arc::new(Semaphore::new(0)), 1);
        let generation_cancel = CancellationToken::new();
        let waiting_source = source.clone();
        let subscribe = tokio::spawn(async move {
            waiting_source
                .acquire_stream_permits(&generation_cancel)
                .await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while source.active_stream_permits.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit waiter must hold the active stream permit");

        let observing_source = source.clone();
        let lifecycle_wait = tokio::spawn(async move {
            observing_source.wait_for_shutdown().await;
        });
        lifecycle.cancel();

        tokio::time::timeout(Duration::from_secs(1), lifecycle_wait)
            .await
            .expect("lifecycle observer must wake")
            .expect("lifecycle observer task");
        let result = tokio::time::timeout(Duration::from_secs(1), subscribe)
            .await
            .expect("subscribe permit wait must cancel")
            .expect("subscribe task");
        let error = match result {
            Ok(_) => panic!("shutdown returned a publication stream"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), PublicationErrorKind::Unavailable);
        assert_eq!(source.active_stream_permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn generation_retirement_cancels_snapshot_permit_wait() {
        let lifecycle = CancellationToken::new();
        let source = source(lifecycle.clone(), Arc::new(Semaphore::new(0)), 1);
        let generation_cancel = CancellationToken::new();
        let waiting_cancel = generation_cancel.clone();
        let waiting_source = source.clone();
        let subscribe =
            tokio::spawn(
                async move { waiting_source.acquire_stream_permits(&waiting_cancel).await },
            );
        tokio::time::timeout(Duration::from_secs(1), async {
            while source.active_stream_permits.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("permit waiter must hold the active stream permit");

        generation_cancel.cancel();

        let result = tokio::time::timeout(Duration::from_secs(1), subscribe)
            .await
            .expect("generation retirement must cancel the permit wait")
            .expect("subscribe task");
        let error = match result {
            Ok(_) => panic!("retired generation acquired publication admission"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), PublicationErrorKind::Unavailable);
        assert!(!lifecycle.is_cancelled());
        assert_eq!(source.active_stream_permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn unknown_generation_is_rejected_before_permit_wait() {
        let source = source(CancellationToken::new(), Arc::new(Semaphore::new(0)), 1);

        let result =
            tokio::time::timeout(Duration::from_secs(1), source.subscribe_pool(producer()))
                .await
                .expect("identity precheck must not wait for a snapshot permit");
        let error = match result {
            Ok(_) => panic!("unknown generation opened a publication stream"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), PublicationErrorKind::NotFound);
        assert_eq!(source.active_stream_permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn active_pool_stream_limit_is_global_and_fail_fast() {
        let lifecycle = CancellationToken::new();
        let source = source(lifecycle.clone(), Arc::new(Semaphore::new(0)), 1);
        let generation_cancel = CancellationToken::new();
        let waiting_cancel = generation_cancel.clone();
        let waiting_source = source.clone();
        let first =
            tokio::spawn(
                async move { waiting_source.acquire_stream_permits(&waiting_cancel).await },
            );
        tokio::time::timeout(Duration::from_secs(1), async {
            while source.active_stream_permits.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first stream must hold the global permit");

        let error = match source.acquire_stream_permits(&generation_cancel).await {
            Ok(_) => panic!("second stream exceeded the global limit"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), PublicationErrorKind::ResourceExhausted);

        lifecycle.cancel();
        assert!(first.await.expect("first subscribe task").is_err());
        assert_eq!(source.active_stream_permits.available_permits(), 1);
    }
}
