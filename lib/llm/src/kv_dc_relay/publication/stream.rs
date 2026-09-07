// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Turns a hub subscription into a snapshot-first stream while enforcing producer identity and
//! contiguous sequencing.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread::sleep;
use std::time::{Duration, Instant};

use async_stream::stream;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;

use super::codec::{PublicationFrame, PublicationFrameKind, encode_snapshot};
use super::hub::PublicationHubSubscription;
use super::source::PublicationError;

const SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY: usize = 1;
const SNAPSHOT_SEND_POLL_INTERVAL: Duration = Duration::from_millis(10);

type PublicationResult = Result<Arc<PublicationFrame>, PublicationError>;
type BoxedPublicationStream = Pin<Box<dyn Stream<Item = PublicationResult> + Send + 'static>>;
pub(super) type TerminalPublicationFailure = Arc<dyn Fn(String) + Send + Sync>;

/// Canonical snapshot chunks followed by contiguous deltas for one producer.
pub struct PoolPublicationStream {
    inner: BoxedPublicationStream,
    _active_stream_permit: OwnedSemaphorePermit,
}

impl Stream for PoolPublicationStream {
    type Item = PublicationResult;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(context)
    }
}

pub(super) fn open(
    mut subscription: PublicationHubSubscription,
    expected: ProducerIdentity,
    active_stream_permit: OwnedSemaphorePermit,
    encoding_permit: OwnedSemaphorePermit,
    progress_timeout: Duration,
    lifecycle: CancellationToken,
    terminal_failure: TerminalPublicationFailure,
) -> Result<PoolPublicationStream, PublicationError> {
    let snapshot = subscription.take_snapshot()?;
    if snapshot.identity() != expected {
        let error = PublicationError::invalid_publication(format!(
            "publication snapshot identity changed from {expected:?} to {:?}",
            snapshot.identity()
        ));
        report_terminal_failure(&terminal_failure, &error);
        return Err(error);
    }
    let snapshot_sequence = snapshot.sequence();
    let frames = match encode_snapshot(snapshot) {
        Ok(frames) => frames,
        Err(error) => {
            let error = PublicationError::invalid_publication(format!(
                "failed to encode publication snapshot: {error}"
            ));
            report_terminal_failure(&terminal_failure, &error);
            return Err(error);
        }
    };
    subscription.ensure_active()?;
    let bootstrap = SnapshotBootstrap::spawn(frames, encoding_permit, progress_timeout);

    let inner = stream! {
        let mut bootstrap = bootstrap;
        while let Some(frame) = match tokio::select! {
            biased;
            _ = lifecycle.cancelled() => return,
            result = bootstrap.recv() => result,
        } {
            Ok(frame) => frame,
            Err(error) => {
                yield Err(error.into_publication_error(&terminal_failure));
                return;
            }
        } {
            if let Err(error) = subscription.ensure_active() {
                yield Err(error.into());
                return;
            }
            if let Err(error) = validate_snapshot_frame(&frame, expected, snapshot_sequence) {
                report_terminal_failure(&terminal_failure, &error);
                yield Err(error);
                return;
            }
            yield Ok(Arc::new(frame));
        }

        if let Err(error) = subscription.ensure_active() {
            yield Err(error.into());
            return;
        }

        let mut current_sequence = snapshot_sequence;
        loop {
            let frame = match tokio::select! {
                biased;
                _ = lifecycle.cancelled() => return,
                result = subscription.recv() => result,
            } {
                Ok(frame) => frame,
                Err(error) => {
                    yield Err(error.into());
                    return;
                }
            };
            if let Err(error) = validate_delta_frame(&frame, expected, current_sequence) {
                report_terminal_failure(&terminal_failure, &error);
                yield Err(error);
                return;
            }
            current_sequence = frame.sequence();
            yield Ok(frame);
        }
    };
    Ok(PoolPublicationStream {
        inner: Box::pin(inner),
        _active_stream_permit: active_stream_permit,
    })
}

fn report_terminal_failure(
    terminal_failure: &TerminalPublicationFailure,
    error: &PublicationError,
) {
    terminal_failure(error.to_string());
}

fn validate_snapshot_frame(
    frame: &PublicationFrame,
    expected: ProducerIdentity,
    snapshot_sequence: u64,
) -> Result<(), PublicationError> {
    if frame.identity() != expected {
        return Err(PublicationError::invalid_publication(format!(
            "snapshot frame identity changed from {expected:?} to {:?}",
            frame.identity()
        )));
    }
    if frame.kind() != PublicationFrameKind::SnapshotChunk
        || frame.base_sequence() != snapshot_sequence
        || frame.sequence() != snapshot_sequence
    {
        return Err(PublicationError::invalid_publication(format!(
            "invalid snapshot frame at sequence {}: kind={:?}, base={}, next={}",
            snapshot_sequence,
            frame.kind(),
            frame.base_sequence(),
            frame.sequence()
        )));
    }
    Ok(())
}

fn validate_delta_frame(
    frame: &PublicationFrame,
    expected: ProducerIdentity,
    current_sequence: u64,
) -> Result<(), PublicationError> {
    if frame.identity() != expected {
        return Err(PublicationError::invalid_publication(format!(
            "delta frame identity changed from {expected:?} to {:?}",
            frame.identity()
        )));
    }
    let Some(expected_sequence) = current_sequence.checked_add(1) else {
        return Err(PublicationError::invalid_publication(
            "publication sequence space exhausted",
        ));
    };
    if frame.kind() != PublicationFrameKind::Delta
        || frame.base_sequence() != current_sequence
        || frame.sequence() != expected_sequence
    {
        return Err(PublicationError::invalid_publication(format!(
            "non-contiguous delta: current={current_sequence}, kind={:?}, base={}, next={}",
            frame.kind(),
            frame.base_sequence(),
            frame.sequence()
        )));
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum SnapshotBootstrapError {
    #[error("snapshot encoding task failed: {0}")]
    EncoderTaskFailed(JoinError),
    #[error("snapshot subscriber made no progress for {0:?}")]
    ProgressTimeout(Duration),
}

impl SnapshotBootstrapError {
    fn into_publication_error(
        self,
        terminal_failure: &TerminalPublicationFailure,
    ) -> PublicationError {
        match self {
            Self::EncoderTaskFailed(error) if error.is_cancelled() => {
                PublicationError::unavailable("publication snapshot encoder task was cancelled")
            }
            Self::EncoderTaskFailed(error) => {
                let error = PublicationError::internal(format!(
                    "publication snapshot encoder task failed: {error}"
                ));
                report_terminal_failure(terminal_failure, &error);
                error
            }
            Self::ProgressTimeout(timeout) => PublicationError::resource_exhausted(format!(
                "publication snapshot subscriber made no progress for {} ms; open a fresh stream",
                timeout.as_millis()
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotEncoderExit {
    Complete,
    Cancelled,
    ReceiverClosed,
    ProgressTimeout,
}

struct SnapshotBootstrap {
    receiver: mpsc::Receiver<PublicationFrame>,
    task: Option<JoinHandle<SnapshotEncoderExit>>,
    cancel: CancellationToken,
    progress_timeout: Duration,
}

impl SnapshotBootstrap {
    fn spawn<I>(
        frames: I,
        encoding_permit: OwnedSemaphorePermit,
        progress_timeout: Duration,
    ) -> Self
    where
        I: Iterator<Item = PublicationFrame> + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel(SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY);
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::task::spawn_blocking(move || {
            let _encoding_permit = encoding_permit;
            for mut frame in frames {
                let stalled_since = Instant::now();
                loop {
                    if task_cancel.is_cancelled() {
                        return SnapshotEncoderExit::Cancelled;
                    }
                    match sender.try_send(frame) {
                        Ok(()) => break,
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            return SnapshotEncoderExit::ReceiverClosed;
                        }
                        Err(mpsc::error::TrySendError::Full(returned)) => {
                            frame = returned;
                            let elapsed = stalled_since.elapsed();
                            if elapsed >= progress_timeout {
                                return SnapshotEncoderExit::ProgressTimeout;
                            }
                            sleep(
                                progress_timeout
                                    .saturating_sub(elapsed)
                                    .min(SNAPSHOT_SEND_POLL_INTERVAL),
                            );
                        }
                    }
                }
            }
            SnapshotEncoderExit::Complete
        });
        Self {
            receiver,
            task: Some(task),
            cancel,
            progress_timeout,
        }
    }

    async fn recv(&mut self) -> Result<Option<PublicationFrame>, SnapshotBootstrapError> {
        if let Some(frame) = self.receiver.recv().await {
            return Ok(Some(frame));
        }
        let Some(task) = self.task.take() else {
            return Ok(None);
        };
        match task
            .await
            .map_err(SnapshotBootstrapError::EncoderTaskFailed)?
        {
            SnapshotEncoderExit::ProgressTimeout => Err(SnapshotBootstrapError::ProgressTimeout(
                self.progress_timeout,
            )),
            SnapshotEncoderExit::Complete
            | SnapshotEncoderExit::Cancelled
            | SnapshotEncoderExit::ReceiverClosed => Ok(None),
        }
    }
}

impl Drop for SnapshotBootstrap {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.receiver.close();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};
    use tokio::sync::Semaphore;

    use super::*;

    struct CountingFrames {
        frame: PublicationFrame,
        remaining: usize,
        produced: Arc<AtomicUsize>,
    }

    impl Iterator for CountingFrames {
        type Item = PublicationFrame;

        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            self.produced.fetch_add(1, Ordering::Relaxed);
            Some(self.frame.clone())
        }
    }

    struct PanickingFrames;

    impl Iterator for PanickingFrames {
        type Item = PublicationFrame;

        fn next(&mut self) -> Option<Self::Item> {
            panic!("injected snapshot encoder panic")
        }
    }

    fn identity(seed: u8) -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn frame(
        identity: ProducerIdentity,
        base_sequence: u64,
        sequence: u64,
        kind: PublicationFrameKind,
    ) -> PublicationFrame {
        PublicationFrame::test_frame(identity, base_sequence, sequence, kind)
    }

    async fn encoding_permit() -> OwnedSemaphorePermit {
        Arc::new(Semaphore::new(1))
            .acquire_owned()
            .await
            .expect("fixture semaphore is open")
    }

    fn counting_terminal_failures() -> (TerminalPublicationFailure, Arc<AtomicUsize>) {
        let failures = Arc::new(AtomicUsize::new(0));
        let observed = failures.clone();
        (
            Arc::new(move |_| {
                observed.fetch_add(1, Ordering::Relaxed);
            }),
            failures,
        )
    }

    #[tokio::test]
    async fn snapshot_bootstrap_keeps_only_one_frame_queued() {
        let produced = Arc::new(AtomicUsize::new(0));
        let frames = CountingFrames {
            frame: frame(identity(1), 4, 4, PublicationFrameKind::SnapshotChunk),
            remaining: 3,
            produced: produced.clone(),
        };
        let mut bootstrap =
            SnapshotBootstrap::spawn(frames, encoding_permit().await, Duration::from_secs(1));

        tokio::time::timeout(Duration::from_secs(1), async {
            while produced.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("snapshot encoder must start");
        assert_eq!(produced.load(Ordering::Relaxed), 2);

        for _ in 0..3 {
            let next = tokio::time::timeout(Duration::from_secs(1), bootstrap.recv())
                .await
                .expect("bootstrap must make progress")
                .expect("encoder must remain healthy");
            assert!(next.is_some());
        }
        assert!(
            tokio::time::timeout(Duration::from_secs(1), bootstrap.recv())
                .await
                .expect("encoder must finish")
                .expect("encoder must remain healthy")
                .is_none()
        );
        assert_eq!(produced.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn stalled_snapshot_bootstrap_requires_a_fresh_stream() {
        let produced = Arc::new(AtomicUsize::new(0));
        let timeout = Duration::from_millis(20);
        let frames = CountingFrames {
            frame: frame(identity(1), 4, 4, PublicationFrameKind::SnapshotChunk),
            remaining: 3,
            produced: produced.clone(),
        };
        let mut bootstrap = SnapshotBootstrap::spawn(frames, encoding_permit().await, timeout);

        tokio::time::timeout(Duration::from_secs(1), async {
            while produced.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("snapshot encoder must start");
        tokio::time::sleep(timeout + Duration::from_millis(20)).await;
        assert!(
            bootstrap
                .recv()
                .await
                .expect("first frame is buffered")
                .is_some()
        );
        let error = bootstrap.recv().await.expect_err("bootstrap must time out");
        assert!(matches!(
            &error,
            SnapshotBootstrapError::ProgressTimeout(observed) if *observed == timeout
        ));
        let (terminal_failure, failures) = counting_terminal_failures();
        assert_eq!(
            error.into_publication_error(&terminal_failure).kind(),
            super::super::source::PublicationErrorKind::ResourceExhausted
        );
        assert_eq!(failures.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn encoder_panic_fences_but_task_cancellation_does_not() {
        let (terminal_failure, failures) = counting_terminal_failures();
        let mut bootstrap = SnapshotBootstrap::spawn(
            PanickingFrames,
            encoding_permit().await,
            Duration::from_secs(1),
        );
        let error = bootstrap.recv().await.expect_err("encoder must panic");
        assert_eq!(
            error.into_publication_error(&terminal_failure).kind(),
            super::super::source::PublicationErrorKind::Internal
        );
        assert_eq!(failures.load(Ordering::Relaxed), 1);

        let task = tokio::spawn(std::future::pending::<()>());
        task.abort();
        let cancelled = task.await.expect_err("task must be cancelled");
        assert_eq!(
            SnapshotBootstrapError::EncoderTaskFailed(cancelled)
                .into_publication_error(&terminal_failure)
                .kind(),
            super::super::source::PublicationErrorKind::Unavailable
        );
        assert_eq!(failures.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn pool_stream_holds_global_permit_until_drop() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits
            .clone()
            .acquire_owned()
            .await
            .expect("stream permit");
        let stream = PoolPublicationStream {
            inner: Box::pin(futures::stream::pending()),
            _active_stream_permit: permit,
        };
        assert!(permits.clone().try_acquire_owned().is_err());

        drop(stream);
        assert!(permits.try_acquire_owned().is_ok());
    }

    #[test]
    fn frame_validation_rejects_identity_and_sequence_drift() {
        let producer = identity(1);
        let other = identity(2);
        assert!(
            validate_snapshot_frame(
                &frame(producer, 4, 4, PublicationFrameKind::SnapshotChunk),
                producer,
                4,
            )
            .is_ok()
        );
        assert_eq!(
            validate_snapshot_frame(
                &frame(other, 4, 4, PublicationFrameKind::SnapshotChunk),
                producer,
                4,
            )
            .expect_err("identity drift must fail")
            .kind(),
            super::super::source::PublicationErrorKind::InvalidPublication
        );

        assert!(
            validate_delta_frame(
                &frame(producer, 4, 5, PublicationFrameKind::Delta),
                producer,
                4,
            )
            .is_ok()
        );
        assert_eq!(
            validate_delta_frame(
                &frame(producer, 3, 5, PublicationFrameKind::Delta),
                producer,
                4,
            )
            .expect_err("sequence gap must fail")
            .kind(),
            super::super::source::PublicationErrorKind::InvalidPublication
        );

        let (terminal_failure, failures) = counting_terminal_failures();
        let error = validate_delta_frame(
            &frame(producer, 3, 5, PublicationFrameKind::Delta),
            producer,
            4,
        )
        .expect_err("sequence gap must fail");
        report_terminal_failure(&terminal_failure, &error);
        assert_eq!(failures.load(Ordering::Relaxed), 1);
    }
}
