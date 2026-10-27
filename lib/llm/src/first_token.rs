// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared worker-side first-token notification.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::sequences::SequencePublishQueueError;
use dynamo_runtime::{
    component::Endpoint, metrics::MetricsHierarchy, traits::DistributedRuntimeProvider,
};
use tokio::sync::watch;

use crate::kv_router::sequence::ActiveSequenceEventPublisher;
use crate::worker_type::WorkerType;

const COMPLETION_CHANNEL_CAPACITY: usize = 4096;

/// Endpoint-scoped source for per-request prefill-completion notifications.
#[derive(Clone)]
pub struct FirstTokenSource {
    publisher: ActiveSequenceEventPublisher,
    worker_id: u64,
    metrics: Option<FirstTokenPublisherMetrics>,
}

#[derive(Clone)]
struct FirstTokenPublisherMetrics {
    enqueue_failures_total: prometheus::IntCounterVec,
}

impl FirstTokenPublisherMetrics {
    fn for_endpoint(endpoint: &Endpoint) -> anyhow::Result<Self> {
        Ok(Self {
            enqueue_failures_total: endpoint.metrics().create_intcountervec(
                "worker_prefill_completion_enqueue_failures_total",
                "Total worker prefill-completion events rejected by the local publish queue",
                &["reason"],
                &[],
            )?,
        })
    }

    fn record_enqueue_failure(&self, error: &anyhow::Error) {
        let reason = match error.downcast_ref::<SequencePublishQueueError>() {
            Some(SequencePublishQueueError::Full { .. }) => "full",
            Some(SequencePublishQueueError::Closed { .. }) => "closed",
            None => return,
        };
        self.enqueue_failures_total
            .with_label_values(&[reason])
            .inc();
    }
}

impl FirstTokenSource {
    /// Create one completion source for an aggregated or decode serving endpoint.
    ///
    /// Event publication is advisory. Setup failures leave the source disabled so the router's
    /// response-side completion fallback remains authoritative.
    pub async fn for_endpoint(endpoint: &Endpoint, worker_type: WorkerType) -> Option<Self> {
        if !Self::supports_worker_type(worker_type) {
            return None;
        }

        let worker_id = endpoint.drt().connection_id();
        let metrics = match FirstTokenPublisherMetrics::for_endpoint(endpoint) {
            Ok(metrics) => Some(metrics),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "worker prefill-completion metrics unavailable"
                );
                None
            }
        };
        let publisher =
            ActiveSequenceEventPublisher::for_endpoint(endpoint, COMPLETION_CHANNEL_CAPACITY).await;
        Self::from_publisher_result_with_metrics(worker_id, publisher, metrics)
    }

    #[cfg(test)]
    pub(crate) fn from_publisher(
        publisher: ActiveSequenceEventPublisher,
        worker_id: u64,
        worker_type: WorkerType,
    ) -> Option<Self> {
        Self::supports_worker_type(worker_type).then_some(Self {
            publisher,
            worker_id,
            metrics: None,
        })
    }

    #[cfg(test)]
    fn from_publisher_result(
        worker_id: u64,
        publisher: anyhow::Result<ActiveSequenceEventPublisher>,
    ) -> Option<Self> {
        Self::from_publisher_result_with_metrics(worker_id, publisher, None)
    }

    fn from_publisher_result_with_metrics(
        worker_id: u64,
        publisher: anyhow::Result<ActiveSequenceEventPublisher>,
        metrics: Option<FirstTokenPublisherMetrics>,
    ) -> Option<Self> {
        match publisher {
            Ok(publisher) => Some(Self {
                publisher,
                worker_id,
                metrics,
            }),
            Err(error) => {
                tracing::warn!(
                    %error,
                    "worker prefill-completion publisher unavailable; continuing with response-side cleanup"
                );
                None
            }
        }
    }

    const fn supports_worker_type(worker_type: WorkerType) -> bool {
        matches!(worker_type, WorkerType::Aggregated | WorkerType::Decode)
    }
}

/// Shared one-shot action fired by explicit engine notification or first output observation.
#[derive(Clone)]
pub struct FirstTokenNotifier {
    inner: Arc<FirstTokenNotifierInner>,
}

struct FirstTokenNotifierInner {
    notified: AtomicBool,
    abort_sender: Option<watch::Sender<bool>>,
    completion: OnceLock<FirstTokenCompletion>,
}

struct FirstTokenCompletion {
    source: FirstTokenSource,
    request_id: String,
    dp_rank: u32,
}

impl FirstTokenNotifier {
    /// Build a notifier for the actions available on this request.
    ///
    /// A missing source or DP rank only disables completion publication. Decode abort release
    /// remains active when its sender is present.
    pub fn for_request(
        abort_sender: Option<watch::Sender<bool>>,
        source: Option<&FirstTokenSource>,
        request_id: &str,
        dp_rank: Option<u32>,
    ) -> Option<Self> {
        let completion = source
            .zip(dp_rank)
            .map(|(source, dp_rank)| FirstTokenCompletion {
                source: source.clone(),
                request_id: request_id.to_string(),
                dp_rank,
            });
        if abort_sender.is_none() && completion.is_none() {
            return None;
        }

        let completion_cell = OnceLock::new();
        if let Some(completion) = completion {
            let _ = completion_cell.set(completion);
        }

        Some(Self {
            inner: Arc::new(FirstTokenNotifierInner {
                notified: AtomicBool::new(false),
                abort_sender,
                completion: completion_cell,
            }),
        })
    }

    /// Attach completion publication to this shared gate before output observation.
    ///
    /// Returns `false` when an action is already attached. Existing abort release and one-shot
    /// state remain shared by every notifier clone.
    #[doc(hidden)]
    pub fn attach_completion(
        &self,
        source: &FirstTokenSource,
        request_id: &str,
        dp_rank: u32,
    ) -> bool {
        self.inner
            .completion
            .set(FirstTokenCompletion {
                source: source.clone(),
                request_id: request_id.to_string(),
                dp_rank,
            })
            .is_ok()
    }

    /// Run the configured abort-release and completion-publication actions at most once.
    pub fn notify(&self) {
        if self.inner.notified.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(sender) = &self.inner.abort_sender {
            let _ = sender.send(true);
        }
        if let Some(completion) = self.inner.completion.get() {
            let result = completion.source.publisher.mark_prefill_completed(
                completion.request_id.clone(),
                completion.source.worker_id,
                completion.dp_rank,
            );
            if let (Err(error), Some(metrics)) = (result, &completion.source.metrics) {
                metrics.record_enqueue_failure(&error);
            }
        }
    }

    #[doc(hidden)]
    pub fn abort_sender(&self) -> Option<&watch::Sender<bool>> {
        self.inner.abort_sender.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc::error::TryRecvError;
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn source(
        worker_type: WorkerType,
    ) -> (
        Option<FirstTokenSource>,
        tokio::sync::mpsc::Receiver<dynamo_kv_router::protocols::ActiveSequenceEvent>,
    ) {
        let (publisher, receiver) =
            ActiveSequenceEventPublisher::channel(4, CancellationToken::new());
        (
            FirstTokenSource::from_publisher(publisher, 7, worker_type),
            receiver,
        )
    }

    #[test]
    fn only_aggregated_and_decode_roles_create_sources() {
        assert!(source(WorkerType::Aggregated).0.is_some());
        assert!(source(WorkerType::Decode).0.is_some());
        assert!(source(WorkerType::Prefill).0.is_none());
        assert!(source(WorkerType::Encode).0.is_none());
    }

    #[test]
    fn missing_dp_rank_disables_completion_notification() {
        let (source, mut receiver) = source(WorkerType::Decode);
        let notifier = FirstTokenNotifier::for_request(None, source.as_ref(), "missing-rank", None);

        assert!(notifier.is_none());
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn duplicate_notification_publishes_once() {
        let (source, mut receiver) = source(WorkerType::Decode);
        let notifier =
            FirstTokenNotifier::for_request(None, source.as_ref(), "request-1", Some(3)).unwrap();

        notifier.notify();
        notifier.notify();

        let event = receiver.try_recv().unwrap();
        assert_eq!(event.request_id, "request-1");
        assert_eq!(event.worker.worker_id, 7);
        assert_eq!(event.worker.dp_rank, 3);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn attaching_completion_preserves_abort_and_shared_gate() {
        let (source, mut receiver) = source(WorkerType::Decode);
        let source = source.unwrap();
        let (abort_tx, mut abort_rx) = watch::channel(false);
        let notifier = FirstTokenNotifier::for_request(Some(abort_tx), None, "", None).unwrap();
        let notifier_clone = notifier.clone();

        assert!(notifier.attach_completion(&source, "request-1", 3));
        assert!(!notifier_clone.attach_completion(&source, "request-2", 4));

        notifier.notify();
        assert!(abort_rx.has_changed().unwrap());
        assert!(*abort_rx.borrow_and_update());
        let event = receiver.try_recv().unwrap();
        assert_eq!(event.request_id, "request-1");
        assert_eq!(event.worker.dp_rank, 3);

        notifier_clone.notify();
        assert!(!abort_rx.has_changed().unwrap());
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn enqueue_failures_are_counted_by_reason() {
        let cancellation_token = CancellationToken::new();
        let (publisher, receiver) = ActiveSequenceEventPublisher::channel(1, cancellation_token);
        let metrics = FirstTokenPublisherMetrics {
            enqueue_failures_total: prometheus::IntCounterVec::new(
                prometheus::Opts::new("test_enqueue_failures_total", "test counter"),
                &["reason"],
            )
            .unwrap(),
        };
        let source = FirstTokenSource {
            publisher,
            worker_id: 7,
            metrics: Some(metrics.clone()),
        };

        FirstTokenNotifier::for_request(None, Some(&source), "accepted", Some(0))
            .unwrap()
            .notify();
        FirstTokenNotifier::for_request(None, Some(&source), "full", Some(0))
            .unwrap()
            .notify();
        assert_eq!(
            metrics
                .enqueue_failures_total
                .with_label_values(&["full"])
                .get(),
            1
        );

        drop(receiver);
        FirstTokenNotifier::for_request(None, Some(&source), "closed", Some(0))
            .unwrap()
            .notify();
        assert_eq!(
            metrics
                .enqueue_failures_total
                .with_label_values(&["closed"])
                .get(),
            1
        );
    }

    #[test]
    fn publisher_setup_failure_is_fail_open() {
        let source = FirstTokenSource::from_publisher_result(
            7,
            Err(anyhow::anyhow!("publisher unavailable")),
        );

        assert!(source.is_none());
    }
}
