// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use dynamo_kv_router::{protocols::WorkerWithDpRank, scheduling::AdmissionAttempt};
use dynamo_runtime::{
    component::Client,
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ResponseStream},
    pipeline::{
        Error, ExactDispatchError, ManyOut, OccupancyReservation, PushRouter, RouterMode, SingleIn,
    },
    protocols::annotated::Annotated,
};
use futures::Stream;
use tokio_util::sync::CancellationToken;

use super::{
    ExternalSpeculationMetrics, SpeculationChooser, SpeculationCompositionSnapshot,
    SpeculationSelection,
};
use crate::protocols::{
    common::{FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest},
    external_speculation::{
        DraftCleanupOutcomeV1, EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY,
        ExternalSpeculationLifecycleV1, RouterHintEnvelope, SpeculativeDecodingRouterHintV1,
    },
};

type LlmPushRouter = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;
const RELEASE_RETRY_INITIAL_DELAY: Duration = Duration::from_millis(10);
const RELEASE_RETRY_MAX_DELAY: Duration = Duration::from_secs(1);

struct SelectionLeaseState {
    chooser: Arc<dyn SpeculationChooser>,
    request_id: Arc<str>,
    worker: WorkerWithDpRank,
    attempt: AdmissionAttempt,
    occupancy: Option<OccupancyReservation>,
}

impl SelectionLeaseState {
    async fn release_until_success(&self) {
        let mut retry_delay = RELEASE_RETRY_INITIAL_DELAY;
        loop {
            match self
                .chooser
                .release(&self.request_id, self.worker, self.attempt)
                .await
            {
                Ok(()) => return,
                Err(error) => {
                    tracing::warn!(
                        request_id = %self.request_id,
                        worker_id = self.worker.worker_id,
                        dp_rank = self.worker.dp_rank,
                        %error,
                        retry_delay_ms = retry_delay.as_millis(),
                        "Retrying speculative booking release"
                    );
                    tokio::time::sleep(retry_delay).await;
                    retry_delay = retry_delay.saturating_mul(2).min(RELEASE_RETRY_MAX_DELAY);
                }
            }
        }
    }
}

struct SelectionLease {
    state: Option<SelectionLeaseState>,
}

impl SelectionLease {
    fn new(
        chooser: Arc<dyn SpeculationChooser>,
        request_id: Arc<str>,
        worker: WorkerWithDpRank,
        attempt: AdmissionAttempt,
        occupancy: Option<OccupancyReservation>,
    ) -> Self {
        Self {
            state: Some(SelectionLeaseState {
                chooser,
                request_id,
                worker,
                attempt,
                occupancy,
            }),
        }
    }

    async fn release(mut self) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        drop(state.occupancy.take());
        if !state.chooser.router_mode().is_kv_routing() {
            self.state.take();
            return;
        }

        state.release_until_success().await;
        self.state.take();
    }

    fn spawn_release(&mut self) {
        let Some(mut state) = self.state.take() else {
            return;
        };
        drop(state.occupancy.take());
        if !state.chooser.router_mode().is_kv_routing() {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                request_id = %state.request_id,
                worker_id = state.worker.worker_id,
                dp_rank = state.worker.dp_rank,
                "No Tokio runtime available to release speculative KV-router booking"
            );
            return;
        };
        handle.spawn(async move {
            state.release_until_success().await;
        });
    }
}

impl Drop for SelectionLease {
    fn drop(&mut self) {
        self.spawn_release();
    }
}

struct PairLease {
    target: Option<SelectionLease>,
    draft: Option<SelectionLease>,
    orphan_cleanup_timeout: Duration,
    metrics: Arc<ExternalSpeculationMetrics>,
    dispatch_attempted: bool,
}

impl PairLease {
    fn new(
        target: SelectionLease,
        draft: SelectionLease,
        orphan_cleanup_timeout: Duration,
        metrics: Arc<ExternalSpeculationMetrics>,
    ) -> Self {
        Self {
            target: Some(target),
            draft: Some(draft),
            orphan_cleanup_timeout,
            metrics,
            dispatch_attempted: false,
        }
    }

    fn mark_dispatch_attempted(&mut self) {
        self.dispatch_attempted = true;
    }

    async fn release_all(&mut self) {
        let target = self.target.take();
        let draft = self.draft.take();
        tokio::join!(
            async move {
                if let Some(target) = target {
                    target.release().await;
                }
            },
            async move {
                if let Some(draft) = draft {
                    draft.release().await;
                }
            }
        );
    }

    async fn release_target(&mut self) {
        if let Some(target) = self.target.take() {
            target.release().await;
        }
    }

    fn take_draft_for_quarantine(&mut self) -> Self {
        let draft = self
            .draft
            .take()
            .expect("a speculative pair must own its draft lease before quarantine");
        Self {
            target: None,
            draft: Some(draft),
            orphan_cleanup_timeout: self.orphan_cleanup_timeout,
            metrics: self.metrics.clone(),
            dispatch_attempted: self.dispatch_attempted,
        }
    }
}

impl Drop for PairLease {
    fn drop(&mut self) {
        if !self.dispatch_attempted && (self.target.is_some() || self.draft.is_some()) {
            self.metrics.observe_lifecycle("pre_dispatch_rollback");
        }
    }
}

/// Owns a selected pair across the cancellation-sensitive dispatch await.
///
/// Once dispatch starts, dropping the route future cannot prove that the request stayed local.
/// An armed guard therefore retains both leases through the same bounded quarantine used for an
/// explicit delivery-unknown result.
struct DispatchGuard {
    leases: Option<PairLease>,
    cleanup: Arc<BackgroundCleanup>,
}

impl DispatchGuard {
    fn new(leases: PairLease, cleanup: Arc<BackgroundCleanup>) -> Self {
        Self {
            leases: Some(leases),
            cleanup,
        }
    }

    fn take(&mut self) -> PairLease {
        self.leases
            .take()
            .expect("dispatch guard must own its leases while armed")
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        if let Some(leases) = self.leases.take() {
            let timeout = leases.orphan_cleanup_timeout;
            self.cleanup.spawn_quarantine(leases, timeout);
        }
    }
}

struct BackgroundCleanup {
    cancellation: CancellationToken,
    active: Arc<AtomicUsize>,
    metrics: Arc<ExternalSpeculationMetrics>,
}

impl BackgroundCleanup {
    fn new(cancellation: CancellationToken, metrics: Arc<ExternalSpeculationMetrics>) -> Self {
        Self {
            cancellation,
            active: Arc::new(AtomicUsize::new(0)),
            metrics,
        }
    }

    fn spawn_quarantine(&self, mut leases: PairLease, timeout: Duration) {
        let cancellation = self.cancellation.clone();
        let active = self.active.clone();
        let metrics = self.metrics.clone();
        let deadline = tokio::time::Instant::now() + timeout;
        active.fetch_add(1, Ordering::AcqRel);
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            active.fetch_sub(1, Ordering::AcqRel);
            metrics.observe_lifecycle("quarantine_runtime_unavailable");
            tracing::warn!("No Tokio runtime available for external-speculation quarantine");
            drop(leases);
            return;
        };
        metrics.begin_quarantine();
        handle.spawn(async move {
            tokio::select! {
                _ = cancellation.cancelled() => {
                    tracing::debug!("Expiring external-speculation quarantine during shutdown");
                }
                _ = tokio::time::sleep_until(deadline) => {
                    tracing::debug!("External-speculation quarantine bound elapsed");
                }
            }
            let mut draft = leases.take_draft_for_quarantine();
            let draft_metrics = metrics.clone();
            tokio::join!(async move { leases.release_target().await }, async move {
                draft.release_all().await;
                draft_metrics.end_quarantine();
            });
            active.fetch_sub(1, Ordering::AcqRel);
        });
    }

    fn spawn_drain(&self, mut stream: ManyOut<Annotated<LLMEngineOutput>>, mut leases: PairLease) {
        let cancellation = self.cancellation.clone();
        let active = self.active.clone();
        let metrics = self.metrics.clone();
        let timeout = leases.orphan_cleanup_timeout;
        let deadline = tokio::time::Instant::now() + timeout;
        active.fetch_add(1, Ordering::AcqRel);
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            active.fetch_sub(1, Ordering::AcqRel);
            metrics.observe_lifecycle("drain_runtime_unavailable");
            tracing::warn!("No Tokio runtime available to drain speculative target response");
            return;
        };
        metrics.begin_quarantine();
        metrics.observe_lifecycle("background_drain_started");
        handle.spawn(async move {
            let mut cleanup_confirmed = false;
            loop {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => break,
                    _ = tokio::time::sleep_until(deadline) => break,
                    item = futures::StreamExt::next(&mut stream) => {
                        let Some(mut item) = item else { break };
                        cleanup_confirmed |= observe_cleanup(&mut item);
                        if cleanup_confirmed {
                            break;
                        }
                    }
                }
            }
            let mut draft = leases.take_draft_for_quarantine();
            let draft_metrics = metrics.clone();
            tokio::join!(async move { leases.release_target().await }, async move {
                if !cleanup_confirmed
                    && tokio::time::Instant::now() < deadline
                    && !cancellation.is_cancelled()
                {
                    tokio::select! {
                        _ = cancellation.cancelled() => {}
                        _ = tokio::time::sleep_until(deadline) => {}
                    }
                }
                draft.release_all().await;
                draft_metrics.end_quarantine();
            });
            metrics.observe_lifecycle(if cleanup_confirmed {
                "background_drain_cleanup_confirmed"
            } else {
                "background_drain_bound_elapsed"
            });
            active.fetch_sub(1, Ordering::AcqRel);
        });
    }
}

impl Drop for BackgroundCleanup {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Pairs independent target/draft selections, then dispatches to the exact target rank.
pub struct ExternalSpeculationRouter {
    inner: LlmPushRouter,
    target_client: Client,
    composition: Arc<ArcSwapOption<SpeculationCompositionSnapshot>>,
    cleanup: Arc<BackgroundCleanup>,
    metrics: Arc<ExternalSpeculationMetrics>,
}

impl ExternalSpeculationRouter {
    pub async fn new(
        target_client: Client,
        composition: Arc<ArcSwapOption<SpeculationCompositionSnapshot>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<Self> {
        let inner =
            LlmPushRouter::from_client_with_monitor(target_client.clone(), RouterMode::KV, None)
                .await?;
        let metrics =
            ExternalSpeculationMetrics::from_component(target_client.endpoint.component());
        Ok(Self {
            inner,
            target_client,
            composition,
            cleanup: Arc::new(BackgroundCleanup::new(cancellation, metrics.clone())),
            metrics,
        })
    }

    async fn route(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let composition = self
            .composition
            .load_full()
            .ok_or_else(|| anyhow::anyhow!("external-speculation composition is not ready"))?;
        let request_context = request.context();
        let request_id: Arc<str> = Arc::from(request_context.id());
        let selection_started = Instant::now();

        let mut target = composition
            .target_chooser
            .select_and_reserve(&request_id, &request, &composition.target_workers)
            .await?;
        let target_lease = SelectionLease::new(
            composition.target_chooser.clone(),
            request_id.clone(),
            target.worker,
            target.attempt,
            target.occupancy.take(),
        );
        let mut draft = match composition
            .draft_chooser
            .select_and_reserve(&request_id, &request, &composition.draft_workers)
            .await
        {
            Ok(draft) => draft,
            Err(error) => {
                target_lease.release().await;
                return Err(error);
            }
        };
        let draft_lease = SelectionLease::new(
            composition.draft_chooser.clone(),
            request_id,
            draft.worker,
            draft.attempt,
            draft.occupancy.take(),
        );

        let Some(transport) = composition.draft_transports.get(&draft.worker).cloned() else {
            tokio::join!(target_lease.release(), draft_lease.release());
            return Err(anyhow::anyhow!(
                "selected draft has no transport descriptor"
            ));
        };
        let mut leases = PairLease::new(
            target_lease,
            draft_lease,
            Duration::from_millis(transport.orphan_cleanup_timeout_ms.into()),
            self.metrics.clone(),
        );

        let composition_valid = (|| -> anyhow::Result<()> {
            let latest = self.composition.load_full().ok_or_else(|| {
                anyhow::anyhow!("external-speculation composition became unavailable")
            })?;
            anyhow::ensure!(
                Arc::ptr_eq(&composition, &latest),
                "external-speculation composition changed during selection"
            );
            anyhow::ensure!(
                composition.target_workers.contains(&target.worker),
                "selected target worker is outside the committed composition"
            );
            anyhow::ensure!(
                self.target_client
                    .instance_ids()
                    .contains(&target.worker.worker_id),
                "selected target worker left discovery before dispatch"
            );
            Ok(())
        })();
        if let Err(error) = composition_valid {
            leases.release_all().await;
            return Err(error);
        }

        let router_mode = composition.target_chooser.router_mode();
        if router_mode.is_kv_routing() {
            self.observe_kv_selection("target", &target);
            self.observe_kv_selection("draft", &draft);
        }
        self.metrics
            .observe_selection_duration(selection_started.elapsed().as_secs_f64());
        tracing::info!(
            router_mode = router_mode.telemetry_label(),
            target_endpoint = %composition.target_endpoint,
            target_worker_id = target.worker.worker_id,
            target_dp_rank = target.worker.dp_rank,
            target_overlap_blocks = target.overlap_blocks,
            target_cached_tokens = target.cached_tokens,
            draft_endpoint = %composition.draft_endpoint,
            draft_worker_id = draft.worker.worker_id,
            draft_dp_rank = draft.worker.dp_rank,
            draft_overlap_blocks = draft.overlap_blocks,
            draft_cached_tokens = draft.cached_tokens,
            draft_incarnation_id = transport.draft_incarnation_id,
            protocol = %transport.protocol,
            "Selected external-speculation target/draft pair"
        );

        let hint = RouterHintEnvelope::speculative(SpeculativeDecodingRouterHintV1 {
            schema_version: SpeculativeDecodingRouterHintV1::SCHEMA_VERSION,
            draft_endpoint: composition.draft_endpoint.clone(),
            draft: draft.worker,
            draft_incarnation_id: transport.draft_incarnation_id,
            transport: transport.router_transport(),
        });
        if let Err(error) = hint.validate() {
            leases.release_all().await;
            return Err(anyhow::Error::msg(error));
        }

        let target_worker = target.worker;
        let (mut content, context) = request.into_parts();
        content.routing_mut().dp_rank = Some(target_worker.dp_rank);
        if let Err(error) = content.replace_router_hint(&hint) {
            leases.release_all().await;
            return Err(error.into());
        }
        let request = context.map(|_| content);

        leases.mark_dispatch_attempted();
        let mut dispatch_guard = DispatchGuard::new(leases, self.cleanup.clone());
        let dispatch_started = Instant::now();
        let dispatched = if router_mode.is_kv_routing() {
            self.inner
                .dispatch_kv_admitted_classified(request, target_worker.worker_id)
                .await
        } else {
            self.inner
                .dispatch_exact_classified(request, target_worker.worker_id)
                .await
        };
        let mut leases = dispatch_guard.take();
        self.metrics
            .observe_dispatch_duration(dispatch_started.elapsed().as_secs_f64());
        let stream = match dispatched {
            Ok(stream) => stream,
            Err(ExactDispatchError::NotSent(error)) => {
                self.metrics.observe_lifecycle("dispatch_not_sent");
                leases.release_all().await;
                return Err(error);
            }
            Err(ExactDispatchError::DeliveryUnknown(error)) => {
                self.metrics.observe_lifecycle("dispatch_error_quarantined");
                let timeout = leases.orphan_cleanup_timeout;
                self.cleanup.spawn_quarantine(leases, timeout);
                return Err(error);
            }
        };
        let paired = PairedResponseStream {
            inner: Some(stream),
            context: request_context.clone(),
            leases: Some(leases),
            cleanup: self.cleanup.clone(),
            pending_release: None,
        };
        Ok(ResponseStream::new(Box::pin(paired), request_context))
    }

    fn observe_kv_selection(&self, pool: &'static str, selection: &SpeculationSelection) {
        self.metrics
            .observe_selection(pool, selection.cached_tokens > 0);
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for ExternalSpeculationRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        self.route(request).await
    }
}

struct PairedResponseStream {
    inner: Option<ManyOut<Annotated<LLMEngineOutput>>>,
    context: Arc<dyn AsyncEngineContext>,
    leases: Option<PairLease>,
    cleanup: Arc<BackgroundCleanup>,
    pending_release: Option<PendingRelease>,
}

struct PendingRelease {
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
    output: Option<Annotated<LLMEngineOutput>>,
}

impl Stream for PairedResponseStream {
    type Item = Annotated<LLMEngineOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(pending) = self.pending_release.as_mut() {
            if pending.future.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            let pending = self
                .pending_release
                .take()
                .expect("completed release must remain installed");
            return Poll::Ready(pending.output);
        }

        let Some(inner) = self.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(mut item)) => {
                let cleanup_confirmed = observe_cleanup(&mut item);
                let normal_terminal = item
                    .data
                    .as_ref()
                    .and_then(|data| data.finish_reason.as_ref())
                    .is_some_and(|reason| {
                        !matches!(reason, FinishReason::Error(_) | FinishReason::Cancelled)
                    });
                if (normal_terminal || cleanup_confirmed)
                    && let Some(mut leases) = self.leases.take()
                {
                    self.cleanup.metrics.observe_lifecycle(if normal_terminal {
                        "successful_terminal"
                    } else {
                        "error_cleanup_confirmed"
                    });
                    self.pending_release = Some(PendingRelease {
                        future: Box::pin(async move { leases.release_all().await }),
                        output: Some(item),
                    });
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.inner.take();
                if let Some(mut leases) = self.leases.take() {
                    self.cleanup
                        .metrics
                        .observe_lifecycle("terminal_without_cleanup");
                    let timeout = leases.orphan_cleanup_timeout;
                    let draft = leases.take_draft_for_quarantine();
                    self.cleanup.spawn_quarantine(draft, timeout);
                    self.pending_release = Some(PendingRelease {
                        future: Box::pin(async move { leases.release_target().await }),
                        output: None,
                    });
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for PairedResponseStream {
    fn drop(&mut self) {
        if self.inner.is_none() || (self.leases.is_none() && self.pending_release.is_none()) {
            return;
        }
        self.context.stop_generating();
        self.cleanup
            .metrics
            .observe_lifecycle("response_stream_dropped");
        if let (Some(stream), Some(leases)) = (self.inner.take(), self.leases.take()) {
            self.cleanup.spawn_drain(stream, leases);
        }
    }
}

fn observe_cleanup(item: &mut Annotated<LLMEngineOutput>) -> bool {
    let Some(data) = item.data.as_mut() else {
        return false;
    };
    let Some(engine_data) = data
        .engine_data
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
    else {
        return false;
    };
    let Some(value) = engine_data.remove(EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY) else {
        return false;
    };
    if engine_data.is_empty() {
        data.engine_data = None;
    }
    let lifecycle =
        serde_json::from_value::<ExternalSpeculationLifecycleV1>(value).and_then(|lifecycle| {
            lifecycle
                .validate()
                .map(|()| lifecycle)
                .map_err(serde::de::Error::custom)
        });
    match lifecycle {
        Ok(ExternalSpeculationLifecycleV1 {
            draft_cleanup:
                DraftCleanupOutcomeV1::Acknowledged | DraftCleanupOutcomeV1::CleanupBoundElapsed,
            ..
        }) => true,
        Err(error) => {
            tracing::warn!(%error, "Ignored malformed external-speculation cleanup marker");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct RecordingChooser {
        mode: RouterMode,
        releases: Mutex<Vec<(String, WorkerWithDpRank)>>,
        release_started: AtomicUsize,
        release_failures_remaining: AtomicUsize,
        release_gate: Option<Arc<tokio::sync::Semaphore>>,
    }

    impl RecordingChooser {
        fn new(mode: RouterMode) -> Self {
            Self {
                mode,
                releases: Mutex::new(Vec::new()),
                release_started: AtomicUsize::new(0),
                release_failures_remaining: AtomicUsize::new(0),
                release_gate: None,
            }
        }

        fn with_release_gate(mode: RouterMode, gate: Arc<tokio::sync::Semaphore>) -> Self {
            Self {
                mode,
                releases: Mutex::new(Vec::new()),
                release_started: AtomicUsize::new(0),
                release_failures_remaining: AtomicUsize::new(0),
                release_gate: Some(gate),
            }
        }

        fn with_release_failures_and_gate(
            mode: RouterMode,
            failures: usize,
            gate: Arc<tokio::sync::Semaphore>,
        ) -> Self {
            Self {
                mode,
                releases: Mutex::new(Vec::new()),
                release_started: AtomicUsize::new(0),
                release_failures_remaining: AtomicUsize::new(failures),
                release_gate: Some(gate),
            }
        }
    }

    #[async_trait]
    impl SpeculationChooser for RecordingChooser {
        async fn select_and_reserve(
            &self,
            _request_id: &str,
            _request: &SingleIn<PreprocessedRequest>,
            _candidates: &std::collections::HashSet<WorkerWithDpRank>,
        ) -> anyhow::Result<SpeculationSelection> {
            unreachable!()
        }

        async fn release(
            &self,
            request_id: &str,
            worker: WorkerWithDpRank,
            _attempt: AdmissionAttempt,
        ) -> anyhow::Result<()> {
            self.release_started.fetch_add(1, Ordering::AcqRel);
            if self
                .release_failures_remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                anyhow::bail!("injected release failure");
            }
            if let Some(gate) = self.release_gate.as_ref() {
                gate.acquire().await.unwrap().forget();
            }
            self.releases
                .lock()
                .unwrap()
                .push((request_id.to_string(), worker));
            Ok(())
        }

        fn router_mode(&self) -> RouterMode {
            self.mode
        }
    }

    #[tokio::test]
    async fn dropped_selection_releases_the_exact_kv_booking() {
        let chooser = Arc::new(RecordingChooser::new(RouterMode::KV));
        let worker = WorkerWithDpRank::new(7, 2);
        let lease = SelectionLease::new(
            chooser.clone(),
            Arc::from("request-1"),
            worker,
            AdmissionAttempt::Untracked,
            None,
        );
        drop(lease);

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if !chooser.releases.lock().unwrap().is_empty() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            chooser.releases.lock().unwrap().as_slice(),
            &[("request-1".to_string(), worker)]
        );
    }

    #[tokio::test]
    async fn cancelled_explicit_release_retries_the_exact_kv_booking() {
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let chooser = Arc::new(RecordingChooser::with_release_gate(
            RouterMode::KV,
            gate.clone(),
        ));
        let worker = WorkerWithDpRank::new(7, 2);
        let lease = SelectionLease::new(
            chooser.clone(),
            Arc::from("request-1"),
            worker,
            AdmissionAttempt::Untracked,
            None,
        );
        let mut release = Box::pin(lease.release());

        assert!(futures::poll!(&mut release).is_pending());
        assert_eq!(chooser.release_started.load(Ordering::Acquire), 1);
        drop(release);

        tokio::time::timeout(Duration::from_secs(1), async {
            while chooser.release_started.load(Ordering::Acquire) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        gate.add_permits(1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while chooser.releases.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert_eq!(
            chooser.releases.lock().unwrap().as_slice(),
            &[("request-1".to_string(), worker)]
        );
    }

    #[test]
    fn dropped_policy_selection_releases_occupancy_without_kv_cleanup() {
        let chooser = Arc::new(RecordingChooser::new(RouterMode::LeastLoaded));
        let occupancy = Arc::new(dynamo_runtime::pipeline::RoutingOccupancyState::default());
        let worker = WorkerWithDpRank::new(7, 2);
        let reservation = occupancy.reserve(worker.worker_id);
        assert_eq!(occupancy.load(worker.worker_id), 1);

        let lease = SelectionLease::new(
            chooser.clone(),
            Arc::from("request-1"),
            worker,
            AdmissionAttempt::Untracked,
            Some(reservation),
        );
        drop(lease);

        assert_eq!(occupancy.load(worker.worker_id), 0);
        assert!(chooser.releases.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn dropped_dispatch_guard_quarantines_policy_pair_until_cleanup_bound() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = distributed
            .namespace("external-speculation-quarantine".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = ExternalSpeculationMetrics::from_component(&component);
        let cleanup = Arc::new(BackgroundCleanup::new(
            CancellationToken::new(),
            metrics.clone(),
        ));
        let chooser = Arc::new(RecordingChooser::new(RouterMode::LeastLoaded));
        let target_state = Arc::new(dynamo_runtime::pipeline::RoutingOccupancyState::default());
        let draft_state = Arc::new(dynamo_runtime::pipeline::RoutingOccupancyState::default());
        let target = WorkerWithDpRank::new(7, 0);
        let draft = WorkerWithDpRank::new(8, 0);
        let mut leases = PairLease::new(
            SelectionLease::new(
                chooser.clone(),
                Arc::from("request-1"),
                target,
                AdmissionAttempt::Untracked,
                Some(target_state.reserve(target.worker_id)),
            ),
            SelectionLease::new(
                chooser.clone(),
                Arc::from("request-1"),
                draft,
                AdmissionAttempt::Untracked,
                Some(draft_state.reserve(draft.worker_id)),
            ),
            Duration::from_millis(25),
            metrics,
        );
        leases.mark_dispatch_attempted();

        let guard = DispatchGuard::new(leases, cleanup.clone());
        drop(guard);
        assert_eq!(cleanup.active.load(Ordering::Acquire), 1);
        assert_eq!(target_state.load(target.worker_id), 1);
        assert_eq!(draft_state.load(draft.worker_id), 1);

        tokio::time::timeout(Duration::from_secs(1), async {
            while cleanup.active.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(target_state.load(target.worker_id), 0);
        assert_eq!(draft_state.load(draft.worker_id), 0);
        assert!(chooser.releases.lock().unwrap().is_empty());

        drop(cleanup);
        runtime.shutdown();
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_stream_releases_draft_at_bound_while_target_release_is_pending() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = distributed
            .namespace("external-speculation-drain-bound".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = ExternalSpeculationMetrics::from_component(&component);
        let cleanup = Arc::new(BackgroundCleanup::new(
            CancellationToken::new(),
            metrics.clone(),
        ));
        let target_gate = Arc::new(tokio::sync::Semaphore::new(0));
        let target_chooser = Arc::new(RecordingChooser::with_release_gate(
            RouterMode::KV,
            target_gate.clone(),
        ));
        let draft_chooser = Arc::new(RecordingChooser::new(RouterMode::KV));
        let target = WorkerWithDpRank::new(7, 0);
        let draft = WorkerWithDpRank::new(8, 0);
        let cleanup_bound = Duration::from_millis(25);
        let mut leases = PairLease::new(
            SelectionLease::new(
                target_chooser.clone(),
                Arc::from("request-1"),
                target,
                AdmissionAttempt::Untracked,
                None,
            ),
            SelectionLease::new(
                draft_chooser.clone(),
                Arc::from("request-1"),
                draft,
                AdmissionAttempt::Untracked,
                None,
            ),
            cleanup_bound,
            metrics,
        );
        leases.mark_dispatch_attempted();

        let response_context = dynamo_runtime::pipeline::Context::new(()).context();
        let inner = ResponseStream::new(
            Box::pin(futures::stream::empty::<Annotated<LLMEngineOutput>>()),
            response_context.clone(),
        );
        let paired = PairedResponseStream {
            inner: Some(inner),
            context: response_context,
            leases: Some(leases),
            cleanup: cleanup.clone(),
            pending_release: None,
        };

        drop(paired);
        assert_eq!(cleanup.active.load(Ordering::Acquire), 1);
        tokio::time::advance(cleanup_bound - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(target_chooser.release_started.load(Ordering::Acquire), 1);
        assert_eq!(draft_chooser.release_started.load(Ordering::Acquire), 0);

        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert!(target_chooser.releases.lock().unwrap().is_empty());
        assert_eq!(
            draft_chooser.releases.lock().unwrap().as_slice(),
            &[("request-1".to_string(), draft)]
        );

        target_gate.add_permits(1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while cleanup.active.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        drop(cleanup);
        runtime.shutdown();
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_stream_releases_draft_immediately_after_cleanup_confirmation() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = distributed
            .namespace("external-speculation-drain-confirmed".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = ExternalSpeculationMetrics::from_component(&component);
        let cleanup = Arc::new(BackgroundCleanup::new(
            CancellationToken::new(),
            metrics.clone(),
        ));
        let target_gate = Arc::new(tokio::sync::Semaphore::new(0));
        let target_chooser = Arc::new(RecordingChooser::with_release_gate(
            RouterMode::KV,
            target_gate.clone(),
        ));
        let draft_chooser = Arc::new(RecordingChooser::new(RouterMode::KV));
        let target = WorkerWithDpRank::new(7, 0);
        let draft = WorkerWithDpRank::new(8, 0);
        let mut leases = PairLease::new(
            SelectionLease::new(
                target_chooser.clone(),
                Arc::from("request-1"),
                target,
                AdmissionAttempt::Untracked,
                None,
            ),
            SelectionLease::new(
                draft_chooser.clone(),
                Arc::from("request-1"),
                draft,
                AdmissionAttempt::Untracked,
                None,
            ),
            Duration::from_secs(60),
            metrics,
        );
        leases.mark_dispatch_attempted();

        let output = LLMEngineOutput {
            engine_data: Some(serde_json::json!({
                EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY: {
                    "schema_version": 1,
                    "draft_cleanup": "acknowledged"
                }
            })),
            ..Default::default()
        };
        let response_context = dynamo_runtime::pipeline::Context::new(()).context();
        let inner = ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            response_context.clone(),
        );
        let paired = PairedResponseStream {
            inner: Some(inner),
            context: response_context,
            leases: Some(leases),
            cleanup: cleanup.clone(),
            pending_release: None,
        };
        let started = tokio::time::Instant::now();

        drop(paired);
        tokio::task::yield_now().await;
        assert_eq!(tokio::time::Instant::now(), started);
        assert_eq!(target_chooser.release_started.load(Ordering::Acquire), 1);
        assert!(target_chooser.releases.lock().unwrap().is_empty());
        assert_eq!(
            draft_chooser.releases.lock().unwrap().as_slice(),
            &[("request-1".to_string(), draft)]
        );

        target_gate.add_permits(1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while cleanup.active.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        drop(cleanup);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn cleanup_confirmed_item_waits_for_both_kv_releases() {
        use futures::StreamExt;

        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = distributed
            .namespace("external-speculation-release-order".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = ExternalSpeculationMetrics::from_component(&component);
        let cleanup = Arc::new(BackgroundCleanup::new(
            CancellationToken::new(),
            metrics.clone(),
        ));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let chooser = Arc::new(RecordingChooser::with_release_failures_and_gate(
            RouterMode::KV,
            2,
            gate.clone(),
        ));
        let target = WorkerWithDpRank::new(7, 0);
        let draft = WorkerWithDpRank::new(8, 0);
        let mut leases = PairLease::new(
            SelectionLease::new(
                chooser.clone(),
                Arc::from("request-1"),
                target,
                AdmissionAttempt::Untracked,
                None,
            ),
            SelectionLease::new(
                chooser.clone(),
                Arc::from("request-1"),
                draft,
                AdmissionAttempt::Untracked,
                None,
            ),
            Duration::from_millis(25),
            metrics,
        );
        leases.mark_dispatch_attempted();

        let output = LLMEngineOutput {
            finish_reason: Some(FinishReason::Error("target failed".to_string())),
            engine_data: Some(serde_json::json!({
                EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY: {
                    "schema_version": 1,
                    "draft_cleanup": "acknowledged"
                }
            })),
            ..Default::default()
        };
        let response_context = dynamo_runtime::pipeline::Context::new(()).context();
        let inner = ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            response_context.clone(),
        );
        let mut paired = PairedResponseStream {
            inner: Some(inner),
            context: response_context,
            leases: Some(leases),
            cleanup,
            pending_release: None,
        };
        let mut next = Box::pin(paired.next());

        assert!(futures::poll!(&mut next).is_pending());
        assert!(futures::poll!(&mut next).is_pending());
        assert_eq!(chooser.release_started.load(Ordering::Acquire), 2);
        assert!(chooser.releases.lock().unwrap().is_empty());

        tokio::time::sleep(RELEASE_RETRY_INITIAL_DELAY).await;
        assert!(futures::poll!(&mut next).is_pending());
        assert_eq!(chooser.release_started.load(Ordering::Acquire), 4);
        assert!(chooser.releases.lock().unwrap().is_empty());

        gate.add_permits(2);
        let item = next.await.unwrap();
        assert!(item.data.unwrap().engine_data.is_none());
        assert_eq!(chooser.releases.lock().unwrap().len(), 2);

        drop(paired);
        runtime.shutdown();
    }

    #[tokio::test(start_paused = true)]
    async fn eof_quarantines_draft_while_target_release_is_pending() {
        use futures::StreamExt;

        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let distributed = dynamo_runtime::DistributedRuntime::new(
            runtime.clone(),
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let component = distributed
            .namespace("external-speculation-eof-quarantine".to_string())
            .unwrap()
            .component("frontend".to_string())
            .unwrap();
        let metrics = ExternalSpeculationMetrics::from_component(&component);
        let cleanup = Arc::new(BackgroundCleanup::new(
            CancellationToken::new(),
            metrics.clone(),
        ));
        let target_gate = Arc::new(tokio::sync::Semaphore::new(0));
        let target_chooser = Arc::new(RecordingChooser::with_release_gate(
            RouterMode::KV,
            target_gate.clone(),
        ));
        let draft_chooser = Arc::new(RecordingChooser::new(RouterMode::KV));
        let target = WorkerWithDpRank::new(7, 0);
        let draft = WorkerWithDpRank::new(8, 0);
        let cleanup_bound = Duration::from_millis(25);
        let mut leases = PairLease::new(
            SelectionLease::new(
                target_chooser.clone(),
                Arc::from("request-1"),
                target,
                AdmissionAttempt::Untracked,
                None,
            ),
            SelectionLease::new(
                draft_chooser.clone(),
                Arc::from("request-1"),
                draft,
                AdmissionAttempt::Untracked,
                None,
            ),
            cleanup_bound,
            metrics,
        );
        leases.mark_dispatch_attempted();

        let response_context = dynamo_runtime::pipeline::Context::new(()).context();
        let inner = ResponseStream::new(
            Box::pin(futures::stream::empty::<Annotated<LLMEngineOutput>>()),
            response_context.clone(),
        );
        let mut paired = PairedResponseStream {
            inner: Some(inner),
            context: response_context,
            leases: Some(leases),
            cleanup: cleanup.clone(),
            pending_release: None,
        };
        let mut next = Box::pin(paired.next());

        assert!(futures::poll!(&mut next).is_pending());
        assert_eq!(cleanup.active.load(Ordering::Acquire), 1);
        assert!(futures::poll!(&mut next).is_pending());
        assert_eq!(target_chooser.release_started.load(Ordering::Acquire), 1);
        assert_eq!(draft_chooser.release_started.load(Ordering::Acquire), 0);

        drop(next);
        drop(paired);
        tokio::time::advance(cleanup_bound - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(cleanup.active.load(Ordering::Acquire), 1);
        assert_eq!(draft_chooser.release_started.load(Ordering::Acquire), 0);

        tokio::time::advance(Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(cleanup.active.load(Ordering::Acquire), 0);
        assert_eq!(
            draft_chooser.releases.lock().unwrap().as_slice(),
            &[("request-1".to_string(), draft)]
        );

        tokio::time::timeout(Duration::from_secs(1), async {
            while target_chooser.release_started.load(Ordering::Acquire) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        target_gate.add_permits(1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while target_chooser.releases.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        drop(cleanup);
        runtime.shutdown();
    }

    #[test]
    fn cleanup_marker_is_consumed_before_forwarding() {
        let mut output = LLMEngineOutput {
            engine_data: Some(serde_json::json!({
                EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY: {
                    "schema_version": 1,
                    "draft_cleanup": "acknowledged"
                }
            })),
            ..Default::default()
        };
        let mut item = Annotated::from_data(output.clone());
        assert!(observe_cleanup(&mut item));
        assert!(item.data.as_ref().unwrap().engine_data.is_none());

        output.engine_data = Some(serde_json::json!({
            EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY: {
                "schema_version": 9,
                "draft_cleanup": "acknowledged"
            }
        }));
        let mut item = Annotated::from_data(output);
        assert!(!observe_cleanup(&mut item));
    }
}
