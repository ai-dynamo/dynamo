// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native OpenTelemetry request-lifecycle spans.
//!
//! This initial registry emits only native span timing and causal parentage. It
//! intentionally does not attach request attributes, metrics, or detail data.

use tracing::Span;

use crate::config::environment_names::lifecycle_tracing::{
    DYN_DISAGGREGATION_MODE,
    DYN_LIFECYCLE_TRACE_ENABLED,
};

/// Static tracing target used exclusively by lifecycle spans.
pub const LIFECYCLE_TARGET: &str = "dynamo.request_lifecycle";

/// A duration-bearing boundary in the request-lifecycle convention.
///
/// `WorkerOperationPrefill` and `WorkerOperationDecode` are coarse Dynamo
/// runtime boundaries around the worker operation; they are not direct engine
/// execution measurements. Keep `engine.*` and `kv.transfer` for future spans
/// driven by authoritative engine or NIXL lifecycle signals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleStage {
    RequestLifecycle,
    RequestPreprocessing,
    RouterQueue,
    RouterSelection,
    WorkerAdmission,
    RequestDispatch,
    KvTransfer,
    EngineQueue,
    WorkerOperationPrefill,
    WorkerOperationDecode,
    ResponseStreaming,
    ResponseStreamingPrefill,
    ResponseStreamingDecode,
    ResponseStreamingWorker,
}

impl LifecycleStage {
    /// Stable OpenTelemetry span name for this stage.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RequestLifecycle => "request.lifecycle",
            Self::RequestPreprocessing => "request.preprocessing",
            Self::RouterQueue => "router.queue",
            Self::RouterSelection => "router.selection",
            Self::WorkerAdmission => "worker.admission",
            Self::RequestDispatch => "request.dispatch",
            Self::KvTransfer => "kv.transfer",
            Self::EngineQueue => "engine.queue",
            Self::WorkerOperationPrefill => "worker.operation.prefill",
            Self::WorkerOperationDecode => "worker.operation.decode",
            Self::ResponseStreaming => "response.streaming",
            Self::ResponseStreamingPrefill => "response.streaming.prefill",
            Self::ResponseStreamingDecode => "response.streaming.decode",
            Self::ResponseStreamingWorker => "response.streaming.worker",
        }
    }

    fn span(self) -> Span {
        // Each branch is a static callsite, allowing inexpensive target filtering.
        match self {
            Self::RequestLifecycle => tracing::info_span!(target: "dynamo.request_lifecycle", "request.lifecycle"),
            Self::RequestPreprocessing => tracing::info_span!(target: "dynamo.request_lifecycle", "request.preprocessing"),
            Self::RouterQueue => tracing::info_span!(target: "dynamo.request_lifecycle", "router.queue"),
            Self::RouterSelection => tracing::info_span!(target: "dynamo.request_lifecycle", "router.selection"),
            Self::WorkerAdmission => tracing::info_span!(target: "dynamo.request_lifecycle", "worker.admission"),
            Self::RequestDispatch => tracing::info_span!(target: "dynamo.request_lifecycle", "request.dispatch"),
            Self::KvTransfer => tracing::info_span!(target: "dynamo.request_lifecycle", "kv.transfer"),
            Self::EngineQueue => tracing::info_span!(target: "dynamo.request_lifecycle", "engine.queue"),
            Self::WorkerOperationPrefill => tracing::info_span!(target: "dynamo.request_lifecycle", "worker.operation.prefill"),
            Self::WorkerOperationDecode => tracing::info_span!(target: "dynamo.request_lifecycle", "worker.operation.decode"),
            Self::ResponseStreaming => tracing::info_span!(target: "dynamo.request_lifecycle", "response.streaming"),
            Self::ResponseStreamingPrefill => tracing::info_span!(target: "dynamo.request_lifecycle", "response.streaming.prefill"),
            Self::ResponseStreamingDecode => tracing::info_span!(target: "dynamo.request_lifecycle", "response.streaming.decode"),
            Self::ResponseStreamingWorker => tracing::info_span!(target: "dynamo.request_lifecycle", "response.streaming.worker"),
        }
    }
}

/// Request-scoped lifecycle capture state.
///
/// Construct this once when request state is created, freezing the feature gate
/// for the lifetime of that request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LifecycleTrace {
    enabled: bool,
}

impl LifecycleTrace {
    /// Capture the lifecycle feature gate for a newly-created request.
    pub fn from_environment() -> Self {
        Self::new(lifecycle_tracing_enabled())
    }

    /// Construct capture state explicitly, primarily for integrations and tests.
    pub const fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Whether lifecycle spans are emitted for this request.
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Start a duration-only lifecycle span.
    #[must_use]
    pub fn start(self, stage: LifecycleStage) -> Span {
        if self.enabled { stage.span() } else { Span::none() }
    }

    /// Start the worker response-streaming boundary with its configured
    /// disaggregation role encoded in the timing span name. This is deliberately
    /// fieldless during the timing-only rollout; the role will become the common
    /// `dynamo.operation.role` attribute when typed lifecycle fields are added.
    #[must_use]
    pub fn start_worker_response_streaming(self) -> Span {
        self.start(worker_response_streaming_stage())
    }

    /// Start the worker operation boundary for a disaggregated role.
    ///
    /// This bounds the Dynamo runtime's worker-side operation. It intentionally
    /// includes any backend-internal queueing or decode-side KV wait. Reserve
    /// `engine.*` names for future direct engine signals.
    #[must_use]
    pub fn start_worker_operation(self) -> Span {
        match worker_disaggregation_mode().as_deref() {
            Some("prefill") => self.start(LifecycleStage::WorkerOperationPrefill),
            Some("decode") => self.start(LifecycleStage::WorkerOperationDecode),
            _ => Span::none(),
        }
    }
}

fn worker_disaggregation_mode() -> Option<String> {
    std::env::var(DYN_DISAGGREGATION_MODE)
        .ok()
        .map(|mode| mode.trim().to_ascii_lowercase())
}

fn worker_response_streaming_stage() -> LifecycleStage {
    match worker_disaggregation_mode().as_deref() {
        Some("prefill") => LifecycleStage::ResponseStreamingPrefill,
        Some("decode") => LifecycleStage::ResponseStreamingDecode,
        _ => LifecycleStage::ResponseStreamingWorker,
    }
}

pub(crate) fn lifecycle_tracing_enabled() -> bool {
    std::env::var(DYN_LIFECYCLE_TRACE_ENABLED)
        .ok()
        .is_some_and(|value| matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::Subscriber;
    use tracing_subscriber::{Layer, layer::Context, prelude::*};

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    struct CapturedSpan {
        name: &'static str,
        target: &'static str,
        field_count: usize,
    }

    struct CaptureLayer(Arc<Mutex<Vec<CapturedSpan>>>);

    impl<S: Subscriber> Layer<S> for CaptureLayer {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::Id,
            _ctx: Context<'_, S>,
        ) {
            let metadata = attrs.metadata();
            self.0.lock().unwrap().push(CapturedSpan {
                name: metadata.name(),
                target: metadata.target(),
                field_count: metadata.fields().len(),
            });
        }
    }

    #[test]
    fn enabled_trace_creates_a_fieldless_registered_span() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let _span = LifecycleTrace::new(true).start(LifecycleStage::RouterQueue);

        assert_eq!(
            captured.lock().unwrap().as_slice(),
            [CapturedSpan {
                name: "router.queue",
                target: LIFECYCLE_TARGET,
                field_count: 0,
            }]
        );
    }

    #[test]
    fn disabled_trace_is_a_noop() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let _span = LifecycleTrace::new(false).start(LifecycleStage::RouterQueue);

        assert!(captured.lock().unwrap().is_empty());
    }

    #[test]
    fn response_streaming_stage_names_distinguish_owners() {
        assert_eq!(
            LifecycleStage::WorkerOperationPrefill.name(),
            "worker.operation.prefill"
        );
        assert_eq!(
            LifecycleStage::WorkerOperationDecode.name(),
            "worker.operation.decode"
        );
        assert_eq!(LifecycleStage::ResponseStreaming.name(), "response.streaming");
        assert_eq!(
            LifecycleStage::ResponseStreamingPrefill.name(),
            "response.streaming.prefill"
        );
        assert_eq!(
            LifecycleStage::ResponseStreamingDecode.name(),
            "response.streaming.decode"
        );
        assert_eq!(
            LifecycleStage::ResponseStreamingWorker.name(),
            "response.streaming.worker"
        );
    }
}
