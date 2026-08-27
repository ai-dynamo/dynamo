// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native OpenTelemetry request-lifecycle spans.
//!
//! This initial registry emits only native span timing and causal parentage. It
//! intentionally does not attach request attributes, metrics, or detail data.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use tracing::Span;

use crate::config::environment_names::lifecycle_tracing::{
    DYN_DISAGGREGATION_MODE, DYN_LIFECYCLE_TRACE_ENABLED, DYN_LIFECYCLE_TRACE_MODE,
    DYN_LIFECYCLE_TRACE_PROFILE,
};

/// Static tracing target used exclusively by lifecycle spans.
pub const LIFECYCLE_TARGET: &str = "dynamo.request_lifecycle";

/// Context-registry key used to preserve lifecycle identity through frontend stages.
pub const LIFECYCLE_TRACE_CONTEXT_KEY: &str = "dynamo.request_lifecycle.trace";

const LIFECYCLE_SCHEMA: &str = "v1";
const DEFAULT_PROFILE: &str = "generic.v1";
const DEFAULT_MODE: &str = "core";

static PROCESS_EPOCH: OnceLock<String> = OnceLock::new();
static INSTANCE_ID: OnceLock<String> = OnceLock::new();

/// Operation owner used to distinguish the frontend and P/D worker waves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleOperationRole {
    Frontend,
    Prefill,
    Decode,
    Worker,
}

impl LifecycleOperationRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Frontend => "frontend",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Worker => "worker",
        }
    }
}

/// One-shot outcome recorded on the request root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalOutcome {
    Success,
    Rejected,
    Cancelled,
    TimedOut,
    Failed,
    Unknown,
}

impl TerminalOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
        }
    }

    const fn is_error(self) -> bool {
        !matches!(self, Self::Success)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LifecycleIdentity {
    request_id: String,
    /// Identifies one lifecycle wave within a request.
    ///
    /// `request_id` groups every prefill, decode, retry, and migration for an
    /// end-user request. `operation_id` identifies one of those waves. When
    /// cross-operation propagation is added, a successor (for example, decode)
    /// records its producing operation (for example, prefill) as
    /// `dynamo.operation.parent_id` and carries an OTel link. That causal
    /// relation cannot be inferred reliably from `request_id` alone.
    operation_id: String,
    role: LifecycleOperationRole,
    profile: String,
    mode: String,
    identity_state: &'static str,
}

impl LifecycleIdentity {
    fn new(request_id: Option<String>, role: LifecycleOperationRole) -> Self {
        let (request_id, identity_state) = match request_id.filter(|id| !id.is_empty()) {
            Some(id) => (id, "complete"),
            None => ("unknown".to_string(), "missing_request_id"),
        };
        Self {
            request_id,
            operation_id: uuid::Uuid::new_v4().to_string(),
            role,
            profile: lifecycle_profile(),
            mode: lifecycle_mode(),
            identity_state,
        }
    }
}

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

    const fn component(self) -> &'static str {
        match self {
            Self::RequestLifecycle | Self::RequestPreprocessing | Self::ResponseStreaming => "frontend",
            Self::RouterQueue | Self::RouterSelection => "router",
            Self::WorkerAdmission | Self::RequestDispatch | Self::WorkerOperationPrefill
            | Self::WorkerOperationDecode | Self::ResponseStreamingPrefill
            | Self::ResponseStreamingDecode | Self::ResponseStreamingWorker => "worker",
            Self::KvTransfer => "kv_transfer",
            Self::EngineQueue => "engine",
        }
    }

    fn span(self, identity: &LifecycleIdentity) -> Span {
        macro_rules! common_span {
            ($name:literal) => {
                tracing::info_span!(
                    target: "dynamo.request_lifecycle", $name,
                    "dynamo.request.id" = %identity.request_id,
                    "dynamo.request.attempt" = 0_u64,
                    "dynamo.operation.id" = %identity.operation_id,
                    "dynamo.operation.role" = identity.role.as_str(),
                    "dynamo.lifecycle.schema" = LIFECYCLE_SCHEMA,
                    "dynamo.lifecycle.profile" = %identity.profile,
                    "dynamo.lifecycle.mode" = %identity.mode,
                    "dynamo.component" = self.component(),
                    "dynamo.instance.id" = instance_id(),
                    "dynamo.process.epoch" = process_epoch(),
                    "dynamo.lifecycle.identity.state" = identity.identity_state,
                    "dynamo.lifecycle.capture.state" = "recorded",
                )
            };
        }
        // Each branch is a static callsite, allowing inexpensive target filtering.
        match self {
            Self::RequestLifecycle => tracing::info_span!(
                target: "dynamo.request_lifecycle", "request.lifecycle",
                "dynamo.request.id" = %identity.request_id,
                "dynamo.request.attempt" = 0_u64,
                "dynamo.operation.id" = %identity.operation_id,
                "dynamo.operation.role" = identity.role.as_str(),
                "dynamo.lifecycle.schema" = LIFECYCLE_SCHEMA,
                "dynamo.lifecycle.profile" = %identity.profile,
                "dynamo.lifecycle.mode" = %identity.mode,
                "dynamo.component" = self.component(),
                "dynamo.instance.id" = instance_id(),
                "dynamo.process.epoch" = process_epoch(),
                "dynamo.lifecycle.identity.state" = identity.identity_state,
                "dynamo.lifecycle.capture.state" = "recorded",
                "dynamo.session.id" = tracing::field::Empty,
                "dynamo.session.source" = tracing::field::Empty,
                "dynamo.request.terminal.outcome" = tracing::field::Empty,
                "dynamo.request.terminal.error" = tracing::field::Empty,
            ),
            Self::RequestPreprocessing => common_span!("request.preprocessing"),
            Self::RouterQueue => common_span!("router.queue"),
            Self::RouterSelection => common_span!("router.selection"),
            Self::WorkerAdmission => common_span!("worker.admission"),
            Self::RequestDispatch => common_span!("request.dispatch"),
            Self::KvTransfer => common_span!("kv.transfer"),
            Self::EngineQueue => common_span!("engine.queue"),
            Self::WorkerOperationPrefill => common_span!("worker.operation.prefill"),
            Self::WorkerOperationDecode => common_span!("worker.operation.decode"),
            Self::ResponseStreaming => common_span!("response.streaming"),
            Self::ResponseStreamingPrefill => common_span!("response.streaming.prefill"),
            Self::ResponseStreamingDecode => common_span!("response.streaming.decode"),
            Self::ResponseStreamingWorker => common_span!("response.streaming.worker"),
        }
    }
}

/// Request-scoped lifecycle capture state.
///
/// Construct this once when request state is created, freezing the feature gate
/// for the lifetime of that request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleTrace {
    enabled: bool,
    identity: LifecycleIdentity,
    session: Option<(String, &'static str)>,
}

impl LifecycleTrace {
    /// Capture the lifecycle feature gate for a newly-created request.
    pub fn from_environment() -> Self {
        Self::new(lifecycle_tracing_enabled())
    }

    /// Construct capture state explicitly, primarily for integrations and tests.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            identity: LifecycleIdentity::new(None, LifecycleOperationRole::Worker),
            session: None,
        }
    }

    /// Construct worker capture state using the request ID propagated at ingress.
    pub fn from_request_id(request_id: impl Into<String>) -> Self {
        Self::with_role(request_id, worker_operation_role())
    }

    /// Construct router capture state. M3 will propagate explicit P/D operation links.
    pub fn router_request(request_id: impl Into<String>) -> Self {
        Self::with_role(request_id, LifecycleOperationRole::Frontend)
    }

    /// Construct frontend capture state and root-only session identity.
    pub fn frontend_request(request_id: impl Into<String>, session_id: Option<String>) -> Self {
        let request_id = request_id.into();
        let session = match session_id.filter(|id| !id.is_empty()) {
            Some(id) => (id, "agent_context"),
            None => (request_id.clone(), "request_id_fallback"),
        };
        Self {
            enabled: lifecycle_tracing_enabled(),
            identity: LifecycleIdentity::new(Some(request_id), LifecycleOperationRole::Frontend),
            session: Some(session),
        }
    }

    fn with_role(request_id: impl Into<String>, role: LifecycleOperationRole) -> Self {
        Self {
            enabled: lifecycle_tracing_enabled(),
            identity: LifecycleIdentity::new(Some(request_id.into()), role),
            session: None,
        }
    }

    /// Whether lifecycle spans are emitted for this request.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start the request root and return a recorder shared with all terminal paths.
    #[must_use]
    pub fn start_request(&self) -> LifecycleRequest {
        let span = self.start(LifecycleStage::RequestLifecycle);
        if let Some((session_id, source)) = &self.session {
            span.record("dynamo.session.id", session_id.as_str());
            span.record("dynamo.session.source", *source);
        }
        LifecycleRequest {
            span: span.clone(),
            terminal: LifecycleTerminal(Arc::new(TerminalState {
                enabled: self.enabled,
                span,
                finished: AtomicBool::new(false),
            })),
        }
    }

    /// Start a duration-only lifecycle span.
    #[must_use]
    pub fn start(&self, stage: LifecycleStage) -> Span {
        if self.enabled {
            stage.span(&self.identity)
        } else {
            Span::none()
        }
    }

    /// Start the worker response-streaming boundary with its configured
    /// disaggregation role encoded in the timing span name. This is deliberately
    /// fieldless during the timing-only rollout; the role will become the common
    /// `dynamo.operation.role` attribute when typed lifecycle fields are added.
    #[must_use]
    pub fn start_worker_response_streaming(&self) -> Span {
        self.start(worker_response_streaming_stage())
    }

    /// Start the worker operation boundary for a disaggregated role.
    ///
    /// This bounds the Dynamo runtime's worker-side operation. It intentionally
    /// includes any backend-internal queueing or decode-side KV wait. Reserve
    /// `engine.*` names for future direct engine signals.
    #[must_use]
    pub fn start_worker_operation(&self) -> Span {
        match worker_disaggregation_mode().as_deref() {
            Some("prefill") => self.start(LifecycleStage::WorkerOperationPrefill),
            Some("decode") => self.start(LifecycleStage::WorkerOperationDecode),
            _ => Span::none(),
        }
    }
}

/// Request root span plus the shared terminal recorder.
pub struct LifecycleRequest {
    span: Span,
    terminal: LifecycleTerminal,
}

impl LifecycleRequest {
    #[must_use]
    pub fn span(&self) -> Span {
        self.span.clone()
    }

    #[must_use]
    pub fn terminal(&self) -> LifecycleTerminal {
        self.terminal.clone()
    }
}

/// A terminal recorder that is safe to clone across completion and cancellation paths.
#[derive(Clone)]
pub struct LifecycleTerminal(Arc<TerminalState>);

struct TerminalState {
    enabled: bool,
    span: Span,
    finished: AtomicBool,
}

impl LifecycleTerminal {
    /// Record the first observed terminal result. Later races are ignored.
    pub fn finish(&self, outcome: TerminalOutcome) {
        if self.0.enabled && !self.0.finished.swap(true, Ordering::AcqRel) {
            self.0
                .span
                .record("dynamo.request.terminal.outcome", outcome.as_str());
            self.0
                .span
                .record("dynamo.request.terminal.error", outcome.is_error());
        }
    }
}

impl Drop for TerminalState {
    fn drop(&mut self) {
        if self.enabled && !self.finished.swap(true, Ordering::AcqRel) {
            self.span
                .record("dynamo.request.terminal.outcome", TerminalOutcome::Unknown.as_str());
            self.span.record("dynamo.request.terminal.error", true);
        }
    }
}

fn lifecycle_profile() -> String {
    std::env::var(DYN_LIFECYCLE_TRACE_PROFILE)
        .ok()
        .filter(|profile| !profile.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PROFILE.to_string())
}

fn lifecycle_mode() -> String {
    match std::env::var(DYN_LIFECYCLE_TRACE_MODE) {
        Ok(mode) if mode.trim().eq_ignore_ascii_case("investigation") => "investigation".to_string(),
        _ => DEFAULT_MODE.to_string(),
    }
}

fn process_epoch() -> &'static str {
    PROCESS_EPOCH
        .get_or_init(|| format!("{}:{}", std::process::id(), uuid::Uuid::new_v4()))
        .as_str()
}

fn instance_id() -> &'static str {
    INSTANCE_ID
        .get_or_init(|| {
            std::env::var("HOSTNAME")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| format!("pid-{}", std::process::id()))
        })
        .as_str()
}

fn worker_disaggregation_mode() -> Option<String> {
    std::env::var(DYN_DISAGGREGATION_MODE)
        .ok()
        .map(|mode| mode.trim().to_ascii_lowercase())
}

fn worker_operation_role() -> LifecycleOperationRole {
    match worker_disaggregation_mode().as_deref() {
        Some("prefill") => LifecycleOperationRole::Prefill,
        Some("decode") => LifecycleOperationRole::Decode,
        _ => LifecycleOperationRole::Worker,
    }
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
