// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native OpenTelemetry request-lifecycle spans.
//!
//! Coarse runtime timing, causal parentage, request identity, and terminal outcomes.
//! Engine-internal metrics and invariants are intentionally not instrumented here.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use std::time::{SystemTime, UNIX_EPOCH};
use tracing::Span;

use crate::config::environment_names::lifecycle_tracing::{
    DYN_LIFECYCLE_TRACE_ENABLED, DYN_LIFECYCLE_TRACE_MODE,
};

/// Static tracing target used exclusively by lifecycle spans.
pub const LIFECYCLE_TARGET: &str = "dynamo.request_lifecycle";

/// Context-registry key used to preserve lifecycle identity through frontend stages.
pub const LIFECYCLE_TRACE_CONTEXT_KEY: &str = "dynamo.request_lifecycle.trace";

/// Internal wire metadata marking requests with a frontend lifecycle root.
/// Absent on legacy or uninstrumented frontends; those requests keep ordinary tracing.
pub const LIFECYCLE_ROOT_METADATA_KEY: &str = "dynamo.lifecycle.root";

const LIFECYCLE_SCHEMA: &str = "v1";
const DEFAULT_PROFILE: &str = "generic.v1";
const DEFAULT_MODE: &str = "core";

static PROCESS_EPOCH: OnceLock<String> = OnceLock::new();
static INSTANCE_ID: OnceLock<String> = OnceLock::new();
static LIFECYCLE_ENABLED: OnceLock<bool> = OnceLock::new();
static LIFECYCLE_MODE: OnceLock<&'static str> = OnceLock::new();

/// Operation owner used to distinguish the frontend and worker stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleOperationRole {
    Frontend,
    Encode,
    Prefill,
    Decode,
    Worker,
}

impl LifecycleOperationRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Frontend => "frontend",
            Self::Encode => "encode",
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
    /// Identifies one lifecycle wave within a request, which may span retries
    /// or prefill/decode operations.
    operation_id: String,
    role: LifecycleOperationRole,
    profile: &'static str,
    mode: &'static str,
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
            profile: DEFAULT_PROFILE,
            mode: lifecycle_mode(),
            identity_state,
        }
    }
}

/// A duration-bearing boundary in the request-lifecycle convention.
///
/// `WorkerOperationPrefill` and `WorkerOperationDecode` are coarse Dynamo
/// runtime boundaries around the worker operation; they are not direct engine
/// execution measurements.
///
/// Router queue and selection stages are intentionally deferred. A later
/// instrumentation milestone will add spans at the actual scheduling
/// boundaries together with the router-specific metrics and invariants needed
/// to interpret them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleStage {
    RequestLifecycle,
    RequestPreprocessing,
    WorkerAdmission,
    RequestDispatch,
    WorkerOperation,
    WorkerOperationEncode,
    WorkerOperationPrefill,
    WorkerOperationDecode,
    ResponseStreaming,
    ResponseStreamingEncode,
    ResponseStreamingPrefill,
    ResponseStreamingDecode,
    ResponseStreamingWorker,
}

impl LifecycleStage {
    const fn component(self) -> &'static str {
        match self {
            Self::RequestLifecycle | Self::RequestPreprocessing | Self::ResponseStreaming => {
                "frontend"
            }
            Self::WorkerAdmission
            | Self::RequestDispatch
            | Self::WorkerOperation
            | Self::WorkerOperationEncode
            | Self::WorkerOperationPrefill
            | Self::WorkerOperationDecode
            | Self::ResponseStreamingPrefill
            | Self::ResponseStreamingEncode
            | Self::ResponseStreamingDecode
            | Self::ResponseStreamingWorker => "worker",
        }
    }

    fn span(self, identity: &LifecycleIdentity) -> Span {
        macro_rules! common_span {
            ($name:literal) => {
                tracing::info_span!(
                    target: LIFECYCLE_TARGET, $name,
                    "dynamo.request.id" = %identity.request_id,
                    "dynamo.operation.id" = %identity.operation_id,
                    "dynamo.operation.role" = identity.role.as_str(),
                    "dynamo.lifecycle.schema" = LIFECYCLE_SCHEMA,
                    "dynamo.lifecycle.profile" = %identity.profile,
                    "dynamo.lifecycle.mode" = %identity.mode,
                    "dynamo.component" = self.component(),
                    "dynamo.instance.id" = instance_id(),
                    "dynamo.process.epoch" = process_epoch(),
                    "dynamo.lifecycle.identity.state" = identity.identity_state,
                )
            };
        }
        // Each branch is a static callsite, allowing inexpensive target filtering.
        match self {
            Self::RequestLifecycle => tracing::info_span!(
                target: LIFECYCLE_TARGET, "request.lifecycle",
                "dynamo.request.id" = %identity.request_id,
                "dynamo.operation.id" = %identity.operation_id,
                "dynamo.operation.role" = identity.role.as_str(),
                "dynamo.lifecycle.schema" = LIFECYCLE_SCHEMA,
                "dynamo.lifecycle.profile" = %identity.profile,
                "dynamo.lifecycle.mode" = %identity.mode,
                "dynamo.component" = self.component(),
                "dynamo.instance.id" = instance_id(),
                "dynamo.process.epoch" = process_epoch(),
                "dynamo.lifecycle.identity.state" = identity.identity_state,
                "dynamo.session.id" = tracing::field::Empty,
                "dynamo.session.source" = tracing::field::Empty,
                "dynamo.request.terminal.outcome" = tracing::field::Empty,
                "dynamo.request.terminal.error" = tracing::field::Empty,
                "dynamo.request.terminal.timestamp_unix_ns" = tracing::field::Empty,
            ),
            Self::RequestPreprocessing => common_span!("request.preprocessing"),
            Self::WorkerAdmission => common_span!("worker.admission"),
            Self::RequestDispatch => common_span!("request.dispatch"),
            Self::WorkerOperation => common_span!("worker.operation"),
            Self::WorkerOperationEncode => common_span!("worker.operation.encode"),
            Self::WorkerOperationPrefill => common_span!("worker.operation.prefill"),
            Self::WorkerOperationDecode => common_span!("worker.operation.decode"),
            Self::ResponseStreaming => common_span!("response.streaming"),
            Self::ResponseStreamingEncode => common_span!("response.streaming.encode"),
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
    identity: Option<LifecycleIdentity>,
}

impl LifecycleTrace {
    /// Construct capture state explicitly, primarily for integrations and tests.
    pub fn new(enabled: bool) -> Self {
        if enabled {
            Self::enabled(LifecycleIdentity::new(None, LifecycleOperationRole::Worker))
        } else {
            Self::disabled()
        }
    }

    /// Construct worker capture state with its statically configured role.
    pub fn from_request_id_with_role(
        request_id: impl Into<String>,
        role: LifecycleOperationRole,
    ) -> Self {
        if !lifecycle_tracing_enabled() {
            return Self::disabled();
        }
        Self::enabled(LifecycleIdentity::new(Some(request_id.into()), role))
    }

    /// Construct a frontend trace before request parsing has made a session ID available.
    pub fn frontend_request_without_session(request_id: impl Into<String>) -> Self {
        Self::with_role(request_id, LifecycleOperationRole::Frontend)
    }

    fn with_role(request_id: impl Into<String>, role: LifecycleOperationRole) -> Self {
        if lifecycle_tracing_enabled() {
            Self::enabled(LifecycleIdentity::new(Some(request_id.into()), role))
        } else {
            Self::disabled()
        }
    }

    fn enabled(identity: LifecycleIdentity) -> Self {
        Self {
            enabled: true,
            identity: Some(identity),
        }
    }

    const fn disabled() -> Self {
        Self {
            enabled: false,
            identity: None,
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
        LifecycleRequest {
            span: span.clone(),
            terminal: LifecycleTerminal(self.enabled.then(|| {
                Arc::new(TerminalState {
                    span,
                    finished: AtomicBool::new(false),
                })
            })),
        }
    }

    /// Start a duration-only lifecycle span.
    #[must_use]
    pub fn start(&self, stage: LifecycleStage) -> Span {
        if let Some(identity) = self.identity.as_ref() {
            stage.span(identity)
        } else {
            Span::none()
        }
    }

    /// Start the worker response-streaming boundary with its configured
    /// disaggregation role encoded in the timing span name.
    #[must_use]
    pub fn start_worker_response_streaming(&self) -> Span {
        if !self.enabled {
            return Span::none();
        }
        let role = self.identity.as_ref().map(|identity| identity.role);
        self.start(worker_response_streaming_stage(role))
    }

    /// Start the worker operation boundary for a disaggregated role.
    ///
    /// This bounds the Dynamo runtime's worker-side operation. It intentionally
    /// includes any backend-internal queueing or decode-side KV wait.
    #[must_use]
    pub fn start_worker_operation(&self) -> Span {
        if !self.enabled {
            return Span::none();
        }
        let role = self.identity.as_ref().map(|identity| identity.role);
        self.start(worker_operation_stage(role))
    }
}

/// Request root span plus the shared terminal recorder.
///
/// The span stays open while detached request work cleans up. Its duration may
/// therefore extend past client termination; `dynamo.request.terminal.timestamp_unix_ns`
/// records the first terminal observation, independently of that cleanup.
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

    /// Record session identity after request parsing, or the request-ID fallback on early errors.
    pub fn record_session(&self, request_id: &str, session_id: Option<&str>) {
        let (session_id, source) = session_id
            .filter(|id| !id.is_empty())
            .map(|id| (id, "agent_context"))
            .unwrap_or((request_id, "request_id_fallback"));
        self.span.record("dynamo.session.id", session_id);
        self.span.record("dynamo.session.source", source);
    }
}

/// A terminal recorder that is safe to clone across completion and cancellation paths.
#[derive(Clone)]
pub struct LifecycleTerminal(Option<Arc<TerminalState>>);

struct TerminalState {
    span: Span,
    finished: AtomicBool,
}

impl LifecycleTerminal {
    /// Record the first observed terminal result. Later races are ignored.
    pub fn finish(&self, outcome: TerminalOutcome) {
        if let Some(state) = &self.0
            && !state.finished.swap(true, Ordering::AcqRel)
        {
            state.record(outcome);
        }
    }
}

impl TerminalState {
    fn record(&self, outcome: TerminalOutcome) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        self.span.record(
            "dynamo.request.terminal.timestamp_unix_ns",
            u64::try_from(timestamp.as_nanos()).unwrap_or(u64::MAX),
        );
        self.span
            .record("dynamo.request.terminal.outcome", outcome.as_str());
        self.span
            .record("dynamo.request.terminal.error", outcome.is_error());
    }
}

impl Drop for TerminalState {
    fn drop(&mut self) {
        if !self.finished.swap(true, Ordering::AcqRel) {
            self.record(TerminalOutcome::Unknown);
        }
    }
}

fn lifecycle_mode() -> &'static str {
    LIFECYCLE_MODE.get_or_init(|| match std::env::var(DYN_LIFECYCLE_TRACE_MODE) {
        Ok(mode) if mode.trim().eq_ignore_ascii_case("investigation") => "investigation",
        _ => DEFAULT_MODE,
    })
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

fn worker_operation_stage(role: Option<LifecycleOperationRole>) -> LifecycleStage {
    match role {
        Some(LifecycleOperationRole::Encode) => LifecycleStage::WorkerOperationEncode,
        Some(LifecycleOperationRole::Prefill) => LifecycleStage::WorkerOperationPrefill,
        Some(LifecycleOperationRole::Decode) => LifecycleStage::WorkerOperationDecode,
        _ => LifecycleStage::WorkerOperation,
    }
}

fn worker_response_streaming_stage(role: Option<LifecycleOperationRole>) -> LifecycleStage {
    match role {
        Some(LifecycleOperationRole::Encode) => LifecycleStage::ResponseStreamingEncode,
        Some(LifecycleOperationRole::Prefill) => LifecycleStage::ResponseStreamingPrefill,
        Some(LifecycleOperationRole::Decode) => LifecycleStage::ResponseStreamingDecode,
        _ => LifecycleStage::ResponseStreamingWorker,
    }
}

pub(crate) fn lifecycle_tracing_enabled() -> bool {
    *LIFECYCLE_ENABLED.get_or_init(|| crate::config::env_is_truthy(DYN_LIFECYCLE_TRACE_ENABLED))
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
        role: Option<String>,
    }

    #[derive(Default)]
    struct RoleCapture(Option<String>);

    impl tracing::field::Visit for RoleCapture {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            if field.name() == "dynamo.operation.role" {
                self.0 = Some(value.to_owned());
            }
        }

        fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
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
            let mut role = RoleCapture::default();
            attrs.record(&mut role);
            self.0.lock().unwrap().push(CapturedSpan {
                name: metadata.name(),
                target: metadata.target(),
                role: role.0,
            });
        }
    }

    #[test]
    fn enabled_trace_creates_a_registered_span() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let _span = LifecycleTrace::new(true).start(LifecycleStage::RequestPreprocessing);

        assert_eq!(
            captured.lock().unwrap().as_slice(),
            [CapturedSpan {
                name: "request.preprocessing",
                target: LIFECYCLE_TARGET,
                role: Some("worker".to_string()),
            }]
        );
    }

    #[test]
    fn disabled_trace_is_a_noop() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let trace = LifecycleTrace::new(false);
        assert!(trace.identity.is_none());
        let request = trace.start_request();
        assert!(
            request.terminal.0.is_none(),
            "disabled requests must not allocate terminal state"
        );
        let terminal = request.terminal();
        assert!(terminal.0.is_none());
        terminal.finish(TerminalOutcome::Success);
        request.record_session("request-id", None);
        let _span = trace.start(LifecycleStage::RequestPreprocessing);

        assert!(captured.lock().unwrap().is_empty());
    }

    #[test]
    fn worker_stage_selection_emits_expected_spans() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        for role in [
            LifecycleOperationRole::Worker,
            LifecycleOperationRole::Encode,
            LifecycleOperationRole::Prefill,
            LifecycleOperationRole::Decode,
        ] {
            let trace = LifecycleTrace::enabled(LifecycleIdentity::new(
                Some("request-id".to_string()),
                role,
            ));
            let _operation = trace.start_worker_operation();
            let _streaming = trace.start_worker_response_streaming();
        }

        let captured = captured.lock().unwrap();
        assert!(captured.iter().all(|span| span.target == LIFECYCLE_TARGET));
        assert_eq!(
            captured
                .iter()
                .map(|span| span.role.as_deref())
                .collect::<Vec<_>>(),
            [
                Some("worker"),
                Some("worker"),
                Some("encode"),
                Some("encode"),
                Some("prefill"),
                Some("prefill"),
                Some("decode"),
                Some("decode")
            ]
        );
        assert_eq!(
            captured.iter().map(|span| span.name).collect::<Vec<_>>(),
            [
                "worker.operation",
                "response.streaming.worker",
                "worker.operation.encode",
                "response.streaming.encode",
                "worker.operation.prefill",
                "response.streaming.prefill",
                "worker.operation.decode",
                "response.streaming.decode",
            ]
        );
    }
}
