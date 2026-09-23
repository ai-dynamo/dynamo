// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native OpenTelemetry request-lifecycle spans.
//!
//! Coarse runtime timing, causal parentage, request identity, and terminal outcomes.
//! Engine-internal metrics and invariants are intentionally not instrumented here.
//! Lifecycle attributes are deliberately bounded: core mode records stable identifiers
//! and decision summaries, while investigation mode may add bounded detail.

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
/// Router queue and selection stages describe scheduling boundaries and carry
/// router-specific evidence for interpreting those decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleStage {
    RequestLifecycle,
    RequestPreprocessing,
    RouterQueue,
    RouterSelection,
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
            Self::RequestLifecycle
            | Self::RequestPreprocessing
            | Self::RequestDispatch
            | Self::ResponseStreaming => "frontend",
            Self::RouterQueue | Self::RouterSelection => "router",
            Self::WorkerAdmission
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
            ($name:literal $(, $field:literal = $value:expr)* $(,)?) => {
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
                    $($field = $value,)*
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
            Self::RequestPreprocessing => common_span!(
                "request.preprocessing",
                "dynamo.stage.outcome" = tracing::field::Empty,
                "dynamo.stage.checkpoint" = tracing::field::Empty,
            ),
            Self::RouterQueue => common_span!(
                "router.queue",
                "dynamo.request.attempt" = 0_u64,
                "dynamo.lifecycle.capture.state" = "recorded",
                "dynamo.lifecycle.detail_schema" = "router_queue.v1",
                "dynamo.router.queue.class" = tracing::field::Empty,
                "dynamo.router.queue.policy" = tracing::field::Empty,
                "dynamo.router.queue.depth.in" = tracing::field::Empty,
                "dynamo.router.queue.depth.out" = tracing::field::Empty,
                "dynamo.router.queue.deferred" = tracing::field::Empty,
                "dynamo.router.queue.outcome" = tracing::field::Empty,
                "dynamo.router.queue.reason" = tracing::field::Empty,
            ),
            Self::RouterSelection => common_span!(
                "router.selection",
                "dynamo.request.attempt" = 0_u64,
                "dynamo.lifecycle.capture.state" = "recorded",
                "dynamo.lifecycle.detail_schema" = "router_selection.v1",
                "dynamo.router.candidate.count" = tracing::field::Empty,
                "dynamo.router.candidate.eligible.count" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.count" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.not_allowed" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.constraints" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.overloaded" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.unavailable" = tracing::field::Empty,
                "dynamo.router.candidate.filtered.policy" = tracing::field::Empty,
                "dynamo.router.algorithm.id" = tracing::field::Empty,
                "dynamo.router.algorithm.version" = tracing::field::Empty,
                "dynamo.router.decision.schema" = tracing::field::Empty,
                "dynamo.router.selection.policy" = tracing::field::Empty,
                "dynamo.router.pool.role" = tracing::field::Empty,
                "dynamo.router.selected.worker.id" = tracing::field::Empty,
                "dynamo.router.selected.dp.rank" = tracing::field::Empty,
                "dynamo.router.selected.score" = tracing::field::Empty,
                "dynamo.router.best.worker.id" = tracing::field::Empty,
                "dynamo.router.best.dp.rank" = tracing::field::Empty,
                "dynamo.router.best.score" = tracing::field::Empty,
                "dynamo.router.best.margin" = tracing::field::Empty,
                "dynamo.router.candidates.detail_schema" = tracing::field::Empty,
                "dynamo.router.candidates.top_k" = tracing::field::Empty,
            ),
            Self::WorkerAdmission => common_span!(
                "worker.admission",
                "dynamo.request.attempt" = 0_u64,
                "dynamo.lifecycle.capture.state" = "recorded",
                "dynamo.lifecycle.detail_schema" = "admission.v1",
                "dynamo.worker.admission.transport" = tracing::field::Empty,
                "dynamo.worker.admission.payload.bytes" = tracing::field::Empty,
                "dynamo.worker.admission.result" = tracing::field::Empty,
            ),
            Self::RequestDispatch => common_span!(
                "request.dispatch",
                "dynamo.request.attempt" = 0_u64,
                "dynamo.lifecycle.capture.state" = "recorded",
                "dynamo.lifecycle.detail_schema" = "dispatch.v1",
                "dynamo.dispatch.destination.worker.id" = tracing::field::Empty,
                "dynamo.dispatch.destination.dp.rank" = tracing::field::Empty,
                "dynamo.dispatch.route" = tracing::field::Empty,
                "dynamo.dispatch.result" = tracing::field::Empty,
            ),
            Self::WorkerOperation => common_span!("worker.operation"),
            Self::WorkerOperationEncode => common_span!("worker.operation.encode"),
            Self::WorkerOperationPrefill => common_span!("worker.operation.prefill"),
            Self::WorkerOperationDecode => common_span!("worker.operation.decode"),
            Self::ResponseStreaming => common_span!(
                "response.streaming",
                "dynamo.stage.outcome" = tracing::field::Empty,
                "dynamo.stream.events" = tracing::field::Empty,
            ),
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

    /// Investigation mode permits bounded, per-request decision detail.
    pub fn is_investigation_mode(&self) -> bool {
        self.enabled
            && self
                .identity
                .as_ref()
                .is_some_and(|identity| identity.mode == "investigation")
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

    /// Observe a frontend stage without per-event tracing or payload capture.
    /// Detailed state is written once, on drop, and only in investigation mode.
    #[must_use]
    pub fn observe_stage(&self, stage: LifecycleStage) -> LifecycleStageObservation {
        LifecycleStageObservation {
            span: self.start(stage),
            detailed: self.is_investigation_mode(),
            outcome: None,
            checkpoint: None,
            events: matches!(stage, LifecycleStage::ResponseStreaming).then_some(0),
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

/// Bounded frontend stage state. `abandoned` means dropped without an observed
/// outcome, not necessarily client cancellation; consult the request terminal.
/// Event counts describe SSE items yielded by the frontend, not tokens or client
/// receipt. No per-event clock reads, allocations, or tracing calls are needed.
pub struct LifecycleStageObservation {
    span: Span,
    detailed: bool,
    outcome: Option<TerminalOutcome>,
    checkpoint: Option<&'static str>,
    events: Option<u64>,
}

impl LifecycleStageObservation {
    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn checkpoint(&mut self, checkpoint: &'static str) {
        if self.detailed {
            self.checkpoint = Some(checkpoint);
        }
    }

    pub fn finish(&mut self, outcome: TerminalOutcome) {
        if self.detailed {
            self.outcome = Some(outcome);
        }
    }

    pub fn observe_event(&mut self) {
        if self.detailed
            && let Some(events) = &mut self.events
        {
            *events = events.saturating_add(1);
        }
    }
}

impl Drop for LifecycleStageObservation {
    fn drop(&mut self) {
        if !self.detailed {
            return;
        }
        self.span.record(
            "dynamo.stage.outcome",
            self.outcome.map_or("abandoned", TerminalOutcome::as_str),
        );
        if let Some(checkpoint) = self.checkpoint {
            self.span.record("dynamo.stage.checkpoint", checkpoint);
        }
        if let Some(events) = self.events {
            self.span.record("dynamo.stream.events", events);
        }
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

    #[derive(Default)]
    struct DetailFields(Vec<(String, String)>);

    impl tracing::field::Visit for DetailFields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.push((field.name().to_owned(), format!("{value:?}")));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.push((field.name().to_owned(), value.to_owned()));
        }
    }

    struct DetailCapture(Arc<Mutex<DetailFields>>);

    impl<S: Subscriber> Layer<S> for DetailCapture {
        fn on_record(
            &self,
            _id: &tracing::Id,
            values: &tracing::span::Record<'_>,
            _ctx: Context<'_, S>,
        ) {
            values.record(&mut *self.0.lock().unwrap());
        }
    }

    #[test]
    fn lifecycle_stage_details_are_bounded_and_investigation_only() {
        for stage in [
            LifecycleStage::RequestPreprocessing,
            LifecycleStage::ResponseStreaming,
        ] {
            for mode in ["core", "investigation"] {
                for outcome in [
                    None,
                    Some(TerminalOutcome::Success),
                    Some(TerminalOutcome::Failed),
                ] {
                    let captured = Arc::new(Mutex::new(DetailFields::default()));
                    let subscriber =
                        tracing_subscriber::registry().with(DetailCapture(captured.clone()));
                    let _guard = tracing::subscriber::set_default(subscriber);
                    let mut identity = LifecycleIdentity::new(
                        Some("request".into()),
                        LifecycleOperationRole::Frontend,
                    );
                    identity.mode = mode;
                    let trace = LifecycleTrace::enabled(identity);
                    let mut observation = trace.observe_stage(stage);
                    let expected = if stage == LifecycleStage::RequestPreprocessing {
                        observation.checkpoint("preprocess_request");
                        ("dynamo.stage.checkpoint", "preprocess_request")
                    } else {
                        observation.observe_event();
                        observation.observe_event();
                        ("dynamo.stream.events", "2")
                    };
                    if let Some(outcome) = outcome {
                        observation.finish(outcome);
                    }
                    assert!(captured.lock().unwrap().0.is_empty(), "record only on drop");
                    drop(observation);
                    let fields = &captured.lock().unwrap().0;
                    if mode == "core" {
                        assert!(fields.is_empty());
                    } else {
                        assert_eq!(fields.len(), 2);
                        assert!(fields.contains(&(
                            "dynamo.stage.outcome".into(),
                            outcome.map_or("abandoned", TerminalOutcome::as_str).into()
                        )));
                        assert!(fields.contains(&(expected.0.into(), expected.1.into())));
                    }
                }
            }
        }
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
