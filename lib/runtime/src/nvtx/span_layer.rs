// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `tracing` layer that mirrors request-scoped spans into NVTX start/end ranges.
//!
//! See the "Request-scoped annotations" section of [`crate::nvtx`] for the contract.
//!
//! A span is mirrored when it has a request id: its own `request_id` /
//! `dynamo.request.id` field, else the one inherited from its parent span. The id and its
//! [`rid_key`] are remembered in the span's extensions so descendants inherit them. The
//! range message is the stage only (span name, plus `:<phase>`) and its payload is the key.
//! The NVTX range starts in `on_new_span` and ends when the guard is removed from the
//! extensions in `on_close` (start/end ranges, so it does not matter which thread closes
//! the span).
//!
//! A span that takes its id from its own field while its parent does not already carry the
//! same id also emits the request identity mark (`dynamo.request rid=... trace=... xrid=...`),
//! once per request per process. When the span has a valid `trace_id` field (an inbound
//! `traceparent`), the mark is emitted before its range. Otherwise the trace id is the one
//! OpenTelemetry generates for the span, which exists only once the span has been entered,
//! so the mark is emitted when the span is first exited (or closed, if it never was).
//!
//! The layer only reads `OtelData` after the span's first enter, when the OpenTelemetry
//! layer has already built the span's context. Building it earlier, in `on_new_span`,
//! would break `span.set_parent(..)`, which Dynamo calls right after creating request spans.

use std::sync::Arc;

use tracing::field::{Field, Visit};
use tracing::span::Attributes;
use tracing::{Id, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, Filter};
use tracing_subscriber::registry::LookupSpan;

use super::{NvtxRangeGuard, identity_mark_unchecked, identity_message, rid_key};

/// Placeholder some lifecycle spans record before the real id is known.
const UNKNOWN_RID: &str = "unknown";

/// Where mirrored ranges and identity marks go. `Range` ends the range when dropped.
pub(crate) trait RangeSink: Send + Sync + 'static {
    type Range: Send + Sync + 'static;
    /// Start a range with message `stage` and payload `key`.
    fn start(&self, stage: &str, key: u64) -> Self::Range;
    /// Emit the identity mark `message` with payload `key`.
    fn identity(&self, message: &str, key: u64);
}

/// Production sink: NVTX FFI, bypassing the runtime flag (the layer is installed only when
/// `DYN_NVTX` is on, before `nvtx::init` has stored it).
pub(crate) struct NvtxSink;

impl RangeSink for NvtxSink {
    type Range = NvtxRangeGuard;

    fn start(&self, stage: &str, key: u64) -> NvtxRangeGuard {
        NvtxRangeGuard::new_with_payload_unchecked(stage, key)
    }

    fn identity(&self, message: &str, key: u64) {
        identity_mark_unchecked(message, key);
    }
}

/// Request id of a span and its key, stored so descendants inherit them.
struct RequestScope {
    rid: Arc<str>,
    key: u64,
}

/// The live NVTX range of a mirrored span.
struct MirroredRange<R>(#[allow(dead_code)] R);

/// An identity mark waiting for the span's OpenTelemetry trace id.
struct PendingIdentity {
    xrid: Option<String>,
}

/// Fields of interest, collected without allocating for anything else.
#[derive(Default)]
struct RequestFields {
    rid: Option<String>,
    trace: Option<String>,
    xrid: Option<String>,
    phase: Option<String>,
}

impl RequestFields {
    fn slot(&mut self, name: &str) -> Option<&mut Option<String>> {
        match name {
            "request_id" | "dynamo.request.id" => Some(&mut self.rid),
            "trace_id" => Some(&mut self.trace),
            "x_request_id" => Some(&mut self.xrid),
            "phase" => Some(&mut self.phase),
            _ => None,
        }
    }
}

impl Visit for RequestFields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if let Some(slot) = self.slot(field.name()) {
            *slot = Some(value.to_owned());
        }
    }

    // `%x` arrives wrapped in `DisplayValue` and `?x` as the Debug form; for the
    // string-like values read here both print the text itself.
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if let Some(slot) = self.slot(field.name()) {
            *slot = Some(format!("{value:?}"));
        }
    }
}

fn is_trace_id(value: &str) -> bool {
    value.len() == 32
        && value.bytes().all(|b| b.is_ascii_hexdigit())
        && value.bytes().any(|b| b != b'0')
}

/// Mirrors request-scoped spans into `S`'s ranges.
pub(crate) struct NvtxSpanLayer<K: RangeSink> {
    sink: K,
}

impl<K: RangeSink> NvtxSpanLayer<K> {
    pub(crate) fn new(sink: K) -> Self {
        Self { sink }
    }
}

impl<S, K> Layer<S> for NvtxSpanLayer<K>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    K: RangeSink,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut fields = RequestFields::default();
        attrs.record(&mut fields);

        let Some(span) = ctx.span(id) else { return };
        let own_rid = fields
            .rid
            .as_deref()
            .filter(|rid| !rid.is_empty() && *rid != UNKNOWN_RID);
        // `parent()` resolves explicit and contextual parents alike, skipping any that this
        // layer's filter hid.
        let parent_scope = span.parent().and_then(|parent| {
            parent
                .extensions()
                .get::<RequestScope>()
                .map(|s| (s.rid.clone(), s.key))
        });
        let (scope, emit_identity) = match (own_rid, parent_scope) {
            (Some(rid), Some((parent_rid, key))) if *parent_rid == *rid => (
                RequestScope {
                    rid: parent_rid,
                    key,
                },
                false,
            ),
            (Some(rid), _) => (
                RequestScope {
                    key: rid_key(rid),
                    rid: Arc::from(rid),
                },
                true,
            ),
            (None, Some((rid, key))) => (RequestScope { rid, key }, false),
            (None, None) => return,
        };

        let mut pending_identity = None;
        if emit_identity {
            match fields.trace.as_deref().filter(|t| is_trace_id(t)) {
                Some(trace) => self.emit_identity(
                    &scope.rid,
                    scope.key,
                    &trace.to_ascii_lowercase(),
                    fields.xrid.as_deref(),
                ),
                None => {
                    pending_identity = Some(PendingIdentity {
                        xrid: fields.xrid.take(),
                    })
                }
            }
        }

        // Spans below INFO (enabled by a verbose DYN_LOG) pass the request id on to their
        // children but get no range of their own.
        if *span.metadata().level() > Level::INFO {
            let mut extensions = span.extensions_mut();
            extensions.insert(scope);
            if let Some(pending) = pending_identity {
                extensions.insert(pending);
            }
            return;
        }

        // Only a phase-qualified stage needs building; everything else borrows the
        // span's static name.
        let qualified;
        let stage: &str = match fields.phase.as_deref() {
            Some(phase) if !phase.is_empty() => {
                qualified = format!("{}:{}", span.name(), phase.to_ascii_lowercase());
                &qualified
            }
            _ => span.name(),
        };
        let range = self.sink.start(stage, scope.key);
        let mut extensions = span.extensions_mut();
        extensions.insert(scope);
        extensions.insert(MirroredRange(range));
        if let Some(pending) = pending_identity {
            extensions.insert(pending);
        }
    }

    // The OpenTelemetry layer builds the span's context on its first enter, after this
    // layer's `on_enter`, so the first exit is the earliest point the trace id is readable.
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            self.flush_identity(&span);
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            self.flush_identity(&span);
            // Dropping the guard ends the range exactly once.
            span.extensions_mut().remove::<MirroredRange<K::Range>>();
        }
    }
}

impl<K: RangeSink> NvtxSpanLayer<K> {
    fn emit_identity(&self, rid: &str, key: u64, trace: &str, xrid: Option<&str>) {
        let message =
            identity_message(rid, &[("trace", trace), ("xrid", xrid.unwrap_or_default())]);
        self.sink.identity(&message, key);
    }

    /// Emit a deferred identity mark, with the span's OpenTelemetry trace id when it has one.
    fn flush_identity<S>(&self, span: &tracing_subscriber::registry::SpanRef<'_, S>)
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        // Only the request's first span has a pending mark; every other exit stops here.
        let Some(pending) = span.extensions_mut().remove::<PendingIdentity>() else {
            return;
        };
        let extensions = span.extensions();
        let Some(scope) = extensions.get::<RequestScope>() else {
            return;
        };
        let trace = extensions
            .get::<tracing_opentelemetry::OtelData>()
            .and_then(tracing_opentelemetry::OtelData::trace_id)
            .map(|t| t.to_string())
            .filter(|t| is_trace_id(t));
        self.emit_identity(
            &scope.rid,
            scope.key,
            trace.as_deref().unwrap_or_default(),
            pending.xrid.as_deref(),
        );
    }
}

/// The production layer: NVTX ranges for request-scoped spans, behind `filter`.
///
/// Pass the console logger's filter. A filter that admits spans the other layers reject
/// would create those spans for NVTX alone, and one that disagrees with the others on a
/// callsite turns its cached `always`/`never` interest into `sometimes`, re-evaluating
/// every filter on every hit.
pub(crate) fn layer<S, F>(filter: F) -> impl Layer<S>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    F: Filter<S> + Send + Sync + 'static,
{
    NvtxSpanLayer::new(NvtxSink).with_filter(filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tracing::{debug_span, info_span};
    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::SubscriberExt;

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Ev {
        Start(usize, String, u64),
        Identity(String, u64),
        End(usize),
    }

    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<Ev>>>);

    struct RecRange(usize, Arc<Mutex<Vec<Ev>>>);

    impl Drop for RecRange {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(Ev::End(self.0));
        }
    }

    impl RangeSink for Recorder {
        type Range = RecRange;
        fn start(&self, stage: &str, key: u64) -> RecRange {
            let mut events = self.0.lock().unwrap();
            let n = events.len();
            events.push(Ev::Start(n, stage.to_owned(), key));
            RecRange(n, self.0.clone())
        }
        fn identity(&self, message: &str, key: u64) {
            self.0
                .lock()
                .unwrap()
                .push(Ev::Identity(message.to_owned(), key));
        }
    }

    impl Recorder {
        fn events(&self) -> Vec<Ev> {
            self.0.lock().unwrap().clone()
        }
        /// `(stage, key)` of every range start.
        fn starts(&self) -> Vec<(String, u64)> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Ev::Start(_, stage, key) => Some((stage, key)),
                    _ => None,
                })
                .collect()
        }
        /// `(message, key)` of every identity mark.
        fn identities(&self) -> Vec<(String, u64)> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Ev::Identity(m, key) => Some((m, key)),
                    _ => None,
                })
                .collect()
        }
        fn ends(&self) -> usize {
            self.events()
                .iter()
                .filter(|e| matches!(e, Ev::End(_)))
                .count()
        }
    }

    fn pair(stage: &str, rid: &str) -> (String, u64) {
        (stage.to_owned(), rid_key(rid))
    }

    /// The layer behind a filter admitting everything, as with `DYN_LOG=trace`.
    fn with_layer<R>(f: impl FnOnce(&Recorder) -> R) -> R {
        let rec = Recorder::default();
        let subscriber = Registry::default().with(
            NvtxSpanLayer::new(rec.clone())
                .with_filter(tracing_subscriber::filter::LevelFilter::TRACE),
        );
        tracing::subscriber::with_default(subscriber, || f(&rec))
    }

    const TRACE: &str = "0af7651916cd43dd8448eb211c80319c";

    #[test]
    fn own_request_id_field_both_names() {
        with_layer(|rec| {
            let a = "r1".to_string();
            let _a = info_span!("http-request", request_id = %a);
            let _b = info_span!("request.lifecycle", dynamo.request.id = "r2");
            assert_eq!(
                rec.starts(),
                [pair("http-request", "r1"), pair("request.lifecycle", "r2")]
            );
            // Without a trace id the mark waits for the first exit.
            assert!(rec.identities().is_empty());
            _a.in_scope(|| ());
            _b.in_scope(|| ());
            assert_eq!(
                rec.identities(),
                [
                    ("dynamo.request rid=r1".to_owned(), rid_key("r1")),
                    ("dynamo.request rid=r2".to_owned(), rid_key("r2")),
                ]
            );
        });
    }

    #[test]
    fn inherits_through_explicit_and_contextual_parents() {
        with_layer(|rec| {
            let root = info_span!("root", request_id = "r1");
            let explicit = info_span!(parent: &root, "explicit");
            let contextual = root.in_scope(|| info_span!("contextual"));
            let grandchild = info_span!(parent: &explicit, "grand");
            assert_eq!(
                rec.starts(),
                [
                    pair("root", "r1"),
                    pair("explicit", "r1"),
                    pair("contextual", "r1"),
                    pair("grand", "r1"),
                ]
            );
            assert_eq!(rec.identities().len(), 1, "only the root emits identity");
            drop((grandchild, contextual, explicit, root));
        });
    }

    #[test]
    fn child_with_same_own_rid_does_not_repeat_identity_but_different_rid_does() {
        with_layer(|rec| {
            let root = info_span!("http-request", request_id = "r1", trace_id = TRACE);
            let same = info_span!(parent: &root, "request.lifecycle", request_id = "r1");
            let route = info_span!(parent: &same, "kv_router.route_request", request_id = "r1");
            assert_eq!(
                rec.identities(),
                [(
                    format!("dynamo.request rid=r1 trace={TRACE}"),
                    rid_key("r1")
                )]
            );

            let other = info_span!(parent: &root, "other", request_id = "r2");
            other.in_scope(|| ());
            assert_eq!(rec.identities().len(), 2);
            assert_eq!(
                rec.identities()[1],
                ("dynamo.request rid=r2".to_owned(), rid_key("r2"))
            );
            assert_eq!(rec.starts().last(), Some(&pair("other", "r2")));
            drop((other, route, same, root));
        });
    }

    #[test]
    fn identity_precedes_its_range_when_the_trace_id_is_known() {
        with_layer(|rec| {
            let _a = info_span!("root", request_id = "r1", trace_id = TRACE);
            let events = rec.events();
            assert!(matches!(events[0], Ev::Identity(..)));
            assert!(matches!(events[1], Ev::Start(..)));
        });
    }

    #[test]
    fn identity_without_trace_id_waits_for_first_exit_and_is_emitted_once() {
        with_layer(|rec| {
            let span = info_span!("root", request_id = "r1");
            assert_eq!(rec.events(), [Ev::Start(0, "root".into(), rid_key("r1"))]);
            span.in_scope(|| ());
            span.in_scope(|| ());
            drop(span);
            assert_eq!(
                rec.events(),
                [
                    Ev::Start(0, "root".into(), rid_key("r1")),
                    Ev::Identity("dynamo.request rid=r1".into(), rid_key("r1")),
                    Ev::End(0),
                ]
            );
        });
    }

    #[test]
    fn identity_of_a_never_entered_span_is_emitted_on_close() {
        with_layer(|rec| {
            drop(info_span!("root", request_id = "r1"));
            assert_eq!(
                rec.identities(),
                [("dynamo.request rid=r1".to_owned(), rid_key("r1"))]
            );
        });
    }

    /// The production order: NVTX layer below the OpenTelemetry layer. A span with no
    /// inbound trace id gets the one OpenTelemetry generated for it.
    #[test]
    fn identity_carries_the_generated_opentelemetry_trace_id() {
        use opentelemetry::trace::{TraceContextExt, TracerProvider};
        use tracing_opentelemetry::OpenTelemetrySpanExt;

        let rec = Recorder::default();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
        let subscriber = Registry::default()
            .with(
                NvtxSpanLayer::new(rec.clone())
                    .with_filter(tracing_subscriber::filter::LevelFilter::TRACE),
            )
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        tracing::subscriber::with_default(subscriber, || {
            let span = info_span!("http-request", request_id = "r1");
            span.in_scope(|| ());
            let trace = span.context().span().span_context().trace_id().to_string();
            assert!(is_trace_id(&trace));
            assert_eq!(
                rec.identities(),
                [(
                    format!("dynamo.request rid=r1 trace={trace}"),
                    rid_key("r1")
                )]
            );
        });
    }

    #[test]
    fn unknown_and_empty_ids_are_ignored_but_parent_wins() {
        with_layer(|rec| {
            let _a = info_span!("a", dynamo.request.id = "unknown");
            let _b = info_span!("b", request_id = "");
            assert!(rec.events().is_empty());

            let root = info_span!("root", request_id = "r1");
            let _c = info_span!(parent: &root, "c", dynamo.request.id = "unknown");
            assert_eq!(rec.starts(), [pair("root", "r1"), pair("c", "r1")]);
            root.in_scope(|| ());
            assert_eq!(rec.identities().len(), 1);
        });
    }

    #[test]
    fn unscoped_spans_are_skipped() {
        with_layer(|rec| {
            let _a = info_span!("no_rid", model = "m");
            let _b = info_span!(parent: &_a, "child");
            assert!(rec.events().is_empty());
        });
    }

    #[test]
    fn debug_spans_are_not_mirrored_but_pass_the_id_on() {
        with_layer(|rec| {
            let _a = debug_span!("dbg", request_id = "r1");
            assert!(rec.starts().is_empty());
            _a.in_scope(|| ());
            assert_eq!(rec.identities().len(), 1, "identity is level-agnostic");

            let root = info_span!("root", request_id = "r2");
            let dbg = root.in_scope(|| debug_span!("dbg"));
            let _child = dbg.in_scope(|| info_span!("child"));
            assert_eq!(rec.starts(), [pair("root", "r2"), pair("child", "r2")]);
            assert_eq!(rec.identities().len(), 2);
        });
    }

    #[test]
    fn debug_root_with_info_child_emits_identity_once() {
        with_layer(|rec| {
            let dbg = debug_span!("dbg", request_id = "r1");
            let _child = info_span!(parent: &dbg, "child", request_id = "r1");
            _child.in_scope(|| ());
            dbg.in_scope(|| ());
            assert_eq!(rec.identities().len(), 1);
            assert_eq!(rec.starts(), [pair("child", "r1")]);
        });
    }

    #[test]
    fn phase_qualifier_trace_and_xrid() {
        #[derive(Debug)]
        enum Phase {
            Prefill,
        }
        with_layer(|rec| {
            let xrid: Option<String> = Some("client id 7".to_string());
            let _a = info_span!(
                "kv_router.route_request",
                request_id = "r1",
                phase = ?Phase::Prefill,
                trace_id = TRACE,
                x_request_id = xrid,
            );
            let none: Option<String> = None;
            let _b = info_span!("worker", request_id = "r2", x_request_id = none);
            let _c = info_span!("badtrace", request_id = "r3", trace_id = "nothex");
            let _d = info_span!("zerotrace", request_id = "r4", trace_id = "0".repeat(32));
            assert_eq!(
                rec.starts(),
                [
                    pair("kv_router.route_request:prefill", "r1"),
                    pair("worker", "r2"),
                    pair("badtrace", "r3"),
                    pair("zerotrace", "r4"),
                ]
            );
            for span in [&_b, &_c, &_d] {
                span.in_scope(|| ());
            }
            let ids: Vec<String> = rec.identities().into_iter().map(|(m, _)| m).collect();
            assert_eq!(
                ids,
                [
                    format!("dynamo.request rid=r1 trace={TRACE} xrid=client_id_7"),
                    "dynamo.request rid=r2".to_string(),
                    "dynamo.request rid=r3".to_string(),
                    "dynamo.request rid=r4".to_string(),
                ]
            );
        });
    }

    #[test]
    fn range_ends_exactly_once_on_close() {
        with_layer(|rec| {
            let span = info_span!("s", request_id = "r1");
            assert_eq!(rec.ends(), 0);
            let _guard = span.enter();
            drop(_guard);
            assert_eq!(rec.ends(), 0, "entering/exiting must not end the range");
            drop(span);
            assert_eq!(
                rec.events(),
                [
                    Ev::Start(0, "s".into(), rid_key("r1")),
                    Ev::Identity("dynamo.request rid=r1".into(), rid_key("r1")),
                    Ev::End(0)
                ]
            );
        });
    }

    #[test]
    fn span_closed_from_another_thread() {
        with_layer(|rec| {
            let span = info_span!("s", request_id = "r1");
            std::thread::spawn(move || drop(span)).join().unwrap();
            assert_eq!(rec.ends(), 1);
        });
    }
}
