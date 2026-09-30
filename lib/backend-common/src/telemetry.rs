// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers for engine-author observability.
//!
//! For **static** span names use `tracing::info_span!` directly — it nests
//! under the framework's `engine.generate` parent via the runtime's
//! `.instrument(...)` chain.
//!
//! For **dynamic** span names use [`start_span`] — `tracing` requires
//! compile-time names, so this helper goes through OTel directly while
//! still inheriting the bridged parent context.
//!
//! Both paths land in the same OTel trace tree.

use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span as OtelSpan, TraceContextExt, Tracer};
use opentelemetry::{KeyValue, Value, global};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Open an OTel span under the currently-active `tracing` span (typically
/// `engine.generate`). Use this when the span name is computed at runtime;
/// for compile-time names use `tracing::info_span!`.
///
/// Returns a no-op guard when the bridge layer isn't installed (no OTel
/// context on the parent). The guard ends the span on drop.
pub fn start_span(name: impl Into<String>) -> SpanGuard {
    let parent_ctx = tracing::Span::current().context();
    if !parent_ctx.span().span_context().is_valid() {
        return SpanGuard { span: None };
    }
    let tracer = global::tracer("dynamo");
    let span = tracer
        .span_builder(name.into())
        .start_with_context(&tracer, &parent_ctx);
    SpanGuard { span: Some(span) }
}

/// RAII guard for a dynamic-name OTel span. Ends the span when dropped.
/// All methods are silent no-ops when the underlying span is absent (bridge
/// not installed, or already closed).
pub struct SpanGuard {
    span: Option<BoxedSpan>,
}

impl SpanGuard {
    /// Set an attribute on this span. Accepts any name; OTel imposes no
    /// pre-declaration constraint.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        if let Some(span) = self.span.as_mut() {
            span.set_attribute(KeyValue::new(key.into(), value.into()));
        }
    }

    /// Emit a structured event on this span.
    pub fn add_event(&mut self, name: impl Into<String>, attrs: Vec<KeyValue>) {
        if let Some(span) = self.span.as_mut() {
            span.add_event(name.into(), attrs);
        }
    }

    /// End the span. Idempotent. Called automatically on drop.
    pub fn close(&mut self) {
        if let Some(mut span) = self.span.take() {
            span.end();
        }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the bridge layer installed, `start_span` returns a no-op
    /// guard and all method calls are silent. Asserts no panic and that
    /// `close()` is idempotent (called explicitly then again via drop).
    #[test]
    fn start_span_is_noop_without_bridge() {
        let mut guard = start_span("test_dynamic");
        guard.set_attribute("k", "v");
        guard.add_event("e", vec![]);
        guard.close();
        guard.close(); // idempotent
    }
}
