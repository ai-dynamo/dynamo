// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Double-init behavior when a `log`-crate logger is already installed.
//!
//! This is the scenario that originally broke CI: a scoped
//! `SubscriberInitExt::set_default()` elsewhere in the process permanently
//! installs the global `LogTracer`, then `logging::init()` runs. The global
//! tracing dispatcher slot is still free, so dynamo's subscriber (including
//! the OTel layer) installs successfully — only the `log` bridge cannot be
//! re-registered. `init()` must treat this as a successful install: no panic,
//! and the global OTel tracer provider must be swapped in so direct
//! `opentelemetry::global::tracer()` users get real (non-noop) spans.
//!
//! Must stay in its own integration-test binary: it manipulates
//! process-global logger/dispatcher/provider state.

use opentelemetry::trace::{Span, Tracer};

#[test]
fn init_with_preexisting_log_logger_still_installs_otel() {
    // Simulate the poisoned state: a global `log` logger already registered.
    tracing_log::LogTracer::init().expect("first LogTracer install must succeed");

    // JSONL mode exercises the OTel branch with a local-only provider
    // (no OTLP exporter, no network).
    // SAFETY: single-threaded at this point; this binary owns the process.
    unsafe { std::env::set_var("DYN_LOGGING_JSONL", "1") };

    // Old `.init()` behavior would panic here; the subscriber itself must
    // install fine since the dispatcher slot is free.
    dynamo_runtime::logging::init();

    // The install succeeded, so the global OTel provider must have been
    // swapped in: spans from the global tracer carry real (valid) contexts.
    // A noop provider would yield an all-zero, invalid span context.
    let mut span = opentelemetry::global::tracer("probe").start("probe-span");
    assert!(
        span.span_context().is_valid(),
        "global tracer provider was not registered even though the \
         subscriber installed (log-bridge failure must not gate OTel side effects)"
    );
    span.end();
}
