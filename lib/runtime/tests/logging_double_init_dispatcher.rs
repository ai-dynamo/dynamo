// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Double-init behavior when a global tracing dispatcher is already set.
//!
//! Embedded scenario: a host application (e.g. a Python process with its own
//! telemetry) installed a global tracing subscriber before dynamo's
//! `logging::init()` ran. Dynamo's subscriber cannot install, so `init()`
//! must not perform the global OTel side effects either: swapping the global
//! tracer provider would orphan its batch-exporter threads and clobber the
//! host's provider. The provider it built must be left unregistered (and
//! thereby shut down on drop).
//!
//! Must stay in its own integration-test binary: it manipulates
//! process-global logger/dispatcher/provider state.

use opentelemetry::trace::{Span, Tracer};

#[test]
fn init_with_preexisting_dispatcher_skips_otel_side_effects() {
    // Simulate the host application owning tracing for this process.
    tracing::dispatcher::set_global_default(tracing::Dispatch::new(tracing_subscriber::registry()))
        .expect("first dispatcher install must succeed");

    // JSONL mode exercises the OTel branch with a local-only provider
    // (no OTLP exporter, no network).
    // SAFETY: single-threaded at this point; this binary owns the process.
    unsafe { std::env::set_var("DYN_LOGGING_JSONL", "1") };

    // Old `.init()` behavior would panic here; must warn and continue.
    dynamo_runtime::logging::init();

    // The subscriber install was skipped, so the global OTel provider must
    // still be the default noop provider: spans from the global tracer have
    // invalid (all-zero) contexts. A valid context means `init()` swapped in
    // its provider despite losing the init race — the leak/hijack bug.
    let mut span = opentelemetry::global::tracer("probe").start("probe-span");
    assert!(
        !span.span_context().is_valid(),
        "global tracer provider was swapped despite the subscriber install \
         being skipped (orphans exporter threads, hijacks the host's provider)"
    );
    span.end();
}
