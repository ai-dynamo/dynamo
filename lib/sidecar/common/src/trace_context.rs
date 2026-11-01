// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The current span's W3C Trace Context, for outbound engine RPCs.

/// The current span's W3C Trace Context as `(header, value)` pairs:
/// `traceparent`, then `tracestate` when the span carries one.
///
/// Empty when the runtime's distributed trace layer is not installed, which
/// is the case unless `OTEL_EXPORT_ENABLED` or JSONL logging is on. An engine
/// that receives no context starts no span for the request, or one of its
/// own; with the context it parents its spans on the sidecar's
/// `engine.generate` span, so the trace runs frontend, sidecar, engine.
pub fn trace_context_metadata() -> Vec<(&'static str, String)> {
    let Some(context) = dynamo_runtime::logging::get_distributed_tracing_context() else {
        return Vec::new();
    };
    let mut headers = Vec::with_capacity(2);
    headers.push(("traceparent", context.create_traceparent()));
    if let Some(tracestate) = context.tracestate {
        headers.push(("tracestate", tracestate));
    }
    headers
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::layer::SubscriberExt;

    use super::trace_context_metadata;

    #[test]
    fn follows_the_current_span() {
        let _default = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(dynamo_runtime::logging::DistributedTraceIdLayer),
        );
        assert!(trace_context_metadata().is_empty());

        let span = tracing::info_span!(
            "handle_payload",
            trace_id = "0af7651916cd43dd8448eb211c80319c",
            span_id = "b7ad6b7169203331",
            trace_flags = "01",
            tracestate = "vendor=value",
        );
        let _entered = span.enter();
        assert_eq!(
            trace_context_metadata(),
            [
                (
                    "traceparent",
                    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string()
                ),
                ("tracestate", "vendor=value".to_string()),
            ]
        );
    }
}
