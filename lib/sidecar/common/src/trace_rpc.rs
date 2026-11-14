// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Included once per Tonic version (see `lib.rs`), so `tonic` resolves to the
// version the including module aliases.

/// Adds the current span's W3C Trace Context (`traceparent`, `tracestate`) to
/// `request`'s metadata, the standard gRPC carrier that OpenEngine and the
/// vLLM gRPC servers read. Sends nothing without a distributed trace context;
/// see [`crate::trace_context_metadata`].
pub fn inject_trace_context<T>(request: &mut tonic::Request<T>) {
    for (key, value) in crate::trace_context_metadata() {
        match value.parse::<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>() {
            Ok(value) => {
                request.metadata_mut().insert(key, value);
            }
            Err(_) => tracing::debug!(key, "dropping trace metadata that is not ASCII"),
        }
    }
}
