// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Names of the environment variables that configure the frontend request metrics.
//!
//! Defined here so the request metrics and the `dynamo-runtime` environment registry
//! (`config::environment_names::llm::metrics`) name the same strings. The registry documents
//! each variable.

pub const DYN_METRICS_PREFIX: &str = "DYN_METRICS_PREFIX";
pub const DYN_METRICS_REQUEST_DURATION: &str = "DYN_METRICS_REQUEST_DURATION";
pub const DYN_METRICS_INPUT_SEQUENCE: &str = "DYN_METRICS_INPUT_SEQUENCE";
pub const DYN_METRICS_OUTPUT_SEQUENCE: &str = "DYN_METRICS_OUTPUT_SEQUENCE";
pub const DYN_METRICS_TTFT: &str = "DYN_METRICS_TTFT";
pub const DYN_METRICS_ITL: &str = "DYN_METRICS_ITL";
pub const DYN_METRICS_ITL_BUCKETS: &str = "DYN_METRICS_ITL_BUCKETS";
pub const DYN_METRICS_EMBEDDING_LATENCY: &str = "DYN_METRICS_EMBEDDING_LATENCY";

/// Deprecated prefix of the doubled histogram bucket variable names, still read as a fallback.
pub const DEPRECATED_HISTOGRAM_PREFIX: &str = "DYN_HISTOGRAM_";
