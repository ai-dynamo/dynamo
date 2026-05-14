// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
pub(crate) mod components;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod event_sink;
pub(crate) mod events;
mod progress;
pub(crate) mod runtime_utils;
pub(crate) mod single;
pub(crate) mod state;
pub use event_sink::CompositeEventSink;
pub use event_sink::EventSink;

pub use agg::AggRuntime;
pub use components::{AdmissionQueue, ReplayMode, StreamingAdmission};
pub use disagg::DisaggRuntime;

pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, simulate_concurrency, simulate_concurrency_disagg,
    simulate_concurrency_workload, simulate_concurrency_workload_disagg, simulate_trace,
    simulate_trace_disagg, simulate_trace_workload, simulate_trace_workload_disagg,
};
