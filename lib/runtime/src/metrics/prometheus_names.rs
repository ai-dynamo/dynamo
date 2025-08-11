// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Prometheus metric name constants
//!
//! This module provides centralized Prometheus metric name constants for various components
//! to ensure consistency and avoid duplication across the codebase.

/// NATS Prometheus metric names
pub mod nats {
    /// Prefix for all NATS client metrics
    pub const PREFIX: &str = "nats_";

    /// ===== DistributedRuntime metrics =====
    /// Total number of bytes received by NATS client
    pub const IN_BYTES: &str = "nats_in_bytes";

    /// Total number of bytes sent by NATS client
    pub const OUT_BYTES: &str = "nats_out_bytes";

    /// Total number of messages received by NATS client
    pub const IN_MESSAGES: &str = "nats_in_messages";

    /// Total number of messages sent by NATS client
    pub const OUT_MESSAGES: &str = "nats_out_messages";

    /// Total number of connections established by NATS client
    pub const CONNECTS: &str = "nats_connects";

    /// Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)
    pub const CONNECTION_STATE: &str = "nats_connection_state";

    /// ===== Component metrics (ordered to match NatsStatsMetrics fields) =====
    /// Average processing time in milliseconds (maps to: average_processing_time in nanoseconds)
    pub const AVG_PROCESSING_MS: &str = "nats_avg_processing_time_ms";

    /// Total errors across all endpoints (maps to: num_errors)
    pub const TOTAL_ERRORS: &str = "nats_total_errors";

    /// Total requests across all endpoints (maps to: num_requests)
    pub const TOTAL_REQUESTS: &str = "nats_total_requests";

    /// Total processing time in milliseconds (maps to: processing_time in nanoseconds)
    pub const TOTAL_PROCESSING_MS: &str = "nats_total_processing_time_ms";

    /// Number of active services (derived from ServiceSet.services)
    pub const ACTIVE_SERVICES: &str = "nats_active_services";

    /// Number of active endpoints (derived from ServiceInfo.endpoints)
    pub const ACTIVE_ENDPOINTS: &str = "nats_active_endpoints";
}

/// All NATS client Prometheus metric names as an array for iteration/validation
#[allow(dead_code)]
pub const ALL_NATS_METRICS: &[&str] = &[
    nats::CONNECTION_STATE,
    nats::CONNECTS,
    nats::IN_BYTES,
    nats::IN_MESSAGES,
    nats::OUT_BYTES,
    nats::OUT_MESSAGES,
];

/// All component service Prometheus metric names as an array for iteration/validation
/// (ordered to match NatsStatsMetrics fields)
#[allow(dead_code)]
pub const ALL_COMPONENT_SERVICE_METRICS: &[&str] = &[
    nats::AVG_PROCESSING_MS,   // maps to: average_processing_time (nanoseconds)
    nats::TOTAL_ERRORS,        // maps to: num_errors
    nats::TOTAL_REQUESTS,      // maps to: num_requests
    nats::TOTAL_PROCESSING_MS, // maps to: processing_time (nanoseconds)
    nats::ACTIVE_SERVICES,     // derived from ServiceSet.services
    nats::ACTIVE_ENDPOINTS,    // derived from ServiceInfo.endpoints
];
