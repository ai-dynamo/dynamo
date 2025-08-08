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

//! NATS client Prometheus metrics constants
//!
//! This module provides centralized Prometheus metric name constants for NATS client metrics
//! to ensure consistency and avoid duplication across the codebase.

/// NATS client Prometheus metric names
pub mod nats {
    /// Prefix for all NATS client metrics
    pub const PREFIX: &str = "nats_client";

    /// Total number of bytes received by NATS client
    pub const IN_BYTES: &str = "nats_client_in_bytes";

    /// Total number of bytes sent by NATS client
    pub const OUT_BYTES: &str = "nats_client_out_bytes";

    /// Total number of messages received by NATS client
    pub const IN_MESSAGES: &str = "nats_client_in_messages";

    /// Total number of messages sent by NATS client
    pub const OUT_MESSAGES: &str = "nats_client_out_messages";

    /// Total number of connections established by NATS client
    pub const CONNECTS: &str = "nats_client_connects";

    /// Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)
    pub const CONNECTION_STATE: &str = "nats_client_connection_state";
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
