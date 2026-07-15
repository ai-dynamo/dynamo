// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod session_aware;

pub use session_aware::{
    ConfigError, RegistrationError, STRATEGY_NAME, SessionAwareAdmissionControl,
    SessionAwareAdmissionControlConfig, WatchWorkerCapacity, WorkerCapacity,
    WorkerCapacityProvider, register_builtin_strategies,
};
