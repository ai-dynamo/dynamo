// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod capacity;
mod config;
mod registration;
mod strategy;

pub use capacity::{WatchWorkerCapacity, WorkerCapacity, WorkerCapacityProvider};
pub use config::{ConfigError, ThunderAgentConfig};
pub use registration::{RegistrationError, register_builtin_strategies};
pub use strategy::ThunderAgent;

pub const STRATEGY_NAME: &str = "session_aware";
