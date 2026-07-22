// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod capacity;
mod config;
mod policy;
mod registration;

pub use capacity::{WatchWorkerCapacity, WorkerCapacity, WorkerCapacityProvider};
pub use config::{ConfigError, SessionAwareAdmissionControlConfig};
pub use policy::SessionAwareAdmissionControl;
pub use registration::{RegistrationError, register_builtin_policies};

pub const POLICY_NAME: &str = "session_aware";
