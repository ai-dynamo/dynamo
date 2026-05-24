// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! tonic-generated proto bindings.
//!
//! The generated module name follows the proto `package` (`kvbm_service.v1`)
//! and is re-exported as `v1` here so callers write `proto::v1::Foo`.

#[allow(clippy::all)]
pub mod v1 {
    tonic::include_proto!("kvbm_service.v1");
}

pub use v1::{
    Accepted, Event, Heartbeat, KvbmInstance, LayoutMode, MlaPlaceholder, RegisterRequest,
    RegistrationInstance, ServerShutdownInitiated, ServerShutdownTimedOut, ServiceMode, event,
    kvbm_service_client::KvbmServiceClient,
    kvbm_service_server::{KvbmService as KvbmServiceTrait, KvbmServiceServer},
    layout_mode, registration_instance,
};
