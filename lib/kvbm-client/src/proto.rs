// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! tonic-generated proto bindings for `kvbm_service.v1`.

#[allow(clippy::all)]
pub mod v1 {
    tonic::include_proto!("kvbm_service.v1");
}

pub use v1::{
    Accepted, Event, Heartbeat, KvbmInstance, LayoutMode, MlaPlaceholder, RegisterRequest,
    RegistrationInstance, ServerShutdownInitiated, ServerShutdownTimedOut, ServiceMode, event,
    kvbm_service_client::KvbmServiceClient as RawClient, layout_mode, registration_instance,
};
