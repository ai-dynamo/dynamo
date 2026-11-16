// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod device_aware;
mod occupancy;
mod selector;
mod types;

pub(crate) use occupancy::get_or_create_routing_occupancy_state;
pub use occupancy::{OccupancyReservation, OccupancySelection, RoutingOccupancyState};
pub use selector::{InstanceSelector, ModeSelectors};
pub use types::RouteTarget;
pub(crate) use types::{AdmissionKind, RouteCandidate, RouteContext, RouteDecision, RouteDevice};
